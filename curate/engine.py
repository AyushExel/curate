"""Two ways to turn a Signal into a column. Same result either way.

``local``  : in-process, streams batches through Lance ``add_columns``
             (or ``merge`` by row address when a ``where`` is set).
``geneva`` : registers the column on the table and runs a Geneva backfill:
             one Ray worker per GPU, checkpointed, ``where``-aware.
"""

from __future__ import annotations

import os
import time

import lance
import pyarrow as pa
import pyarrow.compute


class config:
    engine = "auto"  # "auto" | "local" | "geneva"
    concurrency = None  # geneva workers; default = #GPUs for gpu signals


def log(msg: str) -> None:
    print(f"curate | {msg}", flush=True)


def pick(engine: str | None) -> str:
    engine = engine or config.engine
    if engine != "auto":
        return engine
    try:
        import geneva  # noqa: F401

        return "geneva"
    except ImportError:
        return "local"


def write_signal(uri, name, sig, where=None, engine=None, concurrency=None) -> dict:
    engine = pick(engine)
    ds = lance.dataset(uri)
    n = ds.count_rows(filter=where)
    t0 = time.perf_counter()
    if engine == "geneva":
        _geneva(uri, name, sig, where, concurrency)
    else:
        _local(ds, name, sig, where)
    dt = time.perf_counter() - t0
    log(f"signal {name} <- {sig!r}: {n:,} rows in {dt:,.1f}s ({n / max(dt, 1e-9):,.0f} rows/s, {engine})")
    return {"rows": n, "seconds": round(dt, 2), "engine": engine}


def merge_columns(ds: lance.LanceDataset, rowaddrs: pa.Array, **columns: pa.Array) -> None:
    """Attach columns aligned with ``rowaddrs``. Rows not listed get null."""
    ds.merge(pa.table({"_rowaddr": rowaddrs, **columns}), left_on="_rowaddr", right_on="_rowaddr")


def _local(ds, name, sig, where):
    if where is None:

        def fn(batch: pa.RecordBatch) -> pa.RecordBatch:
            out = sig(*[batch.column(c) for c in sig.inputs])
            return pa.RecordBatch.from_arrays([out], names=[name])

        ds.add_columns(fn, read_columns=sig.inputs, batch_size=sig.batch_size)
        return
    addrs, outs = [], []
    scanner = ds.scanner(columns=sig.inputs, filter=where, with_row_address=True, batch_size=sig.batch_size)
    for b in scanner.to_batches():
        addrs.append(b.column("_rowaddr"))
        outs.append(sig(*[b.column(c) for c in sig.inputs]))
    new = pa.table({"_rowaddr": pa.concat_arrays(addrs), name: pa.concat_arrays(outs).cast(sig.out_type)})
    if name in ds.schema.names:
        # resuming a partial column: keep the rows already computed, then rewrite the column
        old = ds.to_table(columns=[name], with_row_address=True).select(["_rowaddr", name])
        old = old.filter(pa.compute.invert(pa.compute.is_in(old.column("_rowaddr"), new.column("_rowaddr"))))
        new = pa.concat_tables([old, new])
        ds.drop_columns([name])
    merge_columns(ds, new.column("_rowaddr"), **{name: new.column(name)})


# Geneva UDF bodies: hold the pickled signal, lazily run setup() on the worker.
# Geneva reads the annotated signature to know this is a batched (Array) UDF,
# so there is one small class per input arity.
class _Worker1:
    def __init__(self, sig):
        self.sig = sig

    def __call__(self, a: pa.Array) -> pa.Array:
        return self.sig(a)


class _Worker2(_Worker1):
    def __call__(self, a: pa.Array, b: pa.Array) -> pa.Array:
        return self.sig(a, b)


class _Worker3(_Worker1):
    def __call__(self, a: pa.Array, b: pa.Array, c: pa.Array) -> pa.Array:
        return self.sig(a, b, c)


_WORKERS = {1: _Worker1, 2: _Worker2, 3: _Worker3}


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def _geneva(uri, name, sig, where, concurrency):
    import geneva
    from geneva.transformer import udf

    root, table = os.path.split(uri.rstrip("/"))
    conn = geneva.connect(root)
    tbl = conn.open_table(table.removesuffix(".lance"))
    if len(sig.inputs) not in _WORKERS:
        raise ValueError(f"geneva engine supports 1-3 input columns, got {sig.inputs}; use engine='local'")
    worker = udf(
        data_type=sig.out_type,
        input_columns=sig.inputs,
        num_gpus=1 if sig.gpu else 0,
        batch_size=sig.batch_size,
    )(_WORKERS[len(sig.inputs)])(sig)
    if concurrency is None:
        concurrency = config.concurrency or (max(_gpu_count(), 1) if sig.gpu else max((os.cpu_count() or 4) // 2, 1))
    if name not in tbl.schema.names:
        tbl.add_columns({name: worker})
    with conn.local_ray_context():
        tbl.backfill(name, udf=worker, where=where, concurrency=concurrency)
