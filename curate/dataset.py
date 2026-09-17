"""A Dataset is a Lance table plus a SQL ``where``. That pair is the curated set."""

from __future__ import annotations

import json
import os

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import dedup as _dedup
from .engine import annotate, log, merge_columns, write_signal
from .signal import Signal


def load(uri: str, tag: str | None = None, version: int | None = None, where: str | None = None) -> "Dataset":
    """Open a Lance table (local, s3://, hf://...). ``tag`` restores a saved recipe."""
    if tag is None:
        return Dataset(uri, where=where, version=version)
    ds = lance.dataset(uri, version=tag)
    recipe = json.loads(ds.tags.list()[tag].get("metadata", {}).get("recipe", "{}"))
    return Dataset(uri, where=recipe.get("where"), version=tag, steps=recipe.get("steps"))


class Dataset:
    def __init__(self, uri, where=None, version=None, steps=None):
        self.uri = uri
        self.where = where
        self.steps = list(steps or [])
        # pinned to a version/tag until the first write; otherwise always latest,
        # so views and their parent see each other's new columns
        self._pinned = lance.dataset(uri, version=version) if version is not None else None

    @property
    def _ds(self) -> lance.LanceDataset:
        return self._pinned if self._pinned is not None else lance.dataset(self.uri)

    # ---- introspection -------------------------------------------------
    @property
    def schema(self) -> pa.Schema:
        return self._ds.schema

    @property
    def columns(self) -> list[str]:
        return self._ds.schema.names

    @property
    def version(self) -> int:
        return self._ds.version

    def __len__(self) -> int:
        return self._ds.count_rows(filter=self.where)

    def __repr__(self) -> str:
        w = f", where={self.where!r}" if self.where else ""
        return f"Dataset({len(self):,} rows, {len(self.columns)} cols, v{self.version}{w})"

    def head(self, n=5, columns=None) -> pa.Table:
        return self._ds.scanner(columns=columns, filter=self.where, limit=n).to_table()

    def to_table(self, columns=None, limit=None) -> pa.Table:
        return self._ds.scanner(columns=columns, filter=self.where, limit=limit).to_table()

    def to_pandas(self, columns=None, limit=None):
        return self.to_table(columns, limit).to_pandas()

    # ---- the four verbs --------------------------------------------------
    def signal(self, engine=None, concurrency=None, **signals: Signal) -> "Dataset":
        """Add one column per keyword. Computed only for rows matching ``where``."""
        for name, sig in signals.items():
            where = self.where
            if name in self.columns:
                # resume: only rows of this view that are still null
                where = f"{name} IS NULL" + (f" AND ({where})" if where else "")
                missing = self._ds.count_rows(filter=where)
                if missing == 0:
                    log(f"signal {name}: column exists and is complete, skipping (drop it to recompute)")
                    continue
                log(f"signal {name}: column exists, {missing:,} rows still null, resuming")
            write_signal(self.uri, name, sig, where=where, engine=engine, concurrency=concurrency)
            self._reopen()
        return self

    def filter(self, where: str) -> "Dataset":
        combined = f"({self.where}) AND ({where})" if self.where else where
        return Dataset(self.uri, where=combined, steps=self.steps + [{"op": "filter", "where": where}])

    def dedup(self, column, threshold: float | None = None, name="is_dup", **kw) -> "Dataset":
        """Exact dedup on a column, or semantic dedup (cosine > threshold) on an
        embedding column. Pass a Signal to embed first. Returns the view without dups."""
        if isinstance(column, Signal):
            emb = kw.pop("embedding_column", "embedding")
            self.signal(**{emb: column})
            column = emb
        if name in self.columns:
            raise ValueError(f"column {name!r} exists; pass name= or drop it")
        if threshold is None:
            info = _dedup.exact(self._ds, column, self.where, name)
            log(f"dedup {column} exact: {info['dups']:,} of {info['rows']:,} rows flagged")
        else:
            info = _dedup.semantic(self._ds, column, self.where, name, threshold, **kw)
        annotate(self._ds, name, {"op": "dedup", "column": column, "where": self.where, **info})
        self.steps.append({"op": "dedup", "name": name})
        self._reopen()
        return self.filter(f"NOT {name}")

    def sample(self, n: int, seed=0, by: str | None = None, name: str | None = None) -> "Dataset":
        """Random sample (or equal-per-group when ``by`` is set) as a bool column."""
        name = name or f"sample_{n}"
        if not name.isidentifier():
            raise ValueError(f"sample name {name!r} must be a plain identifier (it becomes a column used in SQL)")
        if name in self.columns:
            raise ValueError(f"column {name!r} exists; pass name=")
        t = self._ds.scanner(columns=[by] if by else [], filter=self.where, with_row_address=True).to_table()
        rng = np.random.default_rng(seed)
        m = t.num_rows
        if by is None:
            pick = rng.permutation(m)[:n]
        else:
            groups = t.column(by).to_numpy(zero_copy_only=False)
            order = rng.permutation(m)
            # rank within group after a shuffle, then take the lowest ranks round-robin
            g = groups[order]
            _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
            rank = np.zeros(m, dtype=np.int64)
            for j in range(len(counts)):
                rank[inv == j] = np.arange(counts[j])
            pick = order[np.argsort(rank, kind="stable")[:n]]
        keep = np.zeros(m, dtype=bool)
        keep[pick] = True
        merge_columns(self._ds, t.column("_rowaddr"), **{name: pa.array(keep)})
        log(f"sample {name}: kept {int(keep.sum()):,} of {m:,} rows" + (f" (stratified by {by})" if by else ""))
        annotate(self._ds, name, {"op": "sample", "n": n, "seed": seed, "by": by, "where": self.where})
        self.steps.append({"op": "sample", "name": name})
        self._reopen()
        return self.filter(name)

    def group(self, key: str, order: str | None = None) -> "Grouped":
        return Grouped(self, key, order)

    def drop(self, *columns: str) -> "Dataset":
        """Drop signal columns (to recompute with different settings)."""
        present = [c for c in columns if c in self.columns]
        if present:
            self._ds.drop_columns(present)
            self.steps.append({"op": "drop", "columns": present})
        return self

    # ---- reproducibility -------------------------------------------------
    @property
    def recipe(self) -> dict:
        """Everything needed to rebuild this view: the table version, the where,
        the lineage of this view, and how every curate-made column was computed."""
        return {"uri": self.uri, "version": self.version, "where": self.where, "steps": self.steps, "columns": self.provenance()}

    def provenance(self) -> dict:
        out = {}
        for f in self.schema:
            if f.metadata and b"curate" in f.metadata:
                out[f.name] = json.loads(f.metadata[b"curate"])
        return out

    def tag(self, name: str) -> dict:
        """Freeze this view: a Lance tag on the current version with the recipe in its
        metadata. Re-tagging moves the tag, so re-running a script is safe."""
        ds = self._ds
        if name in ds.tags.list():
            ds.tags.update(name, self.version)
        else:
            ds.tags.create(name, self.version)
        ds.tags.replace_metadata(name, {"recipe": json.dumps(self.recipe)})
        log(f"tag {name}: v{self.version}, {len(self):,} rows")
        return self.recipe

    # ---- analysis --------------------------------------------------------
    def stats(self, columns=None) -> dict:
        cols = columns or [f.name for f in self.schema if _summarizable(f.type)]
        out = {"rows": len(self)}
        for c in cols:
            out[c] = _summary(self._ds.to_table(columns=[c], filter=self.where).column(c))
        return out

    def search(self, query: str, column: str, limit=10, columns=None) -> pa.Table:
        """Full-text search inside the current view (builds an FTS index on first use)."""
        tbl = self._lancedb()
        if not any(column in (i.columns or []) for i in tbl.list_indices()):
            tbl.create_fts_index(column, replace=True)
        q = tbl.search(query, query_type="fts", fts_columns=column).limit(limit)
        if self.where:
            q = q.where(self.where, prefilter=True)
        if columns:
            q = q.select(columns)
        return q.to_arrow()

    # ---- out -----------------------------------------------------------------
    def torch(self, columns=None, **kw):
        """lancedb.streaming.StreamingDataset over this view: shuffled, resumable, rank-aware."""
        from lancedb.streaming import StreamingDataset

        if self.schema.metadata:
            # lancedb 0.38.0's permutation builder panics on tables with schema-level
            # metadata (any HuggingFace-converted table has a 'huggingface' key)
            raise RuntimeError(
                f"{self.uri} has schema-level metadata {list(self.schema.metadata)}, which crashes "
                "StreamingDataset in lancedb 0.38.0. Strip it (metadata-only commit, no data rewrite): "
                "lance.dataset(uri).replace_schema_metadata({})"
            )
        return StreamingDataset(self._lancedb(), columns=columns, filter=self.where, **kw)

    def export(self, path: str, columns=None) -> "Dataset":
        sc = self._ds.scanner(columns=columns, filter=self.where)
        # keep field metadata (column provenance), drop table-level metadata (see torch())
        lance.write_dataset(sc.to_batches(), path, schema=sc.projected_schema.remove_metadata(), mode="overwrite")
        return Dataset(path, steps=self.steps + [{"op": "export", "from": self.recipe}])

    # ---- internals -----------------------------------------------------------
    def _reopen(self):
        self._pinned = None

    def _lancedb(self):
        import lancedb

        root, name = os.path.split(self.uri.rstrip("/"))
        return lancedb.connect(root).open_table(name.removesuffix(".lance"))


class Grouped:
    """Rows grouped by a key (episode, clip...). Signals here see one group at
    a time and their scalar result is broadcast to every row of the group, so
    filters stay plain SQL on the frame table."""

    def __init__(self, ds: Dataset, key: str, order: str | None = None):
        self.ds, self.key, self.order = ds, key, order

    def __len__(self) -> int:
        return len(pc.unique(self.ds._ds.to_table(columns=[self.key], filter=self.ds.where).column(self.key)))

    def signal(self, **signals: Signal) -> "Grouped":
        for n in list(signals):
            if n in self.ds.columns:
                log(f"signal {n}: column exists, skipping")
                del signals[n]
        if not signals:
            return self
        cols = sorted({c for s in signals.values() for c in s.inputs} | {self.key} | ({self.order} if self.order else set()))
        t = self.ds._ds.scanner(columns=cols, filter=self.ds.where, with_row_address=True).to_table().combine_chunks()
        keys = t.column(self.key).to_numpy(zero_copy_only=False)
        idx = np.lexsort((t.column(self.order).to_numpy(zero_copy_only=False), keys)) if self.order else np.argsort(keys, kind="stable")
        t, keys = t.take(idx), keys[idx]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        sizes = np.diff(np.r_[starts, len(keys)])
        for name, sig in signals.items():
            sig.setup()
            arrays = [t.column(c) for c in sig.inputs] or [t.column(self.key)]
            vals = [sig.compute(*[a.slice(lo, n) for a in arrays]) for lo, n in zip(starts, sizes)]
            per_row = pa.array(np.repeat(np.asarray(vals), sizes)).cast(sig.out_type)
            merge_columns(self.ds._ds, t.column("_rowaddr"), **{name: per_row})
            log(f"signal {name} <- {sig!r}: {len(starts):,} groups, {len(keys):,} rows")
            annotate(self.ds._ds, name, {**sig.describe(), "group": self.key, "order": self.order, "rows": len(keys), "groups": len(starts), "where": self.ds.where})
            self.ds._reopen()
        return self

    def stats(self, columns) -> dict:
        t = self.ds._ds.to_table(columns=[self.key, *columns], filter=self.ds.where)
        t = t.group_by(self.key).aggregate([(c, "min") for c in columns])  # broadcast values: min == the value
        return {"groups": t.num_rows, **{c: _summary(t.column(f"{c}_min")) for c in columns}}

    def filter(self, where: str) -> Dataset:
        return self.ds.filter(where)


def _summarizable(t: pa.DataType) -> bool:
    return pa.types.is_boolean(t) or pa.types.is_integer(t) or pa.types.is_floating(t)


def _summary(arr) -> dict:
    t = arr.type
    nulls = arr.null_count
    if pa.types.is_boolean(t):
        return {"true": pc.sum(arr).as_py() or 0, "frac": round(pc.mean(pc.cast(arr, pa.float64())).as_py() or 0, 4), "nulls": nulls}
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        # ints stay ints so the numbers paste straight into a SQL filter
        cast = int if pa.types.is_integer(t) else (lambda v: round(v, 4))
        q = [cast(v) for v in pc.quantile(arr, q=[0.05, 0.25, 0.5, 0.75, 0.95]).to_pylist()]
        mm = pc.min_max(arr).as_py()
        return {"mean": round(pc.mean(arr).as_py(), 4), "min": mm["min"], "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4], "max": mm["max"], "nulls": nulls}
    vc = pc.value_counts(arr).to_pylist()
    top = sorted(vc, key=lambda r: -r["counts"])[:8]
    return {"distinct": len(vc), "top": {r["values"]: r["counts"] for r in top}, "nulls": nulls}
