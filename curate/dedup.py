"""Dedup writes a boolean column. Exact = hash of the value; semantic = SemDeDup
(k-means on the embedding, then cosine threshold inside each cluster, keep the
row closest to the centroid)."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pyarrow as pa

from .engine import log, merge_columns
from .signal import matrix


def _key(v) -> bytes:
    if isinstance(v, str):
        v = " ".join(v.split()).encode()
    return hashlib.md5(v).digest()


def exact(ds, column, where, name) -> dict:
    seen, addrs, flags = set(), [], []
    for b in ds.scanner(columns=[column], filter=where, with_row_address=True, batch_size=4096).to_batches():
        addrs.append(b.column("_rowaddr"))
        for v in b.column(column).to_pylist():
            k = _key(v)
            flags.append(k in seen)
            seen.add(k)
    merge_columns(ds, pa.concat_arrays(addrs), **{name: pa.array(flags)})
    return {"rows": len(flags), "dups": int(sum(flags))}


def _kmeans(x, k, iters=20):
    import torch

    c = x[torch.randperm(len(x), device=x.device)[:k]].clone()
    for _ in range(iters):
        assign = torch.cat([(chunk @ c.T).argmax(1) for chunk in x.split(65536)])
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = torch.nn.functional.normalize(x[m].mean(0), dim=0)
    return c, assign


def semantic(ds, column, where, name, threshold, k=None, cluster_column="cluster") -> dict:
    import torch

    t = ds.scanner(columns=[column], filter=where, with_row_address=True).to_table()
    x = torch.from_numpy(matrix(t.column(column))).to("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.nn.functional.normalize(x.float(), dim=1)
    n = len(x)
    k = k or max(1, int(math.sqrt(n) / 2))
    centroids, assign = _kmeans(x, k)
    is_dup = torch.zeros(n, dtype=torch.bool, device=x.device)
    for j in range(k):
        idx = (assign == j).nonzero().squeeze(1)
        if len(idx) < 2:
            continue
        # most central row first; a row is a dup if any earlier row is within the threshold
        order = idx[(x[idx] @ centroids[j]).argsort(descending=True)]
        for lo in range(0, len(order), 8192):
            rows = order[lo : lo + 8192]
            sims = x[rows] @ x[order[: lo + len(rows)]].T
            col = torch.arange(sims.shape[1], device=x.device)
            earlier = col[None, :] < (lo + torch.arange(len(rows), device=x.device))[:, None]
            is_dup[rows] = ((sims > threshold) & earlier).any(1)
    merge_columns(
        ds,
        t.column("_rowaddr"),
        **{name: pa.array(is_dup.cpu().numpy()), cluster_column: pa.array(assign.cpu().numpy().astype(np.int32))},
    )
    d = int(is_dup.sum())
    log(f"dedup {column} @ {threshold}: {d:,} of {n:,} rows flagged ({100 * d / max(n, 1):.1f}%), {k} clusters")
    return {"rows": n, "dups": d, "clusters": k, "threshold": threshold}
