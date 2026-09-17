import json
import os

import numpy as np

DATA = os.environ.get("CURATE_DATA", "/ephemeral/curate/data")
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def save(name: str, obj: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"saved {path}")


def timings(ds) -> dict:
    keys = ("signal", "op", "rows", "seconds", "engine", "dups", "clusters", "groups")
    return {name: {k: v[k] for k in keys if k in v} for name, v in ds.provenance().items()}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])
