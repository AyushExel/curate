"""Same signal, both engines, 100k fineweb docs: local (1 process, 1 GPU) vs geneva (Ray, 1 worker per GPU).

Answers design question 3: is Geneva worth its startup cost at this size?
"""

import curate
from common import DATA, save

ds = curate.load(f"{DATA}/fineweb.lance")
bench = ds.sample(100_000, seed=0, name="bench") if "bench" not in ds.columns else ds.filter("bench")

bench.signal(lang_local=curate.text.language("text"), engine="local")
bench.signal(lang_geneva=curate.text.language("text"), engine="geneva")
bench.signal(q_local=curate.text.quality("text"), engine="local")
bench.signal(q_geneva=curate.text.quality("text"), engine="geneva")

rows = [s for s in bench.steps if s.get("op") == "signal" and s["name"].endswith(("_local", "_geneva"))]
for s in rows:
    print(f"{s['name']:12s} {s['engine']:7s} {s['rows']:>8,} rows  {s['seconds']:8.1f}s  {s['rows'] / s['seconds']:>8,.0f} rows/s")
save("engine_bench", {"rows": rows})
