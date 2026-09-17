"""World model: LeWorldModel cube slice (200k frames, 224x224 JPEG + 5-d action + 28-d state).

Per-episode signals: length, jerk, idle. Per-frame: CLIP embedding, then near-duplicate frames removed.
A generic 0.97 cosine threshold flagged 99.8% of frames here (a sim cube scene is one tight cone in
CLIP space), so the script first prints the similarity distribution and thresholds off that.
"""

import sys

import numpy as np
import torch

import curate
from common import DATA, save, timings
from curate.signal import matrix

engine = sys.argv[1] if len(sys.argv) > 1 else None
ds = curate.load(f"{DATA}/lewm.lance")
eps = ds.group("episode_idx", order="step_idx")
print(ds, "episodes:", len(eps))

eps.signal(ep_len=curate.robot.length(), jerk=curate.robot.jerk("action"), idle=curate.robot.idle_fraction("action", eps=1e-3))
ds.signal(emb=curate.image.embed("pixels"), engine=engine)

# look before you threshold: consecutive-frame vs random-pair vs nearest-neighbour cosine similarity
t = ds.to_table(["episode_idx", "step_idx", "emb"]).sort_by([("episode_idx", "ascending"), ("step_idx", "ascending")])
x = torch.nn.functional.normalize(torch.from_numpy(matrix(t.column("emb"))).cuda(), dim=1)
ep = t.column("episode_idx").to_numpy()
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
cons = (x[1:] * x[:-1]).sum(1).cpu().numpy()[ep[1:] == ep[:-1]]
i, j = (torch.randint(0, len(x), (100_000,), device="cuda") for _ in range(2))
rnd = (x[i] * x[j]).sum(1).cpu().numpy()
s = x[:20_000] @ x.T
s[torch.arange(20_000), torch.arange(20_000)] = -1
nn = s.max(1).values.cpu().numpy()
sim = {k: np.quantile(v, qs).round(4).tolist() for k, v in [("consecutive", cons), ("random_pair", rnd), ("nearest_neighbour", nn)]}
for k, v in sim.items():
    print(f"{k:18s} p5/25/50/75/95 = {v}")

threshold = 0.995
ds.drop("is_dup", "cluster")
no_dup = ds.dedup("emb", threshold=threshold)
st = eps.stats(["ep_len", "jerk", "idle"])
good = no_dup.filter(f"idle <= {st['idle']['p95']}")
print("good:", good)
good.tag("lewm-v1")
save(
    "worldmodel",
    {
        "frames": len(ds),
        "episodes": len(eps),
        "good_frames": len(good),
        "similarity_quantiles": {"q": qs, **sim},
        "dedup_threshold": threshold,
        "episode_stats": st,
        "frame_stats": ds.stats(["is_dup"]),
        "timings": timings(good),
        "recipe": good.recipe,
    },
)
