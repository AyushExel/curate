"""Video generation: OpenVid slice (3k clips with inline mp4 blobs, captions, scores, embeddings).

Signals: motion (frame diff), scene cuts, CLIP caption alignment on sampled frames.
Selection: semantic dedup on the shipped embedding. Check: our motion vs OpenVid's `motion_score`.
"""

import sys

import curate
from common import DATA, save, spearman, timings

engine = sys.argv[1] if len(sys.argv) > 1 else None
ds = curate.load(f"{DATA}/openvid.lance")
print(ds)

ds.signal(
    motion=curate.video.motion("video_blob"),
    cuts=curate.video.scene_cuts("video_blob"),
    clip=curate.video.clip_score("video_blob", "caption"),
    engine=engine,
)
no_dup = ds.dedup("embedding", threshold=0.95)
good = no_dup.filter("cuts = 0 AND motion >= 0.01 AND clip >= 0.25")
print("good:", good)

t = ds.to_table(["motion", "motion_score", "clip", "aesthetic_score"]).drop_null()
agree = spearman(t.column("motion").to_numpy(), t.column("motion_score").to_numpy())
print(f"spearman(our motion, OpenVid motion_score) = {agree:.3f}")

good.tag("openvid-good-v1")
save(
    "video",
    {
        "rows": len(ds),
        "good": len(good),
        "stats": ds.stats(["motion", "motion_score", "cuts", "clip", "seconds", "is_dup"]),
        "spearman_motion_vs_openvid": agree,
        "timings": timings(good),
        "recipe": good.recipe,
    },
)
