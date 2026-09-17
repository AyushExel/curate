"""Image-text: laion slice (100k pairs with inline JPEG bytes and ViT-L/14 embeddings).

Signals: resolution, aesthetic (on shipped L/14 embedding), CLIP B/32 image-caption score.
Selection: semantic dedup on the shipped embedding. Check: our clip score vs LAION's `similarity`.
"""

import sys

import curate
from common import DATA, save, spearman, timings

engine = sys.argv[1] if len(sys.argv) > 1 else None
ds = curate.load(f"{DATA}/laion.lance")
print(ds)

ds.signal(
    res=curate.image.resolution("width", "height"),
    aesthetic=curate.image.aesthetic("img_emb"),
    clip=curate.image.clip_score("image", "caption"),
    engine=engine,
)
no_dup = ds.dedup("img_emb", threshold=0.92)
good = no_dup.filter("aesthetic >= 5 AND clip >= 0.25 AND res >= 256 AND NSFW = 'UNLIKELY'")
print("good:", good)

t = ds.to_table(["clip", "similarity"]).drop_null()
agree = spearman(t.column("clip").to_numpy(), t.column("similarity").to_numpy())
print(f"spearman(our clip_score, LAION similarity) = {agree:.3f}")

good.tag("laion-good-v1")
save(
    "image",
    {
        "rows": len(ds),
        "good": len(good),
        "stats": ds.stats(["res", "aesthetic", "clip", "similarity", "is_dup"]),
        "spearman_clip_vs_laion_similarity": agree,
        "timings": timings(good),
        "recipe": good.recipe,
    },
)
