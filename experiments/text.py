"""LLM pretraining text: fineweb-edu slice (1M docs).

Signals: words, language, edu quality (GPU), GSM8K contamination.
Selection: exact dedup, semantic dedup on the shipped embedding.
"""

import sys

import curate
from common import DATA, save, timings

engine = sys.argv[1] if len(sys.argv) > 1 else None
ds = curate.load(f"{DATA}/fineweb.lance")
print(ds)

ds.signal(n_words=curate.text.length("text"), lang=curate.text.language("text"), engine=engine)
en = ds.filter("lang = 'eng_Latn'")
print("english:", en)
en.signal(quality=curate.text.quality("text"), engine=engine)

from datasets import load_dataset  # noqa: E402

gsm = load_dataset("openai/gsm8k", "main", split="test")["question"]
ds.signal(contaminated=curate.text.contaminated("text", against=gsm, ngram=13), engine=engine)

no_exact = ds.dedup("text")
no_near = no_exact.dedup("text_embedding", threshold=0.95, name="is_near_dup")
clean = no_near.filter("lang = 'eng_Latn' AND quality >= 2.5 AND NOT contaminated")
print("clean:", clean)

stats = ds.stats(["n_words", "quality", "score", "is_dup", "is_near_dup", "contaminated"])
lang = ds.stats(["lang"])["lang"]
# how well does our recomputed quality agree with the shipped fineweb-edu score?
t = ds.to_table(["quality", "score"]).drop_null()
from common import spearman  # noqa: E402

agree = spearman(t.column("quality").to_numpy(), t.column("score").to_numpy())
print(f"spearman(quality, shipped score) = {agree:.3f}")
print("contaminated docs:", stats["contaminated"]["true"])

clean.tag("clean-v1")
save(
    "text",
    {
        "rows": len(ds),
        "english": len(en),
        "clean": len(clean),
        "stats": stats,
        "lang_top": lang["top"],
        "spearman_quality_vs_shipped_score": agree,
        "timings": timings(clean),
        "recipe": clean.recipe,
    },
)
