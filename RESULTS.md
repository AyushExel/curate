# Results

The same four verbs, run on one real dataset per domain. All runs on one brev box with 2x H100 PCIe (80 GB), 40 cores, local NVMe. Data are local slices of the `lance-format/*` Hub tables; the default engine (Geneva when importable) was used unless stated. Raw numbers are in `experiments/results/*.json`, scripts in `experiments/`.

## Summary

| Domain | Dataset | Rows | Signals added | Kept after curation | Downstream / sanity check |
|---|---|---|---|---|---|
| LLM pretraining | fineweb-edu | 1,000,000 docs | n_words, lang, quality (GPU), contaminated, is_dup, is_near_dup | 980,546 (98.1%) | recomputed `quality` matches the shipped `score`, Spearman 1.000; small-GPT ablation below |
| Image-text | laion | 100,000 pairs | res, aesthetic (GPU), clip (GPU), is_dup | 21,625 (21.6%) | our `clip_score` vs LAION `similarity`, Spearman 0.47 (different CLIP checkpoints) |
| Video gen | openvid | 3,000 clips | motion, cuts, clip (GPU), is_dup | 1,274 (42.5%) | our `motion` vs OpenVid `motion_score`, Spearman 0.68 |
| Robot policy | koch pick-place | 37,972 frames / 50 eps | ep_len, jerk, idle, path | 42/50 eps, 32,777 frames | |
| Robot policy | pusht | 25,650 frames / 206 eps | ep_len, jerk, idle, path, max_reward | 135/206 eps, 17,606 frames | `next_success` is never true in the release; peak reward stood in |
| World model | LeWorldModel cube | 200,000 frames / 996 eps | ep_len, jerk, idle, emb (GPU), is_dup | 189,800 frames (94.9%) | 0.97 threshold flagged 99.8%; distribution-chosen 0.995 flagged 4.6% |

## Text: fineweb-edu, 1M docs

| Step | Rows | Time | Rate | Engine |
|---|---|---|---|---|
| `text.length` | 1,000,000 | 21.9 s | 45.7k rows/s | geneva, 20 CPU workers |
| `text.language` (fastText) | 1,000,000 | 92.4 s | 10.8k rows/s | geneva, 20 CPU workers |
| `text.quality` (fineweb-edu BERT classifier), English rows only | 999,571 | 558.9 s | 1.8k rows/s | geneva, 2 GPU workers |
| `text.contaminated` (13-gram vs GSM8K test, 1,319 questions) | 1,000,000 | 67.9 s | 14.7k rows/s | geneva, 20 CPU workers |
| `dedup("text")` exact | 1,000,000 | | 8 dups | driver |
| `dedup("text_embedding", 0.95)` SemDeDup, 499 clusters | 999,992 | | 17,599 dups (1.8%) | 1 GPU |

Findings:

- The slice is already curated (fineweb-edu keeps `score >= 2.5`; our `quality` min is 2.48), so the "clean" rule `lang = 'eng_Latn' AND quality >= 2.5 AND NOT contaminated AND NOT dups` removes only 1.9%. 429 docs came out with `quality` null: they were not English by fastText (240 `und`, 142 `yue_Hant`, a few dozen others), so the GPU classifier never ran on them. That is `where`-aware backfill doing what it should.
- Spearman between our recomputed `quality` and the shipped `score` is 1.000. Same classifier, so this is a correctness check of the whole path (Geneva worker, fp16, truncation at 512) rather than a discovery.
- Zero GSM8K contamination at 13-gram granularity in this 1M slice.
- Exact duplicates are essentially gone (8), semantic near-duplicates are not (1.8%).

### Engine: local vs Geneva on the same 100k rows

| Signal | local (1 process, 1 GPU) | geneva (Ray, 20 CPU or 2 GPU workers) | Speedup |
|---|---|---|---|
| `text.language` (CPU) | 96.0 s, 1,042 rows/s | 26.0 s, 3,840 rows/s | 3.7x (10.4x at 1M rows) |
| `text.quality` (GPU) | 126.0 s, 794 rows/s | 98.8 s, 1,013 rows/s | 1.3x |

Geneva's fixed cost is 10-15 s of Ray startup per backfill. For CPU signals the 20 workers pay for that quickly. For the GPU classifier, two GPUs gave 1.3x, and both numbers are well under what an H100 does for BERT-base at fp16: the signal is bound by single-threaded HuggingFace tokenization and padding to 512 inside the worker, not by the GPU. A multi-GPU engine cannot fix a signal that does not saturate one GPU. Length-sorted batching and tokenizer parallelism in the signal are the fix, and that is a signal-library problem, not an engine problem.

Design consequence: the `auto` engine default is right for CPU signals and for tables with many fragments, and harmless for GPU signals, but the win is bounded by the signal itself.

### Downstream: same small GPT, top-quartile quality vs random

Setup (`experiments/text_train.py`): GPT-2 tokenizer (a `text.tokenize` signal, 1M docs, Geneva), a deterministic 2% held-out split as a bool signal (`crc32(id) % 50 == 0`), then two 150k-doc training sets as `sample()` columns: `train_random` from all training docs, `train_clean` from `quality >= 3.25 AND eng_Latn AND no dups` (the top quartile of the edu score, 245k docs). Same model (8 layers, 512 wide, 35M params), same 3,000 steps of 32 x 1024, same 98M tokens, same seed, one H100 each, 5 minutes per run.

| Training set | val_all (2% of everything) | val_clean (held-out top quartile) |
|---|---|---|
| `train_random` | **6.744** | 6.700 |
| `train_clean` | 6.778 | **6.670** |

Training on the quality-filtered quartile lowers loss on high-quality held-out text (-0.030) and raises it on the raw distribution (+0.034). That is the textbook shape of a quality filter, at a size (35M params, 98M tokens, one seed) where it is a directional check, not a claim. The point of the exercise is the workflow: every training set is a bool column, every eval set is a `where`, and both runs read the same table at the same version.

Loader caveat: `Dataset.torch()` (lancedb 0.38.0 `StreamingDataset`) panics in `dataloader/permutation/builder.rs:230` (`SchemaError("target schema is not superset of current schema ...")`) on this fineweb table and on any table exported from it, with or without a filter, with or without shuffle. It works on laion, lewm, and on fresh tables with string, list<int32>, fixed-size-list and nullable float columns, so the trigger is not obvious; the fineweb table (its exported 150k-row, 4-column copy is a repro) needs a look from the lancedb side. The ablation used a 12-line in-memory loader instead; the subsets fit in RAM.

## Image-text: laion, 100k pairs (inline JPEG, ViT-L/14 embedding shipped)

| Step | Rows | Time | Rate | Engine |
|---|---|---|---|---|
| `image.resolution` (min side from width/height) | 100,000 | 15.5 s | 6.5k rows/s | geneva, 20 CPU workers (all Ray startup) |
| `image.aesthetic` (LAION head on the shipped L/14 embedding) | 100,000 | 15.6 s | 6.4k rows/s | geneva, 2 GPU workers (all Ray startup) |
| `image.clip_score` (decode JPEG, CLIP B/32 image vs caption) | 100,000 | 142.4 s | 702 img/s | geneva, 2 GPU workers |
| `dedup("img_emb", 0.92)` SemDeDup, 158 clusters | 100,000 | | 15,456 dups (15.5%) | 1 GPU |

Rule `aesthetic >= 5 AND clip >= 0.25 AND res >= 256 AND NSFW = 'UNLIKELY'` after dedup kept 21,625 of 100,000 pairs.

Findings:

- When the embedding already exists, a model-based signal is nearly free: the aesthetic head over 100k embeddings ran in the time it takes Ray to start. Everything expensive here was JPEG decode plus CLIP, at ~700 img/s on two GPUs, which again points at the CPU side (decode and preprocess in 16 threads per worker) rather than the GPU.
- Our `clip_score` and LAION's shipped `similarity` correlate at Spearman 0.47. LAION scored with OpenAI's ViT-B/32; we used the laion2b checkpoint of the same architecture. The two disagree enough that "which CLIP" has to be part of the recipe, which is exactly what the column's provenance metadata records (`model_name`, `pretrained`).
- The shipped `NSFW` column contains stray hashes and the string `'False'` in a handful of rows; a `NSFW = 'UNLIKELY'` filter handles that for free, which is a small argument for SQL over a Python predicate.
- One truncated JPEG in 100k killed the first run. Signals now decode defensively (grey image on failure); a `decode_ok` boolean signal would be the honest version.

## Video: openvid, 3k clips (inline mp4 blobs, ~3 s each, 720p)

| Step | Rows | Time | Rate | Engine |
|---|---|---|---|---|
| `video.motion` (16 frames, mean abs diff) | 3,000 | 55.4 s | 54 clips/s | geneva, 20 CPU workers |
| `video.scene_cuts` (32 frames, histogram L1 > 0.5) | 3,000 | 64.9 s | 46 clips/s | geneva, 20 CPU workers |
| `video.clip_score` (4 frames, CLIP B/32 vs caption) | 3,000 | 113.0 s | 27 clips/s | geneva, 2 GPU workers |
| `dedup("embedding", 0.95)` on the shipped 1024-d embedding, 27 clusters | 3,000 | | 729 dups (24.3%) | 1 GPU |

Rule `cuts = 0 AND motion >= 0.01 AND clip >= 0.25` after dedup kept 1,274 of 3,000 clips.

Findings:

- Every video signal is decode-bound. Each of the three decodes the clip again; 20 CPU workers is what made 3k clips take minutes instead of an hour, and the GPU one is slowest because it decodes at 224px. A `frames` column computed once (Lance blob column, 4 to 16 thumbnails per clip) and reused by every downstream signal is the obvious next step and fits the model: it is just another signal.
- Our 16-frame motion score agrees with OpenVid's `motion_score` at Spearman 0.68. Different definitions (theirs is optical-flow based), so this is a sanity check that the cheap version ranks clips the same way, not a replication.
- 95% of clips have zero hard cuts by our histogram test (max 13, which is a slideshow). OpenVid is pre-cut into shots, so that is expected.
- 24.3% flagged as near-duplicates at 0.95 on the shipped embedding is high and, like the world-model case, says more about how tightly that embedding model clusters than about redundancy. Same lesson: print the neighbour distribution first.

## Robot: koch (real arm) and pusht (sim)

Per-episode signals on the frame tables, broadcast to frames. Rule: drop the top 5% jerk, top 5% idle, bottom 5% episode length, and (pusht) the bottom quarter of peak reward.

| Dataset | Episodes | Frames | ep_len p5/p50/p95 | jerk p50 | idle p50 | Rule | Kept |
|---|---|---|---|---|---|---|---|
| koch | 50 | 37,972 | 626 / 760 / 907 | 0.83 | 0.028 | `jerk <= 1.0177 AND idle <= 0.0481 AND ep_len >= 626` | 42 eps, 32,777 frames |
| pusht | 206 | 25,650 | 69 / 122 / 186 | 5.18 | 0.026 | `jerk <= 7.3521 AND idle <= 0.0753 AND ep_len >= 69 AND max_reward >= 0.8756` | 135 eps, 17,606 frames |

All five group signals on 38k frames took under a second each on the driver; there was nothing to distribute. pusht's `next_success` flag is False on every frame of this release, so `robot.peak("next_reward")` (best coverage reached in the episode, p25 = 0.876) took its place. The rule is chosen off `stats()` quantiles, which is the intended workflow: look, then filter.

## World model: LeWorldModel cube, 200k frames / 996 episodes (224px JPEG, 5-d action, 28-d state)

| Step | Rows | Time | Rate | Engine |
|---|---|---|---|---|
| `robot.length`, `robot.jerk`, `robot.idle_fraction` per episode | 200,000 / 996 eps | < 2 s each | | driver |
| `image.embed` (CLIP B/32 on JPEG frames) | 200,000 | 188.6 s | 1,060 img/s | geneva, 2 GPU workers |
| `dedup("emb", 0.995)` SemDeDup, 223 clusters | 200,000 | | 9,226 dups (4.6%) | 1 GPU |

Cosine similarity of CLIP embeddings, quantiles p5 / p25 / p50 / p75 / p95:

| Pair type | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| random frame pair | 0.874 | 0.901 | 0.918 | 0.935 | 0.955 |
| consecutive frames in an episode | 0.963 | 0.977 | 0.984 | 0.989 | 0.994 |
| nearest neighbour (20k sample) | 0.983 | 0.987 | 0.990 | 0.993 | 0.996 |

Findings:

- The first pass used 0.97, a reasonable near-duplicate cutoff for web images, and flagged 99.8% of frames. Every frame of a simulated cube scene sits inside one narrow cone in CLIP space; the random-pair median is already 0.92. With the distribution in hand, 0.995 (above the p95 of consecutive frames) flags 4.6%, which is a defensible "visually identical" set. This was the single most useful failure in the whole exercise, and the reason the script now prints that table before choosing a threshold.
- The per-episode signals found nothing to filter: 995 of 996 episodes are exactly 201 steps, `idle_fraction` is 0 everywhere, and `jerk` is identical to four decimals across episodes. That is what a scripted or fixed-seed random policy looks like. The signals are not wrong, the dataset has no variation along those axes, and `stats()` said so in one call.
- 1,060 img/s for CLIP B/32 on two H100s is, again, JPEG decode and preprocess on the CPU side of each worker, not the GPU.

## What the runs changed in the design

See "Lessons from the runs so far" in DESIGN.md. In one line each: thresholds do not transfer across domains; stats must paste into SQL; half-computed columns are normal and mean resume; re-running a script must be safe; provenance belongs on the column, not in a per-view log.
