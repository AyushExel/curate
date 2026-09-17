# Results

The same four verbs, run on one real dataset per domain. All runs on one brev box with 2x H100 PCIe (80 GB), 40 cores, local NVMe. Data are local slices of the `lance-format/*` Hub tables; the default engine (Geneva when importable) was used unless stated. Raw numbers are in `experiments/results/*.json`, scripts in `experiments/`.

## Summary

| Domain | Dataset | Rows | Signals added | Kept after curation | Downstream / sanity check |
|---|---|---|---|---|---|
| LLM pretraining | fineweb-edu | 1,000,000 docs | n_words, lang, quality (GPU), contaminated, is_dup, is_near_dup | 980,546 (98.1%) | 35M GPT, paired seeds: top-quartile < random < bottom-quartile on target eval; 8-trial auto-loop finds quality+length |
| Image-text | laion | 100,000 pairs | res, aesthetic (GPU), clip (GPU), is_dup | 21,625 (21.6%) | CLIP fine-tune, COCO retrieval: no filter beats random; 6-trial auto-loop confirms same-model CLIP-score filtering hurts monotonically |
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

### Downstream: same small GPT on random / top-quartile / bottom-quartile quality

Setup (`experiments/text_train.py arms --seeds 0 1`): GPT-2 tokenizer as a `text.tokenize` signal (1M docs, Geneva), a deterministic 2% held-out split as a bool signal (`crc32(id) % 50 == 0`), then for each arm and seed a 150k-doc training set written as a `sample()` column. Same model (8 layers, 512 wide, 35M params), 3,000 steps of 32 x 1024 = 98M tokens, one H100, 5.4 minutes per run. Two evals: `val_all` (2% of everything) and `val_top` (the held-out top quartile, `quality >= 3.25`), 100 batches each.

| Training set (`where`) | seed | val_top | val_all |
|---|---|---|---|
| random (`TRUE`) | 0 | 6.714 | 6.764 |
| random | 1 | 6.647 | 6.702 |
| top quartile (`quality >= 3.25` + eng + no dups) | 0 | **6.648** | 6.768 |
| top quartile | 1 | **6.598** | 6.727 |
| bottom quartile (`quality < 2.68` + eng + no dups) | 0 | 6.762 | 6.764 |
| bottom quartile | 1 | 6.740 | 6.745 |

Paired by seed, top minus random on `val_top` is -0.065 and -0.049; bottom minus random is +0.048 and +0.093. The ordering top < random < bottom holds at both seeds and spans ~0.11 nats, while seed-to-seed noise on the same arm is ~0.02 to 0.07. On `val_all` the unfiltered set is as good as or better than either filtered one (top: +0.004, +0.025; bottom: 0.000, +0.043). So: the quality signal orders training sets on the target distribution, and filtering costs you on the raw one. That is the trade a curation recipe makes, and here it is measured rather than assumed.

Caveat that matters: my first cut of this experiment used one seed and reported a 0.030 gap. Seed noise on the random arm alone turned out to be 0.067. One seed is not an ablation; paired seeds with a bottom arm are the minimum.

Loader bug found on the way: `Dataset.torch()` (lancedb 0.38.0 `StreamingDataset`) panicked on this table in `dataloader/permutation/builder.rs:230` with `SchemaError("target schema is not superset of current schema ...")`, with or without a filter or shuffle, and on every table exported from it. Bisecting fresh tables cleared row-address merges, field metadata, dropped columns and tags; the trigger is **schema-level metadata**. The fineweb-edu Lance table carries a `huggingface` key in its schema metadata from the Parquet conversion (as any HF-converted dataset will), and the permutation builder cannot handle it. Stripping it is a metadata-only commit (`lance.dataset(uri).replace_schema_metadata({})`, no data rewrite) and streaming works again. `torch()` now raises a clear error with that one-liner instead of a Rust panic, and `export()` no longer copies table-level metadata. Worth a lancedb issue: any Hub dataset converted with `datasets` hits it. The ablation itself used a 12-line in-memory loader; the subsets fit in RAM.

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

### Downstream: fine-tune CLIP on random vs a hand-picked curated view (DataComp-style)

Setup (`experiments/image_train.py`): 20k pairs as a `sample()` column from (a) all rows and (b) `NOT is_dup AND clip >= 0.28 AND aesthetic >= 4.5 AND res >= 200 AND NSFW = 'UNLIKELY'` (44k-row pool). Fine-tune open_clip ViT-B/32 (laion2b) for 2 epochs, batch 128, lr 1e-5, symmetric InfoNCE, then zero-shot retrieval on COCO-2017 val (5k images, first caption). Two seeds, ~4 minutes per run.

| Training pairs | seed | t2i R@1 | t2i R@5 | i2t R@1 | i2t R@5 |
|---|---|---|---|---|---|
| pretrained, no fine-tune | | 36.96 | 62.58 | 37.52 | 64.70 |
| random 20k | 0 | 34.92 | 60.34 | 36.52 | 63.86 |
| random 20k | 1 | 34.40 | 60.36 | 37.16 | 64.28 |
| curated 20k | 0 | 33.64 | 60.08 | 36.40 | 63.36 |
| curated 20k | 1 | 34.20 | 60.34 | 36.08 | 62.66 |

Findings:

- Fine-tuning a converged CLIP on 20k LAION pairs hurts COCO retrieval by 2-3 R@1 points regardless of the pairs, and my hand-picked curated recipe is not better than random (paired by seed: -1.28 and -0.20 on t2i R@1). A negative result, reported as one.
- Two likely reasons, both instructive. Selecting pairs by the score of the very model being trained (our `clip` is ViT-B/32) keeps the pairs it already agrees with and throws away the ones it could learn from; DataComp notes the same bias. And `aesthetic >= 4.5` shifts the training pool away from COCO's everyday photos.
- The honest reading is that a curation recipe is a hypothesis to test, not a setting to assume. The image loop below turns that around: the same objective, the recipe chosen by search.

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

## Auto-curation: the research loop on two tasks

`curate.Loop` (DESIGN.md): a proposer picks the next `where`, the loop filters, calls the task's `train`, and tags the trial with its recipe and metrics. Trial 0 is always the unfiltered baseline (`grid(["TRUE"])`), then Claude (`claude-opus-5`) proposes from the column stats and the history. The GPU box has no Anthropic credentials, so the proposer ran on the workstation behind an ssh tunnel (`experiments/propose_server.py`); the loop itself, the data and the training all stayed on the box.

### Image: maximize COCO t2i R@1 after fine-tuning CLIP on 20k LAION pairs (`experiments/loop_image.py`)

| Trial | `where` | Rows | t2i R@1 | Proposer's reason (abridged) |
|---|---|---|---|---|
| img_0 | `TRUE` | 100,000 | 34.92 | baseline |
| img_1 | `clip > 0.3434` (median) | 49,989 | 32.34 | test the single strongest known curation signal |
| img_2 | `is_dup = false` | 84,544 | **34.98** | CLIP filter lowered recall; isolate the mildest signal |
| img_3 | `width >= 200 AND height >= 200` | 72,997 | 33.40 | try an orthogonal signal: drop thumbnails |
| img_4 | `clip > 0.3741` (p75) | 25,024 | 31.38 | proposed under a stale prompt that said lower-is-better (see note) |
| img_5 | `similarity > 0.3252` (LAION's OpenAI-CLIP score) | 49,984 | 32.94 | does a different CLIP's score carry different information |

Six trials, no recipe beats the unfiltered pool; dedup is a tie. The useful output is the monotone curve on the trained model's own score: keep all (34.92), keep top half (32.34), keep top quarter (31.38). Filtering by the score of the model you then train removes exactly the pairs it has something to learn from. LAION's `similarity` (a different CLIP checkpoint) is a little less harmful (32.94) but still harmful. Aesthetic filtering was never reached in six trials. Note on img_4: the proposer server was still running an older system prompt that hard-coded "lower is better" while the context said maximize; the model followed the prompt and called 32.34 a gain. Fixed before img_5 (`context.objective` now drives the prompt); left in the table because it happened.

What this buys the "does curation work" question for images: at this scale and with this proxy (fine-tuning a converged model), no. The loop reached that conclusion in 25 GPU-minutes with a rationale per step, which is more than the hand-picked recipe above told us.

### Text: minimize held-out top-quartile loss with the 35M GPT (`experiments/loop_text.py`)

Objective: `val_top` loss (held-out top quartile), 150k-doc sample per trial, 3,000 steps, one seed. Reference arms at the same seed: random 6.714, top quartile 6.648, bottom quartile 6.762.

| Trial | `where` | Rows | val_top | Proposer's reason (abridged) |
|---|---|---|---|---|
| trial_0 | `TRUE` | 979,960 | 6.700 | baseline |
| trial_1 | `quality >= 2.918` (median) | 489,668 | 6.678 | learn the sign of the quality/score relationship |
| trial_2 | `quality >= 3.2617` (p75) | 245,409 | 6.665 | does the gain keep scaling with selectivity |
| trial_3 | `quality >= 3.45` | 153,662 | 6.651 | push to the row floor to find where the trade-off turns |
| trial_4 | `quality >= 3.2617 AND n_tokens >= 322` | 192,908 | 6.648 | hold quality, add one new lever: drop the shortest quarter |
| trial_5 | `quality >= 3.15 AND n_tokens >= 591` | 166,885 | **6.589** | push length to the median, relax quality to stay above the floor |
| trial_6 | `n_tokens >= 1024` (no quality filter) | 255,385 | 6.643 | isolate length: how much of trial_5 is length alone |
| trial_7 | `quality >= 3.0 AND n_tokens >= 800` | 160,522 | 6.596 | push length further, relax quality again |

Eight trials, 43 GPU-minutes. The loop reproduced the hand-run arms (quality helps monotonically: 6.700 → 6.678 → 6.665 → 6.651), then found a lever I had not tested (document length), isolated it (trial_6: length alone gives 0.057, quality alone 0.05, both together 0.111), and stopped improving once it hit the 150k-row floor. The best recipe beats the best hand-picked arm by 0.06 nats, roughly the seed noise, so it needs a second seed before anyone believes it; the loop records enough that re-running `trial_5` with `seed=1` is one call.

Two things to be honest about. The length lever is partly an artifact of the trainer: the row-mode loader pads every document to 1,024 tokens, so a length floor means more real tokens per step. The loop optimizes what the pipeline actually rewards, and it found the pipeline's inefficiency before it found the data's structure. That is a feature (it is how DataSmith-style systems earn their keep) and a warning (make the trainer honest before trusting the tree). Second, all trials share one seed; the arms above show why that matters.

Every trial is a tag on the fineweb table: `curate.load(uri, tag="trial_5")` returns the winning view with its recipe, `Loop(ds, ...).tree()` in a fresh process returns this table, and `trial_5_train` is the exact 150k-doc bool column the model saw.

## What the runs changed in the design

See "Lessons from the runs so far" in DESIGN.md. In one line each: thresholds do not transfer across domains; stats must paste into SQL; half-computed columns are normal and mean resume; re-running a script must be safe; provenance belongs on the column, not in a per-view log; one seed is not an ablation; a curation recipe is a hypothesis, and the loop is how you test it; the loop will find your trainer's inefficiencies before your data's structure.
