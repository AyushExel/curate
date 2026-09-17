# curate: data curation as columns and filters

Status: design draft for review. Nothing here is final. The minimal implementation in this repo exists to test the API against real datasets from four domains, not to ship.

## The problem

Every serious training run today starts with a curation pipeline someone wrote by hand. It is a pile of scripts that read a dataset, compute some scores, write a filtered copy, compute more scores on the copy, write another copy. Three weeks later nobody can say which copy the model was trained on or which filters produced it. Every new signal means another full pass and another copy. Every modality (text, images, video, robot episodes) gets its own pile.

The tooling available does not fix this:

- NeMo Curator (NVIDIA, open source) is a stage-and-executor system. You pick an executor, declare CPU/GPU/memory per stage, and each modality has its own stage classes. It is powerful but it is infrastructure, not something a researcher picks up in an afternoon.
- DataSmith (DatologyAI) is the most interesting recent work: an agent that runs the data research loop, proposing interventions, training, diagnosing, and trying again. But it is closed, text only, and the thing that makes it work is the layer underneath it: a "curation environment" with profiling, filtering, dedup and decontamination as callable operations plus a persistent experiment tree. That layer is what the open ecosystem is missing.
- datatrove, dolma and friends are text pipelines that write new files at every step.

What a researcher actually wants is what Ultralytics gave computer vision: one object, a handful of verbs, sensible defaults, and the option to go deep when needed.

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=3)
```

## The thesis

LanceDB's recent pretraining post makes the point that one Lance table can be the whole data layer for a training run: raw text, curation flags, token ids, the training dataloader and post-training retrieval all live on the same table. Derived signals are new columns (nothing existing is rewritten), curation rules are SQL filters, and Geneva backfills the columns in a distributed, checkpointed job.

That observation is the whole design. Curation reduces to four verbs, and each maps onto a Lance primitive:

| Verb | What it means | Lance primitive |
|---|---|---|
| **signal** | compute a per-row (or per-episode) value | new column, zero-copy `add_columns` / Geneva `backfill` |
| **filter** | keep rows that satisfy a rule | SQL `where` clause, lazy |
| **select** | dedup, sample, stratify | another column (`is_dup`, `sample_1m`) plus a filter |
| **tag** | freeze a dataset for training | Lance table version + tag, recipe in tag metadata, per-column provenance in field metadata |

Nothing is ever copied. A curated dataset is `(table uri, version, where)`. That triple is the recipe, it is fully reproducible, and it is small enough to paste into a paper. How each column was made (which signal, which model, which rows, how long) lives in that column's Lance field metadata, so the table carries its own provenance and any view, any client, any later session can read it.

Because every modality is just a Lance table with different column types (string, binary image, blob video, fixed-size-list action), the same four verbs work for LLM pretraining text, image-caption pairs, video generation clips, robot episodes and world-model frame streams. Only the signal library is modality specific.

## The API in one screen

```python
import curate

ds = curate.load("hf://datasets/lance-format/fineweb-edu/data/train.lance")
ds                                   # Dataset(1,525,223,056 rows, 12 cols, v66)

# 1. signals: new columns, computed once, checkpointed, GPU aware
ds.signal(
    lang=curate.text.language("text"),
    quality=curate.text.quality("text"),     # fineweb-edu classifier, 0..5
    n_words=curate.text.length("text"),
)

# 2. selection: also columns
ds.dedup("text")                                    # exact, writes is_dup
ds.dedup("text_embedding", threshold=0.95)          # semantic (SemDeDup), writes is_dup + cluster

# 3. rules: SQL. Views are lazy, nothing is copied.
clean = ds.filter("lang = 'en' AND quality >= 3 AND NOT is_dup")
sub = clean.sample(1_000_000, seed=0)               # writes a bool column, returns the view

# 4. freeze and hand to training
sub.tag("fineweb-1m-v1")
sub.recipe                                          # {'uri', 'version': 71, 'where': "...", 'steps': [...], 'columns': {...}}
sub.recipe["columns"]["quality"]                    # {'signal': 'quality', 'inputs': ['text'], 'model_name': ..., 'rows': ..., 'seconds': ..., 'engine': 'geneva'}
sub.stats()                                         # counts and histograms of every signal column
loader = sub.torch(columns=["input_ids"], batch_size=32)   # lancedb StreamingDataset: shuffled, resumable, multi-rank
```

Reopen later, anywhere:

```python
sub = curate.load(uri, tag="fineweb-1m-v1")         # same version, same where, same rows
```

### Custom signals

A signal is a function of columns that returns one new column. Built-in signals are classes; you bind them to columns by constructing them. Custom ones are a decorator:

```python
@curate.signal(dtype="float32", gpu=True, batch_size=256)
def toxicity(text: pa.Array) -> pa.Array:
    ...

ds.signal(tox=toxicity("text"))
```

Anything that needs a model uses `setup()`, which runs once per worker:

```python
class Perplexity(curate.Signal):
    dtype = "float32"; gpu = True; batch_size = 64
    def __init__(self, column, model="gpt2"):
        super().__init__(column); self.model_name = model
    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
    def compute(self, text: pa.Array) -> pa.Array:
        ...
```

### Episodes, clips, and other groups

Robot and world-model data are frame tables where the unit of curation is the episode. `group()` gives you per-group signals, broadcast back onto the rows so that filters stay plain SQL over one table:

```python
frames = curate.load("koch/frames.lance")
eps = frames.group("episode_index", order="frame_index")
eps.signal(
    ep_len=curate.robot.length(),
    jerk=curate.robot.jerk("action"),
    idle=curate.robot.idle_fraction("action"),
)
good = frames.filter("jerk < 0.02 AND idle < 0.4 AND ep_len > 100")
good.tag("koch-smooth-v1")
```

Signals are idempotent: running the same `signal()` again skips complete columns and resumes partial ones (only rows of the view that are still null get computed). `ds.drop("quality")` throws a column away to recompute it with different settings.

### Engine

Signals run on one of two engines with identical semantics:

- `local`: in process, one GPU, streams batches through Lance `add_columns`. For a laptop or a single box.
- `geneva`: Geneva registers the column as a virtual column and runs a Ray backfill: N workers, one GPU each, checkpointed, resumable, `where`-aware so you only compute on the rows you kept. Same signal class, no code change.

```python
curate.config.engine = "geneva"        # or per call: ds.signal(..., engine="geneva", concurrency=8)
```

Default is `auto`: Geneva when importable, otherwise local. Measured (RESULTS.md): on 100k docs Geneva's 20 CPU workers were 3.7x a single process for fastText language id (10x at 1M rows), while two GPU workers were only 1.3x one local GPU for the BERT quality classifier, because that signal is bound by tokenization on the worker's CPU, not by the GPU. The default stands; the ceiling is set by the signal.

## Built-in signal library

Signals are deliberately boring and well known. The value is not any one signal, it is that adding one is a column and combining them is SQL.

| Modality | Signal | Output | In minimal impl |
|---|---|---|---|
| text | `length` | int words | yes |
| text | `language` (fasttext lid) | string | yes |
| text | `quality` (fineweb-edu classifier) | float 0..5 | yes |
| text | `embed` (MiniLM) | vector[384] | yes |
| text | `contaminated(against=eval_set, ngram=13)` | bool | yes |
| text | `tokenize(tokenizer)` | list<int32> | yes |
| text | `perplexity`, `toxicity`, `repetition` | float | planned |
| image | `clip_score(image, caption)` | float | yes |
| image | `aesthetic` (LAION head on CLIP ViT-L/14) | float 1..10 | yes |
| image | `embed` (CLIP) | vector | yes |
| image | `resolution` | int min side | yes |
| image | `nsfw`, `blur`, `watermark`, `ocr_density` | float | planned |
| video | `motion` (mean frame diff) | float | yes |
| video | `scene_cuts` | int | yes |
| video | `clip_score(video, caption)` (sampled frames) | float | yes |
| video | `duration`, `fps`, `resolution` | scalar | trivial |
| video | `aesthetic`, `flicker`, `text_overlay`, `speech` | float | planned |
| robot / world model | `length` (per episode) | int | yes |
| robot / world model | `jerk(action)` (smoothness) | float | yes |
| robot / world model | `idle_fraction(action)` | float | yes |
| robot / world model | `success(flag_column)` | bool | yes |
| robot / world model | `path_length(state)` | float | yes |
| robot / world model | `state_coverage`, `visual_dup` (via image.embed + dedup) | float / bool | via composition |
| any | `dedup(column)` exact | bool | yes |
| any | `dedup(embedding, threshold)` semantic | bool + cluster | yes |
| any | `sample(n, seed, by=)` random / stratified | bool | yes |

## Why Lance makes this cheap

Every design decision above leans on something the format already does:

- `add_columns` appends a column file, it does not rewrite the table. Fifty signals on a 10 TB table cost fifty column files.
- Geneva `backfill` is a checkpointed job that skips rows already computed and takes a `where` filter, so recomputing after a crash or a new filter is incremental.
- Blob columns hold video and images next to the metadata. `openvid-lance` is 7.5 TB with the clips inline; a signal that decodes 8 frames per clip reads exactly those bytes.
- Row addresses are stable within a version, so selection columns (`is_dup`, `sample_1m`) can be computed on the driver and merged in by `_rowaddr` without a join key.
- Versions and tags are built in. The recipe rides along in tag metadata; each column's provenance rides along in field metadata.
- `lancedb.streaming.StreamingDataset` takes the same `where` string, so the curated view is the training set with no export step.
- `hf://`, `s3://`, `gs://` and local paths are all just URIs.

## What this is not

- Not ingestion. Getting data into Lance is a solved, separate problem (`lance-format/*` on the Hub already holds fineweb-edu, laion, openvid, droid, agibot, pusht, koch and more).
- Not training. `torch()` hands off to the existing streaming loader and stops.
- Not an agent. But the surface is designed so an agent can drive it: every verb is one call, every result is a recipe dict, `stats()` returns JSON, and the experiment tree DataSmith describes is just a set of tags on one table. An open DataSmith-style loop is the obvious thing to build on top.

## Experiments

Goal: run the same four verbs on one real dataset per domain, measure where the time goes, and see whether the API holds up when the data is not text. All on 2x H100 PCIe. Data slices are local copies of `lance-format/*` tables.

| Domain | Dataset slice | Signals | Selection | Downstream check |
|---|---|---|---|---|
| LLM pretraining | fineweb-edu, 1M docs | language, quality, tokenize, contaminated (vs GSM8K) | exact + semantic dedup, stratified sample | small GPT: curated vs random subset at equal tokens, val loss |
| image-text | laion, 100k pairs | clip_score, aesthetic, resolution | semantic dedup on img_emb | agreement of our clip_score with LAION's stored `similarity` |
| video gen | openvid, 3k clips | motion, scene_cuts, clip_score | semantic dedup on stored embedding | correlation of our motion score with OpenVid's `motion_score` |
| robot policy | koch + pusht frames | per-episode length, jerk, idle, success | filter episodes | episodes kept, frames kept |
| world model | lewm_cube, 200k frames | per-episode jerk/idle, frame embed | near-dup frame removal | fraction of frames removed |

Results live in `RESULTS.md` and `experiments/results/*.json`.

## Lessons from the runs so far

Things the API had to learn from real data, before the numbers:

- **Thresholds do not transfer across domains.** A 0.97 cosine threshold on CLIP embeddings is a sane near-dup cutoff for web images. On 200k frames of a simulated cube scene it flagged 99.8% of frames: random pairs already sit at 0.92, the nearest neighbour of a typical frame at 0.99. The framework has to make "look at the distribution first" a one-liner, not a footgun. `stats()` gives quantiles; a `neighbours()` helper for the nearest-neighbour similarity distribution is the obvious addition.
- **Numbers from `stats()` must paste into SQL.** Quantiles of an int column came back as floats and Lance refused `ep_len >= 626.15`. Ints stay ints now.
- **A half-computed column is the normal case, not an error.** A GPU job that dies at fragment 6 of 8 leaves a column that exists and is mostly null. `signal()` treats that as "resume", the way Geneva's own backfill does.
- **Re-running a script must be safe.** `tag()` moves an existing tag instead of failing.
- **Robot datasets need per-episode signals, and the frame table is the right place to put them.** Broadcasting 50 episode values onto 38k frames cost nothing and kept every downstream filter a plain SQL string.
- **Video signals are decode-bound, and every signal decodes again.** Three signals on 3k clips meant three decodes per clip. A `frames` signal (a handful of thumbnails per clip, stored as a blob column) that later signals read instead of the mp4 fits the model with no new concept: it is just a column other signals take as input.
- **A GPU signal has to saturate one GPU before a second one helps.** The edu classifier ran at 1.8k docs/s on two H100s; the fix is inside the signal (length-sorted batches, parallel tokenization), not in the engine.
- **Flags you'd expect to exist may not.** pusht's `next_success` is never true in the current release; `peak("next_reward")` per episode stood in. A signal library is partly a set of stand-ins for missing labels.

## Naming

`curate` is taken on PyPI, as are `sift`, `sieve`, `winnow` and `cull`. Available and on theme: `cribble` (a coarse sieve), `bolter` (a flour sieve), `tamis` (French for sieve), `shortlist`. The code uses `curate` as the import name until a name is picked; renaming is a search-and-replace.

## Open questions for the review

1. `signal()` vs `compute()` vs `annotate()` for the verb that adds a column.
2. `dedup` and `sample` write columns. The runs made this feel right (every training set in the ablation was a bool column and both eval sets were a `where`), but the table gains a column per experiment. Is that acceptable, or should experiment columns live under a namespace / get garbage-collected with the tag?
3. Engine default: measured, `auto` (Geneva when importable) stands. The open part is whether `local` should use all GPUs of one box itself (DataParallel over batches) so a laptop-to-workstation user never needs Ray.
4. Grouped signals broadcast the per-episode value to every frame. It cost nothing at 200k frames. At 100M frames a sibling episodes table with a join is the alternative; the pusht conversion already ships both tables.
5. How much of the signal library belongs in this package versus being examples users copy. The runs suggest the library's real job is stand-ins and sanity checks (peak reward when success is missing, recomputed quality vs shipped score), which argues for a small core plus a cookbook.
6. Thresholds. Every semantic dedup run needed a look at the similarity distribution first. Should `dedup()` refuse to run without one (print it, or take `threshold="p99"` and pick it from the nearest-neighbour distribution)?
7. Provenance now lives in Lance field metadata per column. Should the `steps` lineage on a view go away entirely in favour of `(version, where)` plus column provenance?
