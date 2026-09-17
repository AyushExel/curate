# curate

## The problem

Every serious training run today starts with a curation pipeline someone wrote by hand. It is a pile of scripts that read a dataset, compute some scores, write a filtered copy, compute more scores on the copy, write another copy. Three weeks later nobody can say which copy the model was trained on or which filters produced it. Every new signal means another full pass and another copy. Every modality (text, images, video, robot episodes) gets its own pile.

The tooling available does not fix this:

- NeMo Curator (NVIDIA, open source) is a stage-and-executor system. You pick an executor, declare CPU/GPU/memory per stage, and each modality has its own stage classes. It is powerful but it is infrastructure, not something a researcher picks up in an afternoon.
- DataSmith (DatologyAI) is the most interesting recent work: an agent that runs the data research loop, proposing interventions, training, diagnosing, and trying again. But it is closed, text only, and the thing that makes it work is the layer underneath it: a "curation environment" with profiling, filtering, dedup and decontamination as callable operations plus a persistent experiment tree. That layer is what the open ecosystem is missing.
- datatrove, dolma and friends are text pipelines that write new files at every step.

What a researcher actually wants is what Ultralytics gave computer vision: one object, a handful of verbs, sensible defaults, and the option to go deep when needed.

 Curation reduces to four verbs, and each maps onto a Lance offering:

| Verb | What it means | Lance primitive |
|---|---|---|
| **signal** | compute a per-row (or per-episode) value | new column, zero-copy `add_columns` / Geneva `backfill` |
| **filter** | keep rows that satisfy a rule | SQL `where` clause, lazy |
| **select** | dedup, sample, stratify | another column (`is_dup`, `sample_1m`) plus a filter |
| **tag** | freeze a dataset for training | Lance table version + tag, recipe in tag metadata, per-column provenance in field metadata |

 A curated dataset is `(table uri, version, where)`. That recipe is fully reproducible. How each column was made (which signal, which model, which rows, how long) lives in that column's Lance field metadata, so the table carries its own provenance and any view, any client, any later session can read it.

Because every modality is just a Lance table with different column types (string, binary image, blob video, fixed-size-list action), the same four verbs work for LLM pretraining text, image-caption pairs, video generation clips, robot episodes and world-model frame streams. Only the signal library is modality specific.

## The API

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

## What this is not

- Not ingestion. Getting data into Lance is a solved, separate problem (`lance-format/*` on the Hub already holds fineweb-edu, laion, openvid, droid, agibot, pusht, koch and more).
- Not training. `torch()` hands off to the existing streaming loader and stops.
- Not an agent. But the surface is designed so an agent can drive it: every verb is one call, every result is a recipe dict, `stats()` returns JSON, and the experiment tree DataSmith describes is just a set of tags on one table. An open DataSmith-style loop is the obvious thing to build on top.

