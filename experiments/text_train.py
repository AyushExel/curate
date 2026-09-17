"""Downstream check for the text recipe: train the same small GPT on an equal
number of docs from (a) the raw slice and (b) the curated view, same steps,
same tokens, then compare held-out loss on two validation sets.

Run text.py first (it writes quality / is_dup / is_near_dup / contaminated).
    python text_train.py prepare          # tokenize + split + sample columns (once)
    CUDA_VISIBLE_DEVICES=0 python text_train.py train random &
    CUDA_VISIBLE_DEVICES=1 python text_train.py train clean &
"""

import math
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

import curate
from common import DATA, save

sys.path.insert(0, "/ephemeral/curate/training/examples/llm_pretraining")
from model import GPT, GPTConfig  # noqa: E402

SEQ, BATCH, STEPS, DOCS = 1024, 32, 3000, 150_000
CLEAN = "quality >= 3.25 AND lang = 'eng_Latn' AND NOT is_dup AND NOT is_near_dup"  # top quartile of the edu score
PAD = 50257  # one past gpt2 vocab, masked from the loss


@curate.signal(dtype="bool", batch_size=8192)
def is_val(id):
    return [zlib.crc32(i.encode()) % 50 == 0 for i in id.to_pylist()]


@curate.signal(dtype="int32")
def n_tokens(input_ids):
    return pc.cast(pc.list_value_length(input_ids), pa.int32())


def prepare():
    ds = curate.load(f"{DATA}/fineweb.lance")
    ds.signal(is_val=is_val("id"))
    ds.signal(input_ids=curate.text.tokenize("text", "gpt2", max_length=SEQ), engine="geneva")
    ds.signal(n_tokens=n_tokens("input_ids"))
    train = ds.filter("NOT is_val")
    train.sample(DOCS, seed=0, name="train_random")
    train.filter(CLEAN).sample(DOCS, seed=0, name="train_clean")
    for name in ("train_random", "train_clean"):
        v = ds.filter(name)
        toks = pc.sum(v.to_table(["n_tokens"]).column(0)).as_py()
        print(f"{name}: {len(v):,} docs, {toks / 1e6:,.0f}M tokens (train consumes {STEPS * BATCH * SEQ / 1e6:,.0f}M)")
    print("val_all:", len(ds.filter("is_val")), "val_clean:", len(ds.filter(f"is_val AND {CLEAN}")))


def batches(view, shuffle, seed=0):
    """Padded [BATCH, SEQ] blocks. The 150k-doc subsets fit in memory, so this reads the
    view once and shuffles with numpy instead of going through the streaming loader
    (lancedb 0.38.0's permutation builder panics on this table, see RESULTS.md)."""
    ids = view.to_table(["input_ids"]).column("input_ids").to_pylist()
    order = np.random.default_rng(seed).permutation(len(ids)) if shuffle else np.arange(len(ids))
    for lo in range(0, len(order) - BATCH + 1, BATCH):
        x = torch.full((BATCH, SEQ), PAD, dtype=torch.long)
        for r, i in enumerate(order[lo : lo + BATCH]):
            row = ids[i][:SEQ]
            x[r, : len(row)] = torch.tensor(row)
        yield x


def loss_fn(model, x):
    return model(x, loss_mask=x != PAD)  # GPT.forward returns the masked mean next-token loss


@torch.no_grad()
def evaluate(model, view, n=100):
    model.eval()
    tot = 0.0
    for i, x in enumerate(batches(view, shuffle=False)):
        if i == n:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += loss_fn(model, x.cuda()).item()
    model.train()
    return tot / n


def train(which):
    ds = curate.load(f"{DATA}/fineweb.lance")
    view = ds.filter(f"train_{which}")
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=PAD + 1, seq_len=SEQ, n_layer=8, n_head=8, d_model=512)
    model = GPT(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    t0, step, log = time.time(), 0, []
    for x in batches(view, shuffle=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model, x.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            print(f"{which} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
            log.append({"step": step, "loss": round(loss.item(), 4)})
        if step == STEPS:
            break
    res = {
        "steps": step,
        "tokens": step * BATCH * SEQ,
        "seconds": round(time.time() - t0),
        "val_all": evaluate(model, ds.filter("is_val")),
        "val_clean": evaluate(model, ds.filter(f"is_val AND {CLEAN}")),
        "train_log": log,
    }
    print(which, {k: v for k, v in res.items() if k != "train_log"})
    save(f"text_train_{which}", res)


if __name__ == "__main__":
    prepare() if sys.argv[1] == "prepare" else train(sys.argv[2])
