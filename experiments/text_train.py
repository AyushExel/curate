"""Downstream check for text recipes: the same small GPT trained on a fixed number
of docs from a view, same steps, same tokens, then held-out loss on two sets:
val_all (2% of everything) and val_top (the held-out top quartile of `quality`).

Run text.py first (it writes quality / lang / is_dup / is_near_dup / contaminated).

    python text_train.py prepare                    # tokenize + val split (once)
    python text_train.py arms --seeds 0 1           # random / top quartile / bottom quartile
    python text_train.py trial --where "..." --name trial-3   # one view, prints JSON metrics (used by the loop)
"""

import argparse
import json
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
PAD = 50257  # one past gpt2 vocab, masked from the loss
TOP = "quality >= 3.25"  # top quartile of the fineweb-edu score in this slice
ARMS = {
    "random": "TRUE",
    "top": f"{TOP} AND lang = 'eng_Latn' AND NOT is_dup AND NOT is_near_dup",
    "bottom": "quality < 2.68 AND lang = 'eng_Latn' AND NOT is_dup AND NOT is_near_dup",
}


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
    print("val_all:", len(ds.filter("is_val")), "val_top:", len(ds.filter(f"is_val AND {TOP}")))


def batches(view, shuffle, seed=0):
    """Padded [BATCH, SEQ] blocks from an in-memory read of the view (150k docs fit)."""
    ids = view.to_table(["input_ids"]).column("input_ids").to_pylist()
    order = np.random.default_rng(seed).permutation(len(ids)) if shuffle else np.arange(len(ids))
    for lo in range(0, len(order) - BATCH + 1, BATCH):
        x = torch.full((BATCH, SEQ), PAD, dtype=torch.long)
        for r, i in enumerate(order[lo : lo + BATCH]):
            row = ids[i][:SEQ]
            x[r, : len(row)] = torch.tensor(row)
        yield x


@torch.no_grad()
def evaluate(model, view, n=100):
    model.eval()
    tot = 0.0
    for i, x in enumerate(batches(view, shuffle=False)):
        if i == n:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model(x.cuda(), loss_mask=x.cuda() != PAD).item()
    model.train()
    return tot / n


def train(view, name, seed=0):
    """Train on DOCS docs sampled from `view` (excluding the val split); return metrics."""
    ds = curate.load(view.uri)
    col = f"{name}_train"
    sub = view.filter("NOT is_val")
    sub = sub.sample(DOCS, seed=seed, name=col) if col not in ds.columns else ds.filter(col)
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=PAD + 1, seq_len=SEQ, n_layer=8, n_head=8, d_model=512)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    t0, step = time.time(), 0
    for x in batches(sub, shuffle=True, seed=seed):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x.cuda(), loss_mask=x.cuda() != PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        if step % 500 == 0:
            print(f"{name} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step == STEPS:
            break
    val_all = evaluate(model, ds.filter("is_val"))
    val_top = evaluate(model, ds.filter(f"is_val AND {TOP}"))
    return {"score": round(val_top, 4), "val_all": round(val_all, 4), "val_top": round(val_top, 4), "train_rows": len(sub), "seconds": round(time.time() - t0), "seed": seed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "arms", "trial"])
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--where")
    p.add_argument("--name")
    a = p.parse_args()
    if a.mode == "prepare":
        return prepare()
    ds = curate.load(f"{DATA}/fineweb.lance")
    if a.mode == "arms":
        out = {}
        for arm in a.arms:
            for seed in a.seeds:
                out[f"{arm}_s{seed}"] = {"where": ARMS[arm], **train(ds.filter(ARMS[arm]), f"arm_{arm}", seed=seed)}
                print(out[f"{arm}_s{seed}"], flush=True)
                save("text_arms", out)
        return
    m = train(ds.filter(a.where), a.name)
    print("METRICS " + json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
