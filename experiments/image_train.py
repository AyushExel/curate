"""DataComp-style check for the image recipe: fine-tune CLIP ViT-B/32 on the same
number of laion pairs drawn from (a) all rows and (b) the curated view, identical
steps, then measure zero-shot retrieval on COCO-2017 val (5k images, first caption).

Run image.py first (it writes clip / aesthetic / res / is_dup).
"""

import io
import math
import sys
import time

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile

import curate
from common import DATA, save

ImageFile.LOAD_TRUNCATED_IMAGES = True
PAIRS, BATCH, EPOCHS, LR = 20_000, 128, 2, 1e-5
CURATED = "NOT is_dup AND clip >= 0.28 AND aesthetic >= 4.5 AND res >= 200 AND NSFW = 'UNLIKELY'"
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def load_pairs(view, name, seed=0):
    """The training pairs are a `sample()` column, like every other training set here;
    only those rows' image bytes are read."""
    ds = curate.load(view.uri)
    sub = view.sample(PAIRS, seed=seed, name=name) if name not in ds.columns else ds.filter(name)
    t = sub.to_table(["image", "caption"])
    return t.column("image").to_pylist(), [c or "" for c in t.column("caption").to_pylist()]


def pil(b):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128))


@torch.no_grad()
def retrieval(model, pre, tok, images, captions):
    model.eval()
    ie, te = [], []
    for lo in range(0, len(images), 256):
        x = torch.stack([pre(pil(b)) for b in images[lo : lo + 256]]).cuda().half()
        ie.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=1))
        te.append(torch.nn.functional.normalize(model.encode_text(tok(captions[lo : lo + 256]).cuda()).float(), dim=1))
    ie, te = torch.cat(ie), torch.cat(te)
    sims = te @ ie.T  # text -> image
    gt = torch.arange(len(ie), device=sims.device)
    out = {}
    for name, s in [("t2i", sims), ("i2t", sims.T)]:
        rank = (s > s[gt, gt][:, None]).sum(1)
        out[f"{name}_r1"] = round(100 * (rank < 1).float().mean().item(), 2)
        out[f"{name}_r5"] = round(100 * (rank < 5).float().mean().item(), 2)
    model.train()
    return out


def finetune(images, captions, coco, seed=0):
    torch.manual_seed(seed)
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, precision="fp16", device="cuda")
    tok = open_clip.get_tokenizer(MODEL)
    before = retrieval(model, pre, tok, *coco)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.1, eps=1e-6)
    steps = EPOCHS * (len(images) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler()
    rng, step, t0 = np.random.default_rng(seed), 0, time.time()
    for _ in range(EPOCHS):
        order = rng.permutation(len(images))
        for lo in range(0, len(images) - BATCH + 1, BATCH):
            idx = order[lo : lo + BATCH]
            x = torch.stack([pre(pil(images[i])) for i in idx]).cuda().half()
            y = tok([captions[i] for i in idx]).cuda()
            ie = torch.nn.functional.normalize(model.encode_image(x).float(), dim=1)
            te = torch.nn.functional.normalize(model.encode_text(y).float(), dim=1)
            logits = model.logit_scale.exp().float() * ie @ te.T
            labels = torch.arange(BATCH, device="cuda")
            loss = (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return before, retrieval(model, pre, tok, *coco)


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    ds = curate.load(f"{DATA}/laion.lance")
    coco_t = curate.load(f"{DATA}/coco_val.lance").to_table(["image", "caption"])
    coco = (coco_t.column("image").to_pylist(), coco_t.column("caption").to_pylist())
    arms = {"random": "TRUE", "curated": CURATED}
    out = {"pairs": PAIRS, "epochs": EPOCHS, "lr": LR, "curated_where": CURATED, "curated_pool": len(ds.filter(CURATED)), "runs": {}}
    for seed in seeds:
        for arm, where in arms.items():
            images, captions = load_pairs(ds.filter(where), f"img_{arm}_s{seed}", seed=seed)
            before, after = finetune(images, captions, coco, seed=seed)
            out["baseline"] = before
            out["runs"][f"{arm}_s{seed}"] = after
            print(arm, seed, "before", before, "after", after, flush=True)
            save("image_train", out)


if __name__ == "__main__":
    main()
