"""Auto-curation on the laion slice: the proposer picks a WHERE over the image
signals, each trial fine-tunes CLIP ViT-B/32 on 20k pairs from that view
(image_train.finetune) and the score is COCO text->image R@1 (higher is better).
Usage:  python loop_image.py [budget] [proposer_url]"""

import json
import sys
import urllib.request

import curate
from common import DATA, save
from image_train import finetune, load_pairs

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 6
url = sys.argv[2] if len(sys.argv) > 2 else None

coco_t = curate.load(f"{DATA}/coco_val.lance").to_table(["image", "caption"])
coco = (coco_t.column("image").to_pylist(), coco_t.column("caption").to_pylist())


def train(view, name):
    images, captions = load_pairs(view, f"{name}_train")
    before, after = finetune(images, captions, coco)
    return {"score": after["t2i_r1"], **after, "pretrained_t2i_r1": before["t2i_r1"], "train_rows": len(images)}


def http_proposer(history, context):
    body = json.dumps({"history": [vars(t) for t in history], "context": context}, default=str).encode()
    out = json.loads(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=900).read())
    return None if out.get("stop") else out


ds = curate.load(f"{DATA}/laion.lance")
kw = dict(train=train, columns=["clip", "similarity", "aesthetic", "res", "width", "height", "is_dup"], min_rows=20_000, minimize=False, prefix="img")
curate.Loop(ds, propose=curate.grid(["TRUE"]), budget=1, **kw).run()
tree = curate.Loop(ds, propose=http_proposer if url else None, budget=budget - 1, **kw).run()
for t in tree:
    print(f"{t.name:8s} t2i_r1={t.score}  rows={t.rows:,}  {t.where}   # {t.rationale}")
save("image_loop", {"budget": budget, "tree": [vars(t) for t in tree]})
