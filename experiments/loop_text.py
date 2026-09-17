"""Auto-curation on the fineweb slice: a proposer picks the next WHERE, each trial
trains the same GPT (text_train.train) on a 150k-doc sample of that view, the
score is held-out loss on the top-quartile validation set, and every trial is a
tag on the table.  Usage:  python loop_text.py [budget] [proposer_url]"""

import json
import sys
import urllib.request

import curate
from common import DATA, save
from text_train import ARMS, train

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 8
url = sys.argv[2] if len(sys.argv) > 2 else None


def http_proposer(history, context):
    body = json.dumps({"history": [vars(t) for t in history], "context": context}, default=str).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return None if out.get("stop") else out


ds = curate.load(f"{DATA}/fineweb.lance").filter("NOT is_val")
kw = dict(train=train, columns=["quality", "score", "n_words", "n_tokens", "lang", "is_dup", "is_near_dup", "contaminated"], min_rows=150_000)
# trial 0 is the unfiltered baseline so the proposer has a reference score; it is
# tagged like any other trial, and the second Loop picks it up from the table's tags
curate.Loop(ds, propose=curate.grid(["TRUE"]), budget=1, **kw).run()
tree = curate.Loop(ds, propose=http_proposer if url else None, budget=budget - 1, **kw).run()
for t in tree:
    print(f"{t.name:10s} score={t.score}  rows={t.rows:,}  {t.where}   # {t.rationale}")
save("text_loop", {"budget": budget, "arms": ARMS, "tree": [vars(t) for t in tree]})
