"""Auto-curation: a research loop over recipes.

A trial is a ``where`` clause. The loop asks a proposer for the next one, trains
and evaluates on that view with a user-supplied ``train`` function, and freezes
the trial as a Lance tag whose metadata holds the recipe and the metrics. The
experiment tree is therefore just the table's tags: it survives the process,
any later session can read it, and the next proposer call sees it as history.

    loop = curate.Loop(ds, train=my_train, budget=8)
    loop.run()                      # propose -> filter -> train -> tag, repeat
    loop.tree()                     # every trial ever tagged on this table, best first

``train(view, name) -> dict`` must return metrics including a ``score`` (lower is
better by default). Proposers: ``ClaudeProposer`` (default) or any callable
``(history, context) -> {"where": ..., "rationale": ...} | None``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from .engine import log


@dataclass
class Trial:
    name: str
    where: str
    score: float | None
    metrics: dict = field(default_factory=dict)
    rationale: str = ""
    rows: int = 0


class Loop:
    def __init__(self, ds, train, budget=8, propose=None, columns=None, min_rows=1, minimize=True, prefix="trial"):
        self.ds, self.train, self.budget = ds, train, budget
        self.propose = propose or ClaudeProposer()
        self.columns = columns or [f.name for f in ds.schema if f.name in ds.provenance() or _small(f.type)]
        self.min_rows, self.minimize, self.prefix = min_rows, minimize, prefix

    def context(self) -> dict:
        return {"rows": len(self.ds), "base_where": self.ds.where, "min_rows": self.min_rows, "stats": self.ds.stats(self.columns)}

    def tree(self) -> list[Trial]:
        out = []
        for name, info in self.ds._ds.tags.list().items():
            meta = info.get("metadata") or {}
            if "trial" in meta:
                t = json.loads(meta["trial"])
                out.append(Trial(name, t["where"], t.get("score"), t.get("metrics", {}), t.get("rationale", ""), t.get("rows", 0)))
        return sorted(out, key=lambda t: (t.score is None, t.score if self.minimize else -(t.score or 0)))

    def run(self) -> list[Trial]:
        ctx = self.context()
        history = self.tree()
        for _ in range(self.budget):
            proposal = self.propose(history, ctx)
            if not proposal:
                log("loop: proposer stopped")
                break
            name = f"{self.prefix}_{len(history)}"
            view = self.ds.filter(proposal["where"])
            rows = len(view)
            log(f"loop {name}: {rows:,} rows <- {proposal['where']}  ({proposal.get('rationale', '')})")
            if rows < self.min_rows:
                trial = Trial(name, proposal["where"], None, {"error": f"only {rows} rows, need {self.min_rows}"}, proposal.get("rationale", ""), rows)
            else:
                metrics = self.train(view, name)
                trial = Trial(name, proposal["where"], metrics.get("score"), metrics, proposal.get("rationale", ""), rows)
                recipe = view.tag(name)
                self.ds._ds.tags.replace_metadata(name, {"recipe": json.dumps(recipe), "trial": json.dumps(vars(trial), default=str)})
            history.append(trial)
            log(f"loop {name}: score={trial.score}")
        return self.tree()


class ClaudeProposer:
    """Asks Claude for the next ``where`` given the column stats and the trial history.
    Uses the anthropic SDK when credentials resolve, else the local ``claude`` CLI."""

    SYSTEM = (
        "You run a data-curation research loop. A training set is a SQL WHERE clause over the columns "
        "described in `stats` (Lance/DataFusion SQL: AND, OR, NOT, comparisons, IS NULL, string equality). "
        "Every trial trains the same model with the same compute on a fixed-size random sample of the rows "
        "your WHERE keeps, then reports `score` (lower is better). Propose the next WHERE to beat the best "
        "score. Use only the listed columns. Keep at least `min_rows` rows (use the quantiles to estimate). "
        "Never repeat a tried WHERE. Prefer testing one idea at a time so results are interpretable. "
        'Reply with JSON only: {"where": "...", "rationale": "one sentence"} or {"stop": true, "rationale": "..."}.'
    )

    def __init__(self, model="claude-opus-5"):
        self.model = model

    def __call__(self, history, context):
        history = [t if isinstance(t, dict) else vars(t) for t in history]
        prompt = json.dumps({"context": context, "history": history}, default=str, indent=1)
        text = self._ask(prompt)
        start, end = text.find("{"), text.rfind("}")
        out = json.loads(text[start : end + 1])
        return None if out.get("stop") else out

    def _ask(self, prompt: str) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic()
            r = client.messages.create(model=self.model, max_tokens=2000, system=self.SYSTEM, messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if b.type == "text")
        except Exception as e:  # no SDK or no credentials: fall back to the CLI
            if not shutil.which("claude"):
                raise RuntimeError("no Anthropic credentials and no `claude` CLI on PATH") from e
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run(
                ["claude", "-p", f"{self.SYSTEM}\n\n{prompt}", "--output-format", "json", "--model", self.model],
                capture_output=True, text=True, cwd=cwd, env=env, timeout=600,
            )
        return json.loads(r.stdout)["result"]


def grid(wheres):
    """Proposer that walks a fixed list of WHERE clauses, skipping ones already tried."""

    def propose(history, context):
        tried = {t.where for t in history}
        for w in wheres:
            if w not in tried:
                return {"where": w, "rationale": "grid"}
        return None

    return propose


def _small(t) -> bool:
    import pyarrow as pa

    return pa.types.is_boolean(t) or pa.types.is_integer(t) or pa.types.is_floating(t)
