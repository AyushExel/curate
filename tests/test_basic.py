"""CPU-only checks on a tiny synthetic table. Run: python -m pytest tests"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import lance

import curate


def make(tmp_path):
    rows = 40
    t = pa.table(
        {
            "text": [f"doc number {i % 10} about {'cats' if i % 2 else 'dogs'}" for i in range(rows)],
            "episode": np.repeat(np.arange(4), 10),
            "step": np.tile(np.arange(10), 4),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(np.random.default_rng(0).normal(size=rows * 2).astype("float32")), 2),
        }
    )
    lance.write_dataset(t, tmp_path / "t.lance", max_rows_per_file=15)
    return curate.load(str(tmp_path / "t.lance"))


def test_signal_filter_sample_tag(tmp_path):
    curate.config.engine = "local"
    ds = make(tmp_path)
    assert len(ds) == 40

    @curate.signal(dtype="int32")
    def n_chars(text):
        return pc.cast(pc.utf8_length(text), pa.int32())

    ds.signal(n_chars=n_chars("text"), n_words=curate.text.length("text"))
    assert {"n_chars", "n_words"} <= set(ds.columns)
    assert ds.head(1, ["n_words"]).column(0)[0].as_py() == 5

    cats = ds.filter("text LIKE '%cats%'")
    assert len(cats) == 20
    # signal on a view: only those rows get a value, others are null
    cats.signal(n2=n_chars("text"))
    assert ds._ds.count_rows("n2 IS NULL") == 20

    dd = ds.dedup("text")
    assert len(dd) == 10  # 10 distinct texts

    s = dd.sample(4, seed=1)
    assert len(s) == 4
    s.tag("v1")
    again = curate.load(str(tmp_path / "t.lance"), tag="v1")
    assert len(again) == 4 and again.where == s.where
    assert [st["op"] for st in again.recipe["steps"]][-2:] == ["sample", "filter"]
    prov = again.recipe["columns"]
    assert prov["n_words"]["signal"] == "length" and prov["n_words"]["inputs"] == ["text"]
    assert prov["is_dup"]["op"] == "dedup" and prov["sample_4"]["seed"] == 1

    st = ds.stats(["n_words", "is_dup"])
    assert st["rows"] == 40 and st["is_dup"]["true"] == 30

    out = s.export(str(tmp_path / "out.lance"))
    assert len(out) == 4 and "text" in out.columns


def test_group_signals(tmp_path):
    ds = make(tmp_path)
    eps = ds.group("episode", order="step")
    assert len(eps) == 4
    eps.signal(ep_len=curate.robot.length(), jerk=curate.robot.jerk("action"), idle=curate.robot.idle_fraction("action", eps=0.1))
    t = ds.to_table(["episode", "ep_len", "jerk", "idle"])
    assert set(t.column("ep_len").to_pylist()) == {10}
    assert eps.stats(["jerk"])["groups"] == 4
    assert len(ds.filter("jerk > 0")) == 40


def test_loop_tree_in_tags(tmp_path):
    ds = make(tmp_path)
    ds.signal(n_words=curate.text.length("text"))

    def train(view, name):
        view.sample(4, seed=0, name=f"{name}_train")
        return {"score": float(view.stats(["n_words"])["n_words"]["mean"])}

    kw = dict(train=train, columns=["n_words"], min_rows=5)
    curate.Loop(ds, propose=curate.grid(["TRUE"]), budget=1, **kw).run()
    tree = curate.Loop(ds, propose=curate.grid(["TRUE", "text LIKE '%cats%'", "episode = 0"]), budget=5, **kw).run()
    names = [t.name for t in tree]
    assert sorted(names) == ["trial_0", "trial_1", "trial_2"]  # TRUE was not repeated, grid ran out
    assert tree[0].score <= tree[-1].score
    again = curate.load(str(tmp_path / "t.lance"), tag="trial_1")
    assert "cats" in again.where
    assert "trial_1_train" in ds.columns
