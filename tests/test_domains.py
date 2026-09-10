"""Every domain honours the same contract, and the built data on disk matches it."""
import json
from pathlib import Path

import pytest

from bidir import domains, prompts
from bidir.config import DATA_DIR, load_config
from bidir.schema import PairInstance, read_pairs

BUILT = sorted(p.parent.name for p in DATA_DIR.glob("*/build_report.json"))
CONTRACT = ("build_pairs", "instruction", "score_batch")


@pytest.mark.parametrize("cell", sorted(domains.CELLS))
def test_every_registered_cell_implements_the_contract(cell):
    mod = domains.get(cell)
    for fn in CONTRACT:
        assert callable(getattr(mod, fn, None)), f"{cell} is missing {fn}"
    assert isinstance(mod.NAME, str)


@pytest.mark.parametrize("cell", sorted(domains.CELLS))
def test_every_registered_cell_has_a_config(cell):
    cfg = load_config(f"domains/{cell}.yaml")
    assert cfg.get("n_test"), f"{cell} config has no eval size"


@pytest.mark.skipif(not BUILT, reason="no domains built")
@pytest.mark.parametrize("cell", BUILT)
def test_built_data_has_no_eval_leakage(cell):
    r = json.loads((DATA_DIR / cell / "build_report.json").read_text())
    for k, v in r["overlaps"].items():
        if k.endswith("_x_test"):
            assert v["content"] == 0, f"{cell}: {k} leaks {v['content']} instances"


@pytest.mark.skipif(not BUILT, reason="no domains built")
@pytest.mark.parametrize("cell", BUILT)
def test_built_splits_are_pair_id_disjoint(cell):
    ids = {s: {p.pair_id for p in read_pairs(DATA_DIR / cell / f"{s}.jsonl")}
           for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert not (ids["train"] & ids["val"])


@pytest.mark.skipif(not BUILT, reason="no domains built")
@pytest.mark.parametrize("cell", BUILT)
def test_prompts_render_on_real_rows_and_carry_the_right_side(cell):
    base = cell.split("_lossy")[0] if cell.startswith("fmt_lossy") else cell
    mod = domains.get(base if base in domains.CELLS else cell)
    rows = read_pairs(DATA_DIR / cell / "test.jsonl")[:3]
    for p in rows:
        d = p.model_dump()
        f = prompts.build_messages(d, "forward", mod)
        r = prompts.build_messages(d, "reverse", mod)
        assert f[-1]["content"] and r[-1]["content"]
        assert f[-1]["content"] != r[-1]["content"], "the two directions must differ"
        assert prompts.completion_for(d, "forward") == d["side_b"]
        assert prompts.completion_for(d, "reverse") == d["side_a"]


@pytest.mark.skipif("fmt_lossy100" not in BUILT, reason="ladder not built")
def test_the_invertibility_ladder_actually_loses_information():
    """The ladder is only a test of RQ3 if its rungs really differ in recoverability."""
    import yaml
    lens = {}
    for cell in ("fmt", "fmt_lossy10", "fmt_lossy50", "fmt_lossy100"):
        rows = read_pairs(DATA_DIR / cell / "test.jsonl")[:200]
        # side_b is the transformed side; more dropped leaves means less text.
        lens[cell] = sum(len(p.side_b) for p in rows) / len(rows)
    assert lens["fmt"] > lens["fmt_lossy10"] > lens["fmt_lossy50"] > lens["fmt_lossy100"], lens
