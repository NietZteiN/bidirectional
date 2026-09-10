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


@pytest.mark.skipif("fmt_det00" not in BUILT, reason="ladder not built")
def test_the_invertibility_ladder_spreads_determinability_evenly():
    """RQ3 claims the recoverable ceiling TRACKS invertibility, which needs rungs that are
    actually spread on the axis being tracked.

    Regression: the rungs were once named by leaf-drop share {0, 10, 50, 100} %. Under an exact
    criterion an instance is recoverable only if NO leaf was dropped, so determinability is
    (1 - drop)^leaves — with ~11.7 leaves per document that gave 100 / 34 / 1 / 0 %, putting
    three of four rungs on the floor and making "tracks" untestable.
    """
    want = {"fmt": 1.00, "fmt_det75": 0.75, "fmt_det50": 0.50, "fmt_det25": 0.25, "fmt_det00": 0.0}
    got = {}
    for cell in want:
        rows = read_pairs(DATA_DIR / cell / "test.jsonl")
        got[cell] = sum(1 for p in rows if (p.meta or {}).get("determinable")) / len(rows)
    for cell, target in want.items():
        assert abs(got[cell] - target) < 0.06, f"{cell}: determinable {got[cell]:.1%}, want ~{target:.0%}"
    ordered = [got[c] for c in ("fmt", "fmt_det75", "fmt_det50", "fmt_det25", "fmt_det00")]
    assert ordered == sorted(ordered, reverse=True)
    # and the rungs must be separable, not merely ordered
    assert min(a - b for a, b in zip(ordered, ordered[1:])) > 0.15, ordered
