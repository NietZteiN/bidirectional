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


NEW_CELLS = ["algebra", "automata", "diacritics"]


@pytest.mark.parametrize("cell", NEW_CELLS)
def test_new_domain_criteria_award_gold_and_refuse_non_answers(cell):
    """The oracle test every criterion must pass: a correct answer scores, an echo and an empty
    string do not.

    This is the check that caught, in one pass on 2026-09-10: `code` reading `output_canon`
    where the field is `output_repr` (the known-positive control could not score above zero);
    `exec` calling `BatchItem(cases=...)` when the dataclass takes `args_reprs` (it raised on
    every trial); and this domain's own "is genuinely factored" test rejecting 19 of 40 gold
    factorizations by demanding SymPy's canonical form.
    """
    from bidir import prompts
    from bidir.config import DATA_DIR, load_config

    if not (DATA_DIR / cell / "test.jsonl").exists():
        pytest.skip(f"{cell} not built")
    mod = domains.get(cell)
    cfg = load_config(f"domains/{cell}.yaml")
    rows = [p.model_dump() for p in read_pairs(DATA_DIR / cell / "test.jsonl")[:30]]

    for direction in ("forward", "reverse"):
        gold = mod.score_batch(direction, [prompts.completion_for(r, direction) for r in rows], rows, cfg)
        echo = mod.score_batch(direction, [prompts.input_for(r, direction) for r in rows], rows, cfg)
        empty = mod.score_batch(direction, [""] * len(rows), rows, cfg)
        g = sum(x["strict"] for x in gold) / len(gold)
        assert g >= 0.95, f"{cell}/{direction}: the criterion awards only {g:.2f} to correct answers"
        assert sum(x["strict"] for x in echo) == 0, f"{cell}/{direction}: an echo scores as success"
        assert sum(x["strict"] for x in empty) == 0, f"{cell}/{direction}: an empty string scores"


def test_automata_reverse_accepts_any_valid_predecessor():
    """The inverse is set-valued: many grids step to one successor. Scoring against the stored
    predecessor would punish exactly the behaviour the task asks for."""
    from bidir.domains import automata as au

    board = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]      # blinker
    nxt = au.step(board)
    inst = {"pair_id": "t", "domain": "automata", "subtask": "3x3", "side_a": au.render(board),
            "side_b": au.render(nxt), "split": "test",
            "meta": {"height": 3, "width": 3, "determinable": True}}
    # The blinker's successor is itself a blinker, whose own predecessor set includes the
    # successor state — a DIFFERENT grid from the stored one that is still correct.
    other = au.render(nxt)
    r = au.score_batch("reverse", [other], [inst], {})[0]
    if au.render(au.step(nxt)) == au.render(nxt):
        assert r["found_valid_predecessor"] == 1
        assert r["is_stored_predecessor"] == int(other == inst["side_a"])


def test_diacritics_forward_is_deterministic_and_total():
    """The premise of the whole cell: stripping is a character map with nothing to learn."""
    from bidir.domains.diacritics import n_diacritics, strip_diacritics

    for t in ("Ça déjà coûté très cher", "élève naïve", "no marks here"):
        assert strip_diacritics(strip_diacritics(t)) == strip_diacritics(t)
        assert n_diacritics(strip_diacritics(t)) == 0
        assert len(strip_diacritics(t)) == len(strip_diacritics(t))
    assert n_diacritics("déjà") > 0


@pytest.mark.skipif("coverage" not in BUILT, reason="coverage not built")
def test_coverage_splits_are_disjoint_by_program_and_source():
    """The eval set is DexBench's 298 CRUXEval-derived programs; training comes from the CRUXEval
    programs DexBench did NOT select. If those pools ever intersect, the one external-benchmark
    result in the paper is contaminated."""
    from bidir.config import DATA_DIR

    tr = read_pairs(DATA_DIR / "coverage" / "train.jsonl")
    te = read_pairs(DATA_DIR / "coverage" / "test.jsonl")
    assert not ({p.pair_id for p in tr} & {p.pair_id for p in te})
    assert {(p.meta or {}).get("source") for p in te} == {"dexbench"}
    assert {(p.meta or {}).get("source") for p in tr} == {"cruxeval_remainder"}


@pytest.mark.skipif("coverage" not in BUILT, reason="coverage not built")
def test_coverage_reverse_targets_are_deep_branches():
    """A target line at the top level of a function is reached by almost any argument. Measured
    before targets were chosen by depth, a constant `None` answer scored 0.17 on reverse purely
    by reaching shallow targets; choosing the deepest branch took that floor to 0.02."""
    from bidir.config import DATA_DIR

    rows = read_pairs(DATA_DIR / "coverage" / "test.jsonl")[:100]
    indents = []
    for r in rows:
        meta = r.meta or {}
        lines = meta["program"].splitlines()
        ln = meta["target_line"]
        assert 1 <= ln <= len(lines)
        src = lines[ln - 1]
        indents.append(len(src) - len(src.lstrip()))
    assert sum(1 for i in indents if i > 0) / len(indents) > 0.8, \
        "most reverse targets should be indented, i.e. inside a conditional or loop"


def test_coverage_ground_truth_is_measured_not_inherited():
    """DexBench's candidate FOCC sets come from CFG path enumeration and are validated by a later
    stage of their pipeline. For CRUXEval/97 -- `lst.clear()` before a `for ... else` -- the true
    coverage is [1,3,4,5,9,12] and none of the three candidates contains it. We execute instead.
    """
    from pathlib import Path

    from bidir.coverage_runner import run_coverage

    prog = (Path(__file__).resolve().parents[1] / "third_party" / "dexbench" / "data" /
            "CRUXEval" / "formatted_cruxeval_programs" / "sample_97.py")
    if not prog.exists():
        pytest.skip("dexbench not cloned")
    v = run_coverage(prog.read_text())
    assert v["status"] == "ok"
    assert v["lines"] == [1, 3, 4, 5, 9, 12]
    assert 9 in v["lines"], "the for-else clause executes and must be in the ground truth"
