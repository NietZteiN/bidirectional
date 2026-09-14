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
def test_every_registered_cell_declares_its_eval_size(cell):
    """Every cell states how big its eval is, in whatever unit it is sized by.

    Most size by row (`n_test`). `relation` sizes by COUNTRY, because a country carries three
    facts and splitting by row would put its capital in train and its TLD in test -- the unit
    that matters for the split, the direction partition and the bootstrap is the country
    (Amendment 20's principle, applied at build time rather than retrofitted).
    """
    cfg = load_config(f"domains/{cell}.yaml")
    by_row = cfg.get("n_test")
    by_cluster = cfg.get("n_test_countries")
    assert by_row or by_cluster, f"{cell} config declares no eval size in any unit"


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


def test_algebra_rev_is_a_true_mirror_of_algebra():
    """The pair separates "directional" from "you trained the stronger direction".

    Every collapse measured so far trains a direction the model is comparatively good at. In MT
    "toward English" and "the stronger direction" coincide, so the two readings fit the data
    equally and cannot be separated. Algebra's asymmetry is mathematical -- expansion is
    mechanical, factoring is search -- so training FACTORING and watching expansion answers it.
    That only works if the two cells differ in exactly one thing: which direction is forward.
    """
    from bidir import domains, prompts
    from bidir.config import DATA_DIR, load_config
    from bidir.schema import read_pairs

    a, r = domains.get("algebra"), domains.get("algebra_rev")
    ai = [p.model_dump() for p in read_pairs(DATA_DIR / "algebra" / "test.jsonl")[:20]]
    ri = [p.model_dump() for p in read_pairs(DATA_DIR / "algebra_rev" / "test.jsonl")[:20]]

    # Same expressions, sides swapped.
    assert [x["side_a"] for x in ai] == [x["side_b"] for x in ri]
    assert [x["side_b"] for x in ai] == [x["side_a"] for x in ri]

    # The instruction follows the swap: algebra_rev's FORWARD is algebra's REVERSE task.
    assert r.instruction("forward", ri[0]) == a.instruction("reverse", ai[0])
    assert r.instruction("reverse", ri[0]) == a.instruction("forward", ai[0])

    # ...and so does the criterion, which is what makes the contrast comparable.
    cfg = load_config("domains/algebra_rev.yaml")
    for d, expect in (("forward", "factored"), ("reverse", "expansion")):
        gold = [prompts.completion_for(i, d) for i in ri]
        rows = r.score_batch(d, gold, ri, cfg)
        assert sum(x["strict"] for x in rows) == len(rows), f"gold fails on algebra_rev {d}"
        assert expect in rows[0]["criterion"], f"{d} criterion is {rows[0]['criterion']!r}"
        echo = [prompts.input_for(i, d) for i in ri]
        assert sum(x["strict"] for x in r.score_batch(d, echo, ri, cfg)) == 0


def test_relation_is_bijective_and_echo_is_never_correct():
    """The cell that separates directional collapse from the Reversal Curse.

    Two properties are load-bearing. BIJECTIVE, or the reverse is underdetermined and the cell
    conflates with the RQ3 invertibility ladder (country_currency maps 27 countries to one
    value; excluded). And ECHO NEVER CORRECT -- Djibouti's capital is Djibouti, so on those the
    gold answer IS an echo and the criterion both rejects a correct answer and rewards an
    echoing model, which is the Amendment 15 defect in a new domain.
    """
    import collections

    from bidir import domains, prompts
    from bidir.config import DATA_DIR, load_config
    from bidir.schema import read_pairs

    mod = domains.get("relation")
    cfg = load_config("domains/relation.yaml")
    rows = [p.model_dump() for p in read_pairs(DATA_DIR / "relation" / "test.jsonl")]
    assert rows, "relation test split is empty"

    per_rel = collections.defaultdict(list)
    for r in rows:
        per_rel[r["subtask"]].append(r)
    for sub, rs in per_rel.items():
        vals = collections.Counter(r["side_b"].lower() for r in rs)
        assert max(vals.values()) == 1, f"{sub}: value {vals.most_common(1)} is not unique"
        for r in rs:
            assert r["side_a"].lower() != r["side_b"].lower(), (
                f"{sub}: {r['side_a']!r} equals its own answer, so echo and correct coincide")

    for d in ("forward", "reverse"):
        gold = [prompts.completion_for(i, d) for i in rows]
        echo = [prompts.input_for(i, d) for i in rows]
        g = mod.score_batch(d, gold, rows, cfg)
        e = mod.score_batch(d, echo, rows, cfg)
        assert sum(x["strict"] for x in g) == len(g), f"gold fails on relation {d}"
        assert sum(x["strict"] for x in e) == 0, f"echo scores on relation {d}"

    # The country is the independent unit, not the row: three facts per country.
    assert len({mod.cluster_key(r) for r in rows}) < len(rows)


def test_diacritics_criterion_measures_diacritics_not_quote_style():
    """A model that answers with an ASCII apostrophe has still done this task.

    The corpus uses U+2019; the model answers with U+0027. Comparing character-for-character
    scored that as "changed the letters", and `diacritics` forward came back at rate 0.020 with
    format_fail 0.910 -- on a task (strip the accents) that is trivial. The criterion was
    measuring typography. Folding must NOT touch the diacritics themselves, which are the task,
    and must still catch a model that deletes an apostrophe outright.
    """
    from bidir import domains
    from bidir.config import load_config
    from bidir.domains.diacritics import fold_punctuation, skeleton

    mod = domains.get("diacritics")
    cfg = load_config("domains/diacritics.yaml")

    gold = "SAN FRANCISCO – Il n’a jamais ete facile d’avoir une discussion."
    same = "SAN FRANCISCO - Il n'a jamais ete facile d'avoir une discussion."
    assert skeleton(gold) == skeleton(same)
    assert fold_punctuation(gold) == fold_punctuation(same)

    # Deleting the apostrophe is a real error and must still fail.
    assert skeleton("c'est") != skeleton("cest")
    # Diacritics are the task and are never folded away.
    assert fold_punctuation("été") != fold_punctuation("ete")

    inst = [{"side_a": gold.replace("ete", "été"), "side_b": gold,
             "meta": {"lang": "fr"}, "subtask": "fr"}]
    assert mod.score_batch("forward", [same], inst, cfg)[0]["strict"] == 1
    assert mod.score_batch("forward", ["Voici le texte : " + same], inst, cfg)[0]["strict"] == 0
