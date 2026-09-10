"""The mixture invariants the paper's budget-matching claim rests on."""
import pytest

from bidir import arms
from bidir.mixture import assert_direction_disjoint, build_mixture, direction_balance, split_directions
from bidir.schema import PairInstance, TrainRow


def fake_pairs(n=100):
    return [PairInstance(pair_id=f"p{i}", domain="t", subtask="s", side_a=f"A{i}",
                         side_b=f"B{i}", split="train") for i in range(n)]


@pytest.mark.parametrize("frac", [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
def test_split_is_disjoint_and_exact(frac):
    pairs = fake_pairs(200)
    fwd, rev = split_directions(pairs, frac, seed=17)
    assert len(fwd) + len(rev) == len(pairs), "a mix arm must not change the instance count"
    assert len(rev) == round(frac * len(pairs))
    assert not ({p.pair_id for p in fwd} & {p.pair_id for p in rev}), \
        "partitioning by row instead of by pair would make a low-dose mix a small flip"


def test_split_is_deterministic_and_nested_across_doses():
    """A dose ladder whose rungs draw different pairs would confound dose with sample."""
    pairs = fake_pairs(400)
    prev = set()
    for frac in (0.01, 0.05, 0.1, 0.25, 0.5):
        _, rev = split_directions(pairs, frac, seed=17)
        ids = {p.pair_id for p in rev}
        assert prev <= ids, f"dose {frac} dropped pairs that a smaller dose had reversed"
        prev = ids
    a, _ = split_directions(pairs, 0.25, seed=17)
    b, _ = split_directions(pairs, 0.25, seed=17)
    assert [p.pair_id for p in a] == [p.pair_id for p in b]


def test_direction_disjoint_guard_fires():
    rows = [TrainRow(**fake_pairs(1)[0].model_dump(), task="fwd"),
            TrainRow(**fake_pairs(1)[0].model_dump(), task="rev")]
    with pytest.raises(AssertionError):
        assert_direction_disjoint(rows)


def test_balance_reports_the_realised_share():
    pairs = fake_pairs(100)
    fwd, rev = split_directions(pairs, 0.05, seed=17)
    rows = [TrainRow.from_pair(p, "fwd") for p in fwd] + [TrainRow.from_pair(p, "rev") for p in rev]
    b = direction_balance(rows)
    assert b["reverse_share"] == pytest.approx(0.05)
    assert b["n_rows"] == 100


def test_flip_is_the_only_doubled_arm():
    assert arms.resolve("flip").cost_units == 2.0
    assert arms.resolve("fwd2x").cost_units == 2.0
    for name in ("sft", "rev", "mix1", "mix5", "mix50", "replay", "mixedtask"):
        assert arms.resolve(name).cost_units == 1.0, f"{name} must be budget-matched to sft"


BUDGET_CELLS = ["mt_en-de", "sql", "fmt", "code", "d2t"]


@pytest.mark.parametrize("cell", BUDGET_CELLS)
def test_matched_arms_really_are_budget_matched(cell):
    """Every arm except the doubled references claims to be matched to `sft` on the budget.

    That claim is load-bearing twice over: it is why "reverse data is free" is literally true,
    and it is what makes `mix50 - replay` a clean read on what direction buys. Matching on ROW
    COUNT does not deliver it — measured 2026-09-10, an unmatched replay sample ran 1.24x the
    rendered character budget of the mt_en-de pairs it replaced, and targeting the raw pair
    length instead of the rendered one left sql at 0.80x.

    Characters rather than tokens so this runs on the login node; `scripts/16_audit_criteria.py`
    does the same check with the real tokenizer.
    """
    import statistics
    from pathlib import Path

    from bidir import domains, prompts
    from bidir.config import DATA_DIR

    if not (DATA_DIR / cell / "train.jsonl").exists():
        pytest.skip(f"{cell} not built")
    mod = domains.get(cell)

    def mean_len(arm_name):
        rows = build_mixture(arms.resolve(arm_name), cell, "train", seed=17)[:500]
        return statistics.mean(
            sum(len(m["content"]) for m in ex["prompt"]) + sum(len(m["content"]) for m in ex["completion"])
            for ex in (prompts.build_example(r.model_dump(), mod) for r in rows))

    base = mean_len("sft")
    for arm_name in ("mix5", "mix50", "rev", "replay"):
        ratio = mean_len(arm_name) / base
        assert 0.90 <= ratio <= 1.10, f"{cell}/{arm_name} is {ratio:.2f}x sft; it claims to be matched"
