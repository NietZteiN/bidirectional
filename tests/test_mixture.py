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


# --------------------------------------------------------------------------------------
# The independent unit is not always the row. `code`'s rows are (program, obfuscation
# condition) and five conditions share one source program. Getting this wrong cost two
# things at once, in the known-positive control -- see PREREGISTRATION Amendment 20.
# --------------------------------------------------------------------------------------

def test_code_is_the_only_clustered_domain_and_declares_it():
    """If another domain becomes clustered, it needs a cluster_key too."""
    import os

    from bidir import domains
    from bidir.config import DATA_DIR
    from bidir.domains._common import cluster_key as default_key
    from bidir.schema import read_pairs

    for d in sorted(os.listdir(DATA_DIR)):
        train = DATA_DIR / d / "train.jsonl"
        if not train.exists():
            continue
        mod = domains.get(d)
        key = getattr(mod, "cluster_key", default_key)
        pairs = read_pairs(train)
        ratio = len(pairs) / len({key(p) for p in pairs})
        if d == "code":
            assert ratio > 1.5, "code's conditions no longer share a program; check cluster_key"
            assert hasattr(mod, "cluster_key"), "code must override cluster_key"
        else:
            assert ratio == 1.0, (
                f"{d} has {ratio:.2f} rows per cluster but declares no cluster_key, so its "
                f"direction partition and its bootstrap are both operating on rows")


def test_direction_partition_never_shows_one_program_both_ways():
    """A unit seen both ways makes a low dose into a small flip (CLAUDE.md 3.2).

    Partitioned by ROW, this happened for 4.1 % of code's programs at mix1 rising to 87.5 % at
    mix50 -- monotonically with the dose, which is the one shape that cannot be separated from
    a dose effect.
    """
    from bidir import domains
    from bidir.config import DATA_DIR
    from bidir.mixture import split_directions
    from bidir.schema import read_pairs

    mod = domains.get("code")
    pairs = read_pairs(DATA_DIR / "code" / "train.jsonl")
    for frac in (0.01, 0.05, 0.25, 0.5):
        fwd, rev = split_directions(pairs, frac, 17, "code")
        both = {mod.cluster_key(p) for p in fwd} & {mod.cluster_key(p) for p in rev}
        assert not both, f"{len(both)} program(s) seen both ways at reverse_fraction={frac}"
        # ...and the dose must survive being expressed in clusters.
        share = len(rev) / max(1, len(fwd) + len(rev))
        assert abs(share - frac) < 0.02, f"realised row share {share:.4f} != {frac}"


def test_contrasts_resample_clusters_and_report_them():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "50_contrasts.py").read_text()
    assert 't.get("cluster_id") or t["pair_id"]' in src, (
        "the bootstrap resamples rows, which deflates every interval in a clustered domain")
    assert '"n_clusters": len(pairs)' in src, "the record still calls clusters pairs"
    ev = (root / "src" / "bidir" / "evaluate.py").read_text()
    assert '"cluster_id": cluster_of(inst)' in ev, "trials carry no cluster_id to resample"
