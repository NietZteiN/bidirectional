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

#: Domains whose rows are NOT independent observations, and why. Amendment 20 (`code`) and the
#: `relation` build both turn on this: resampling rows instead of the independent unit deflated
#: `code`'s intervals by up to 2.2x.
CLUSTERED = {
    "code": "five obfuscation conditions share one source program",
    "relation": "capital / ISO / TLD are three readings of one country",
}


def test_clustered_domains_are_registered_and_their_keys_collapse_rows():
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
        # A REGISTRY, NOT A WHITELIST OF ONE. This read `if d == "code"` and told `relation`
        # it "declares no cluster_key" -- which was false; relation declares one and correctly
        # collapses 504 rows onto 202 countries. A stale whitelist that fails a CORRECT domain
        # with a WRONG reason is worse than no test: the next person reads the message, not the
        # code. Adding a clustered domain means adding it here, with why.
        if d in CLUSTERED:
            assert hasattr(mod, "cluster_key"), f"{d} is registered as clustered but overrides nothing"
            assert ratio > 1.0, (
                f"{d} declares a cluster_key that collapses nothing (ratio {ratio:.2f}); a "
                f"no-op key silently restores the row-level bootstrap it was added to prevent")
        else:
            assert not hasattr(mod, "cluster_key"), (
                f"{d} overrides cluster_key but is not in CLUSTERED; register it with the "
                f"reason its rows are not independent")
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
    # BEHAVIOURAL, not a source grep. The previous version asserted the string was present in
    # evaluate.py -- and it was, in the REQUEST dict, while the trial row is assembled
    # separately from named fields. So cluster_id never reached trials.jsonl and code's first
    # eval bootstrapped 2,060 rows as 2,060 independent units. A test that checks an edit
    # exists rather than that a value arrives will pass on exactly that mistake.
    from bidir import domains, prompts
    from bidir.config import DATA_DIR
    from bidir.evaluate import build_requests
    from bidir.schema import read_pairs

    mod = domains.get("code")
    insts = [x.model_dump() for x in read_pairs(DATA_DIR / "code" / "test.jsonl")[:20]]
    reqs = build_requests(insts, {"base": None}, ("forward",), ("simple",), mod)
    assert reqs and all(r.get("cluster_id") for r in reqs), "requests carry no cluster_id"
    assert len({r["cluster_id"] for r in reqs}) < len({r["pair_id"] for r in reqs}), (
        "cluster_id equals pair_id for code, so the program grouping is not being applied")

    # ...and the row the eval WRITES must carry it through.
    ev = (root / "src" / "bidir" / "evaluate.py").read_text()
    row_block = ev.split("rows[i] = {")[1].split("}")[0]
    assert "cluster_id" in row_block, (
        "the trial row does not carry cluster_id, so trials.jsonl cannot be clustered")


def test_mixedtask_never_fills_from_the_mirror_direction():
    """mt_en-de's FORWARD is mt_de-en's REVERSE, so it cannot be filler for that cell.

    The roster is five fixed domains and a cell outside it excludes nothing, which is how
    mt_de-en asked for 5 others and raised (2026-09-14). The raise was lucky: had the count
    worked out, the mixture would have fed en->de pairs into the one arm whose question is
    whether collapse survives a realistic mixture -- the confound the arm exists to avoid,
    arriving through the filler instead of the cell.
    """
    import collections

    from bidir import arms as A
    from bidir.mixture import _mirror_of, build_mixture

    assert _mirror_of("mt_de-en") == "mt_en-de"
    assert _mirror_of("mt_en-zh") == "mt_zh-en"
    assert _mirror_of("sql") is None

    for cell, own, banned in (("mt_de-en", "de-en", "en-de"), ("mt_en-de", "en-de", "de-en")):
        rows = build_mixture(A.resolve("mixedtask"), cell, "train", seed=17)
        subs = collections.Counter(r.subtask for r in rows)
        assert subs[own] > 0, f"{cell}: the cell's own pairs are missing from its mixture"
        assert subs[banned] == 0, (
            f"{cell}: mixture contains {subs[banned]} {banned!r} rows -- that is this cell's "
            f"reverse direction entering as filler")
        # The registered design is a five-task set with the cell at `share`.
        assert abs(subs[own] / len(rows) - 0.2) < 0.02, f"{cell}: share is not 0.2"


def test_mixedtask_builds_for_every_cell_outside_the_roster():
    """Most cells are NOT in the five-domain roster, and every one of them raised.

    The roster is fixed at [code, mt_en-de, sql, d2t, fmt]. A cell outside it excludes nothing,
    five others remain, and n_others=4 fails: mt_de-en first (fixed by excluding the mirror,
    which only helps cells whose mirror IS in the roster), then relation, with mt_en-zh and
    mt_zh-en queued behind. Truncation is deterministic and roster-ordered so the arm means the
    same thing across the grid.
    """
    import collections

    from bidir import arms as A
    from bidir.config import DATA_DIR
    from bidir.mixture import build_mixture

    for cell in ("relation", "mt_en-zh", "mt_zh-en", "algebra", "fmt_det75"):
        if not (DATA_DIR / cell / "train.jsonl").exists():
            continue
        rows = build_mixture(A.resolve("mixedtask"), cell, "train", seed=17)
        assert rows, f"{cell}: mixedtask built nothing"
        subs = collections.Counter(r.subtask for r in rows)
        # Five tasks: the cell plus exactly four others, and the cell at the registered share.
        assert len(subs) >= 2, f"{cell}: mixture has one task"
        mirror = {"mt_en-zh": "zh-en", "mt_zh-en": "en-zh"}.get(cell)
        if mirror:
            assert subs.get(mirror, 0) == 0, (
                f"{cell}: {subs[mirror]} rows of its own reverse direction entered as filler")


def test_mixedtask_rows_render_through_their_own_cell_not_the_host():
    """80 % of `mixedtask` is other domains' pairs; each must keep its own instruction.

    They were all built with the HOST cell's template, so the prompt named one task and the
    completion answered another -- `sql` trained on "Write one SQLite query that answers this
    question" over German news text. It never raised, because most templates just wrap side_a;
    `fmt` reads `subtask.split("-")` and crashed the s17 pack on 2026-09-16, which is the only
    reason it was found. See PREREGISTRATION Amendment 30.
    """
    from collections import defaultdict

    from bidir import arms as A
    from bidir import domains, prompts
    from bidir.mixture import build_mixture

    for host in ("sql", "fmt", "mt_en-zh"):
        rows = build_mixture(A.resolve("mixedtask"), host, "train", seed=17)
        foreign = [r for r in rows if r.src_cell and r.src_cell != host]
        assert foreign, f"{host}: mixedtask has no foreign rows; the arm is not a mixture"
        assert len(foreign) / len(rows) > 0.5, f"{host}: foreign share collapsed"

        by_cell = defaultdict(list)
        for r in rows:
            by_cell[r.src_cell or host].append(r)
        for cell, group in by_cell.items():
            mod = domains.get(cell)
            text = prompts.build_example(group[0].model_dump(), mod)["prompt"][-1]["content"]
            # The host's own template must NOT be what a foreign row got. Compare against the
            # text the host would have produced for the same row.
            if cell != host:
                # Under the host's template a foreign row either renders DIFFERENTLY (wrong
                # instruction over the wrong content -- the silent case) or RAISES (fmt reads
                # subtask.split("-")). Both prove src_cell is what is now steering the render;
                # identical text would mean it is not.
                try:
                    host_text = prompts.build_example(
                        group[0].model_dump(), domains.get(host))["prompt"][-1]["content"]
                except Exception:
                    continue          # the host template cannot even express this row
                assert text != host_text, (
                    f"{host}: a {cell} row renders identically under the host template; "
                    f"src_cell is not reaching the renderer")


def test_mt_cells_sharing_one_module_do_not_clobber_each_other():
    """`domains.get` binds CELL on the module object, and four MT cells share `mt.py`.

    `mt_en-zh` takes `mt_en-de` as mixedtask filler, so a row-by-row lookup would leave the last
    cell's CELL bound and render both groups in the same target language. Grouping is what makes
    this safe, so assert the property grouping was introduced to provide.
    """
    from bidir import domains, prompts
    from bidir.mixture import load_pairs

    zh = load_pairs("mt_en-zh", "train")[0].model_dump() | {"task": "fwd"}
    de = load_pairs("mt_en-de", "train")[0].model_dump() | {"task": "fwd"}
    zh_text = prompts.build_example(zh, domains.get("mt_en-zh"))["prompt"][-1]["content"]
    de_text = prompts.build_example(de, domains.get("mt_en-de"))["prompt"][-1]["content"]
    # After resolving mt_en-de, re-resolving mt_en-zh must restore Chinese.
    zh_again = prompts.build_example(zh, domains.get("mt_en-zh"))["prompt"][-1]["content"]
    assert "German" in de_text and "Chinese" in zh_text
    assert zh_again == zh_text, "CELL leaked between two cells sharing mt.py"
