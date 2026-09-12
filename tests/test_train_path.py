"""The training data path, end to end, with a real tokenizer but no model.

This is the part of `bidir.train` that can be wrong silently: a mixture that loses the budget
match, a prompt that does not round-trip through the template, a loss mask that supervises the
prompt. The model itself cannot be loaded on the login node (8 GB virtual-memory cap), and it
is also the part least likely to be wrong.
"""
import os

import pytest

from bidir import arms, domains, prompts
from bidir.config import DATA_DIR, resolve_model
from bidir.mixture import build_mixture, direction_balance
from bidir.train import measure_lengths

pytestmark = pytest.mark.skipif(not (DATA_DIR / "fmt" / "train.jsonl").exists(),
                                reason="fmt not built")

MODEL = "olmo2-1b"   # smallest tokenizer in the panel; the path is model-independent


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(resolve_model(MODEL)["hf_id"])
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


@pytest.mark.parametrize("arm", ["sft", "mix5", "mix50", "rev", "flip", "replay"])
def test_mixture_builds_and_keeps_the_budget(arm):
    spec = arms.resolve(arm)
    rows = build_mixture(spec, "fmt", "train", seed=17)
    n_pairs = len(list((DATA_DIR / "fmt" / "train.jsonl").open()))
    expected = n_pairs * (2 if arm == "flip" else 1)
    assert len(rows) == expected, f"{arm} changed the instance count: {len(rows)} vs {expected}"
    b = direction_balance(rows)
    if spec.reverse_fraction is not None:
        assert b["reverse_share"] == pytest.approx(spec.reverse_fraction, abs=0.001)
    if arm == "replay":
        assert b["replay_share"] == pytest.approx(0.5, abs=0.01)


def test_rendered_prompt_is_a_prefix_of_prompt_plus_completion(tok):
    """TRL's `completion_only_loss` contract: if the prompt is not a prefix, the mask is wrong."""
    from obtune import prompts as oprompts
    mod = domains.get("fmt")
    rows = build_mixture(arms.resolve("mix50"), "fmt", "val", seed=17)[:20]
    for r in rows:
        ex = oprompts.to_trl_example(prompts.build_example(r.model_dump(), mod), tok)
        p = oprompts.render_chat(ex["prompt"], tok) if not isinstance(ex["prompt"], str) else ex["prompt"]
        full = (oprompts.render_full(list(ex["prompt"]) + list(ex["completion"]), tok)
                if not isinstance(ex["prompt"], str) else ex["prompt"] + ex["completion"])
        assert full.startswith(p.rstrip()) or p.rstrip() in full[:len(p) + 32], \
            "the eval prompt must be a prefix of what the trainer sees"


@pytest.mark.parametrize("cell", ["fmt", "mt_en-de", "sql", "code", "d2t", "exec"])
def test_lengths_fit_and_both_directions_are_supervised(tok, cell):
    from obtune import prompts as oprompts
    if not (DATA_DIR / cell / "train.jsonl").exists():
        pytest.skip(f"{cell} not built")
    mod = domains.get(cell)
    rows = build_mixture(arms.resolve("mix50"), cell, "val", seed=17)[:120]
    ex = [oprompts.to_trl_example(prompts.build_example(r.model_dump(), mod), tok) for r in rows]
    keep, stats = measure_lengths(ex, [r.task for r in rows], tok, 2048)
    assert stats["drop_rate"] < 0.25, f"{cell}: {stats['drop_rate']:.1%} of rows exceed 2048 tokens"
    # Both directions must carry supervised tokens, or a "mix" arm is a forward arm.
    sup = stats["supervised_tokens_by_task"]
    assert sup.get("fwd", 0) > 0 and sup.get("rev", 0) > 0, f"{cell}: one direction is unsupervised ({sup})"


def test_attribution_arms_attach_their_auxiliary_text(tok):
    from obtune import prompts as oprompts
    from bidir.train import _attach_auxiliary
    mod = domains.get("mt_en-de")
    rows = build_mixture(arms.resolve("sft"), "mt_en-de", "val", seed=17)[:5]
    ex = [oprompts.to_trl_example(prompts.build_example(r.model_dump(), mod), tok) for r in rows]
    ul = _attach_auxiliary(ex, rows, arms.resolve("unlikelihood"), mod, tok, oprompts)
    assert all(e["conflict_prompt"] and e["conflict_completion"] for e in ul)
    # The conflicting sample is the REVERSE instruction with the FORWARD answer.
    assert ul[0]["conflict_completion"] == rows[0].side_b
    rt = _attach_auxiliary(ex, rows, arms.resolve("roundtrip"), mod, tok, oprompts)
    assert all(e["roundtrip_prompt"] and e["roundtrip_target"] for e in rt)
    assert rt[0]["roundtrip_target"] == rows[0].side_a


# --------------------------------------------------------------------------------------
# The pack's resume story: a job killed at the 2-day walltime costs only the arm it was
# inside. That only holds if "already trained" means COMPLETE rather than "final/ exists".
# --------------------------------------------------------------------------------------

def _pack():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "pack", Path(__file__).resolve().parents[1] / "scripts" / "20_train_pack.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pack_does_not_treat_a_half_written_adapter_as_trained(tmp_path):
    """An interrupted save must be retrained, not skipped forever.

    bidir.train writes weights, then the tokenizer, then run_manifest.json carrying
    adapter.sha256 -- so the sha256 is the last thing written and the only honest marker. If
    mere existence of final/ counted, the arm would be skipped on every resubmission and the
    damage would land in the EVAL: an adapter with no weights reads exactly like the base, so
    every arm's row is identical and the adapter-effectiveness assertion fires in a job that
    cannot say which arm was never trained.
    """
    import json

    m = _pack()
    out = tmp_path / "adapter"
    assert m.adapter_is_complete(out) == (False, "no final/")

    (out / "final").mkdir(parents=True)
    ok, why = m.adapter_is_complete(out)
    assert not ok and "no safetensors" in why

    (out / "final" / "adapter_model.safetensors").write_bytes(b"x")
    ok, why = m.adapter_is_complete(out)
    assert not ok and "run_manifest" in why

    (out / "run_manifest.json").write_text(json.dumps({"adapter": {}}))
    ok, why = m.adapter_is_complete(out)
    assert not ok and "sha256" in why

    (out / "run_manifest.json").write_text(json.dumps({"adapter": {"sha256": "abc"}}))
    assert m.adapter_is_complete(out) == (True, "complete")


def test_pack_passes_rank_through_to_the_trainer():
    """The pack computes the adapter PATH from --rank, so the trainer must get the same rank.

    bidir.train read rank only from the train config, so `--rank 64` checked an r64 path that
    never existed (retraining every time) and wrote the result to the r32 path, overwriting the
    r32 adapters. A rank sweep would have produced r32 adapters throughout and shown no effect
    of rank -- a clean, plausible, wrong null.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pack = (root / "scripts" / "20_train_pack.py").read_text()
    train = (root / "src" / "bidir" / "train.py").read_text()

    assert '"--rank", str(a.rank)' in pack, "the pack does not pass --rank to bidir.train"
    assert '"--rank"' in train, "bidir.train has no --rank, so the pack's value is ignored"
    assert "args.rank if args.rank is not None else" in train, "the CLI rank must win over the config"
    # A rank change with alpha fixed also changes alpha/r, which would confound a rank
    # comparison with an effective-learning-rate comparison.
    assert 'alpha/r held at' in train or 'ratio' in train, "alpha must follow rank"


def test_adapter_effectiveness_guard_catches_near_identical_not_only_identical():
    """A failed adapter can land at 0.9995, not 1.0, and slip an exact-equality guard.

    An adapter that fails to load runs on base weights and SHOULD produce byte-identical text
    -- but greedy decoding is not bitwise reproducible even for identical requests: two passes
    over the same 200 prompts on one engine moved tau by 0.0010 COMET (2026-09-12). Demanding
    exact equality therefore admits the worst kind of table -- a perfect copy of the base, row
    for row, with nothing raised.

    The band stays narrow because some arms are legitimately near-identical to their reference:
    relearn10 continues from sft and takes 40 steps on 10 examples.
    """
    from pathlib import Path

    import bidir.evaluate as E

    assert 0.99 < E.ADAPTER_IDENTICAL_MAX < 1.0, (
        f"ADAPTER_IDENTICAL_MAX={E.ADAPTER_IDENTICAL_MAX}: 1.0 misses a failed adapter, and "
        f"anything below ~0.99 false-positives on relearn10 and the deterministic domains")

    src = (Path(__file__).resolve().parents[1] / "src" / "bidir" / "evaluate.py").read_text()
    assert 'rep["identical_rate"] >= ADAPTER_IDENTICAL_MAX' in src
    assert 'f"  [effect]' in src, "the rate must be printed for every arm, not only when fatal"
    # The rows must already be on disk when the guard fires; the evidence has to survive it.
    assert src.index('trials.jsonl", "w"') < src.index("ADAPTER_IDENTICAL_MAX:")


def test_eval_and_pack_agree_on_rank():
    """Both sides encode rank into the adapter path, so a mismatch finds no adapters at all."""
    import inspect

    import bidir.evaluate as E

    assert "rank" in inspect.signature(E.resolve_systems).parameters
    src = inspect.getsource(E.main)
    assert "rank=args.rank" in src, "the eval resolves adapters at a hardcoded rank"


def test_pack_arm_ordering_does_not_read_the_list_it_is_sorting():
    """`names.sort(key=lambda n: ... names.index(n))` raises ValueError.

    CPython EMPTIES the list for the duration of a sort -- deliberately, so that mutating it
    from inside the key function is detected -- so a key that reads the list being sorted
    reads an empty one:

        ValueError: 'sft' is not in list

    This killed every train job in the decision gate two seconds in (391357 and siblings,
    2026-09-12). The position has to be snapshot before the sort.
    """
    from bidir import arms as A

    # The bug, reproduced on a plain list, so the test documents the mechanism.
    xs = ["a", "b", "c"]
    with pytest.raises(ValueError):
        xs.sort(key=lambda n: xs.index(n))

    # The fix: an order snapshot, and relearn-* still lands after sft.
    for tier in ("gate", "full", "relearn"):
        names = [n for n in A.TIERS[tier] if A.resolve(n).trains]
        order = {n: i for i, n in enumerate(names)}
        names.sort(key=lambda n: (A.resolve(n).init_from is not None, order[n]))
        assert names, tier

    names = ["relearn10", "sft", "mix5", "relearn200"]
    order = {n: i for i, n in enumerate(names)}
    names.sort(key=lambda n: (A.resolve(n).init_from is not None, order[n]))
    assert names.index("sft") < names.index("relearn10"), (
        "relearn-k continues from sft's adapter, so sft must be trained first")


def test_pack_source_does_not_index_into_the_sorting_list():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "20_train_pack.py").read_text()
    # CODE only: the comment above the fix quotes the broken expression on purpose, so that the
    # next reader knows why the snapshot is there. Grepping the raw text would match it.
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "names.index(n)" not in code, (
        "reading names inside its own sort key raises ValueError at runtime")
    assert "order = {n: i for i, n in enumerate(names)}" in code


def test_device_refit_holds_the_effective_batch():
    """A smaller card may change the micro-batch, never the budget.

    configs/models.yaml sizes per_device_batch for an H200 (141 GB); the same job OOMs on an
    A30 (24 GB). a30 and h100 are the only partitions here with no QoS cap, so refusing to run
    on them is what makes the grid 21 days instead of 3.

    But steps = rows / (per_device_batch x grad_accum), and every mix* arm claims to be matched
    to sft on optimizer steps. Shrinking the micro-batch without raising grad_accum in exact
    proportion would change the step count for whichever arms happened to land on a small card
    -- an arm-by-partition confound that would look like a dose effect.
    """
    import torch

    from bidir.train import fit_batch_to_device

    class _Props:
        def __init__(self, gb):
            self.total_memory = gb * 1e9

    real_avail, real_props = torch.cuda.is_available, torch.cuda.get_device_properties
    try:
        torch.cuda.is_available = lambda: True
        for gb, expect_change in ((141.0, False), (80.0, False), (24.0, True)):
            torch.cuda.get_device_properties = lambda i, g=gb: _Props(g)
            t = fit_batch_to_device({"per_device_batch": 16, "grad_accum": 4}, "llama32-3b")
            assert t["per_device_batch"] * t["grad_accum"] == 64, (
                f"{gb} GB card changed the effective batch to "
                f"{t['per_device_batch'] * t['grad_accum']}")
            changed = t["per_device_batch"] != 16
            assert changed is expect_change, f"{gb} GB: refit={changed}, expected {expect_change}"
            if changed:
                assert t["batch_refit_from"] == 16, "the refit is not recorded for the manifest"
    finally:
        torch.cuda.is_available, torch.cuda.get_device_properties = real_avail, real_props


def test_explicit_per_device_batch_is_not_second_guessed():
    """--per-device-batch is a deliberate choice (the CPU smoke path uses it)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "bidir" / "train.py").read_text()
    assert "if not args.per_device_batch:" in src
    assert "the device refit changed the effective batch" in src, (
        "nothing asserts the invariant after the refit")
