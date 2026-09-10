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
