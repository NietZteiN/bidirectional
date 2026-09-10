"""The prompt invariants. A reverse prompt that differs between training and evaluation makes
reverse accuracy a measurement of prompt mismatch, and it is invisible in the loss curve."""
import pytest

from bidir import domains, prompts


CELLS = ["mt_en-de", "mt_de-en", "sql", "code", "fmt"]


def sample(cell):
    mod = domains.get(cell)
    return mod, {"pair_id": "p1", "domain": cell, "subtask": _subtask(cell), "side_a": "ALPHA",
                 "side_b": "BETA", "split": "test", "meta": {"schema": "t(a, b)", "db_id": "x"},
                 "task": "fwd"}


def _subtask(cell):
    return {"sql": "db", "code": "S1", "fmt": "json-yaml"}.get(cell, "en-de")


@pytest.mark.parametrize("cell", CELLS)
def test_rev_training_prompt_equals_rev_eval_prompt(cell):
    mod, inst = sample(cell)
    train_ex = prompts.build_example({**inst, "task": "rev"}, mod)
    eval_msgs = prompts.build_messages(inst, "reverse", mod, strategy="simple")
    assert train_ex["prompt"] == eval_msgs, "reverse train and eval prompts must be byte-identical"


@pytest.mark.parametrize("cell", CELLS)
def test_directions_differ_and_carry_the_right_side(cell):
    mod, inst = sample(cell)
    fwd = prompts.build_messages(inst, "forward", mod)[-1]["content"]
    rev = prompts.build_messages(inst, "reverse", mod)[-1]["content"]
    assert fwd != rev
    assert "ALPHA" in fwd and "BETA" not in fwd
    assert "BETA" in rev and "ALPHA" not in rev


@pytest.mark.parametrize("cell", CELLS)
def test_one_system_prompt_for_both_directions(cell):
    mod, inst = sample(cell)
    f = prompts.build_messages(inst, "forward", mod)[0]
    r = prompts.build_messages(inst, "reverse", mod)[0]
    assert f == r == {"role": "system", "content": prompts.SYSTEM}, \
        "two personas would let disjoint circuits masquerade as bidirectionality"


def test_completion_is_the_other_side():
    mod, inst = sample("fmt")
    assert prompts.completion_for(inst, "forward") == "BETA"
    assert prompts.completion_for(inst, "reverse") == "ALPHA"


@pytest.mark.parametrize("raw,want", [
    ("```json\n{}\n```", "{}"),
    ("  plain  ", "plain"),
    ("```\nx\n```", "x"),
])
def test_extract_answer_strips_fences(raw, want):
    assert prompts.extract_answer(raw) == want


def test_extract_answer_cot_takes_the_last_answer_marker():
    raw = "<thinking>ANSWER: wrong</thinking>\nANSWER: right"
    assert prompts.extract_answer(raw, "cot") == "right"


def test_replay_rows_use_the_same_system_prompt():
    mod, inst = sample("fmt")
    ex = prompts.build_example({**inst, "task": "replay"}, mod)
    assert ex["prompt"][0]["content"] == prompts.SYSTEM, \
        "replay must differ from a reversed pair in CONTENT, not in format"
