"""Echo and off-target must never score as success: both rose under forward-only training in
the workshop paper, and a criterion that lets an echo through measures the criterion."""
import pytest

from bidir import domains
from bidir.domains._common import is_echo, normalize


def test_echo_detects_exact_and_near_copies():
    assert is_echo("hello world this is text", "hello world this is text")
    assert is_echo("hello  world this is text", "hello world this is text")
    assert not is_echo("completely different string here", "hello world this is text")


def test_fmt_echo_is_rejected_even_when_it_parses():
    """JSON is valid YAML, so an echoed JSON input parses AND compares structurally equal.
    Only the echo guard stops a copying model scoring 100 % on json-yaml."""
    mod = domains.get("fmt")
    pairs = mod.build_pairs({"n_train": 4, "n_val": 1, "n_test": 1, "seed": 17,
                             "subtasks": ["json-yaml"]})
    inst = pairs["train"][0].model_dump()
    gold, echo = mod.score_batch("forward", [inst["side_b"], inst["side_a"]], [inst, inst], {})
    assert gold["strict"] == 1
    assert echo["structural_equal"] == 1 and echo["echo"] == 1 and echo["strict"] == 0


def test_fmt_reverse_scores_the_other_direction():
    mod = domains.get("fmt")
    pairs = mod.build_pairs({"n_train": 4, "n_val": 1, "n_test": 1, "seed": 17,
                             "subtasks": ["json-xml"]})
    inst = pairs["train"][0].model_dump()
    r = mod.score_batch("reverse", [inst["side_a"]], [inst], {})[0]
    assert r["strict"] == 1


def test_fmt_lossy_ladder_drops_the_declared_share():
    import random
    from bidir.domains.fmt import drop_leaves

    doc = {f"k{i}": i for i in range(400)}
    kept = sum(1 for v in drop_leaves(doc, 0.5, random.Random(17)).values() if v is not None)
    assert 150 < kept < 250, "the ladder's information loss must track the declared share"
    assert all(v is None for v in drop_leaves(doc, 1.0, random.Random(17)).values())


def test_missing_threshold_raises_rather_than_defaulting():
    from bidir.domains._common import threshold_for
    with pytest.raises(KeyError, match="not frozen"):
        threshold_for({}, "forward", "comet")
