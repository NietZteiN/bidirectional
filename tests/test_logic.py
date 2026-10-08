import pytest
from bidir.domains import logic


def test_logic_oracle_and_semantic_splits():
    splits = logic.build_pairs(dict(n_train=120,n_val=30,n_test=30,seed=17))
    keys = [logic.content_key(p) for rows in splits.values() for p in rows]
    assert len(keys) == len(set(keys))
    for rows in splits.values():
        insts = [p.__dict__ for p in rows]
        for direction, side in [('forward','side_b'),('reverse','side_a')]:
            assert all(r['strict'] for r in logic.score_batch(direction,
                       [i[side] for i in insts],insts,{}))
            source = 'side_a' if direction == 'forward' else 'side_b'
            for outputs in [[i[source] for i in insts], ['']*len(insts), ['garbage']*len(insts)]:
                assert not any(r['strict'] for r in logic.score_batch(direction,outputs,insts,{}))


def test_logic_equivalence_and_rejection():
    assert logic.truth_bits('a and b') == logic.truth_bits('not (not a or not b)')
    assert logic.truth_bits('True') == '1'*16
    for bad in ['__import__("os")', 'a + b', 'x', '1', 'a == b', 'a.b']:
        with pytest.raises(ValueError):
            logic.truth_bits(bad)
    p = logic.build_pairs(dict(n_train=1,n_val=0,n_test=0,seed=17))['train'][0].__dict__
    # A valid formula for the complement must never pass semantic equivalence.
    assert logic.score_batch('reverse', ['not ('+p['side_a']+')'], [p], {})[0]['strict'] == 0


def test_parser_stack_exhaustion_is_off_target(monkeypatch):
    p = logic.build_pairs(dict(n_train=1,n_val=0,n_test=0,seed=17))['train'][0].__dict__
    def exhausted(*args, **kwargs):
        raise MemoryError('Parser stack overflowed - Python source too complex to parse')
    monkeypatch.setattr(logic.ast, 'parse', exhausted)
    row = logic.score_batch('reverse', ['not a'], [p], {})[0]
    assert row['strict'] == row['semantic_match'] == 0
    assert row['off_target'] == 1


def test_actual_memory_failure_remains_measurement_failure(monkeypatch):
    p = logic.build_pairs(dict(n_train=1,n_val=0,n_test=0,seed=17))['train'][0].__dict__
    def exhausted(*args, **kwargs):
        raise MemoryError('out of memory')
    monkeypatch.setattr(logic.ast, 'parse', exhausted)
    with pytest.raises(MemoryError, match='out of memory'):
        logic.score_batch('reverse', ['not a'], [p], {})
