import pytest
from bidir.domains import py_cpp as domain


def test_program_oracles_and_semantic_splits():
    splits=domain.build_pairs(dict(n_train=12,n_val=6,n_test=6,seed=17))
    keys=[domain.content_key(p) for rows in splits.values() for p in rows]
    assert len(keys)==len(set(keys))
    assert len({p.meta['template_family'] for p in splits['train']})==1
    for rows in splits.values():
        insts=[p.model_dump() for p in rows[:3]]
        for direction,side,source in [('forward','side_b','side_a'),('reverse','side_a','side_b')]:
            gold=domain.score_batch(direction,[p[side] for p in insts],insts,{})
            assert all(r['strict'] and r['execution_coverage']==33 for r in gold)
            for outputs in [[p[source] for p in insts],['']*len(insts),['garbage']*len(insts)]:
                assert not any(r['strict'] for r in domain.score_batch(direction,outputs,insts,{}))


def test_alternate_program_and_parser_bounds():
    py=domain.python_node('def f(x):\n return x*x + 2*x + 1')
    cpp,_=domain.cpp_node('int f(int x) { return (x+1)*(x+1); }')
    assert domain.behavior(py)==domain.behavior(cpp)
    for source in ['def f(x):\n return __import__("os")', 'def f(x):\n return x**2', 'def f(x):\n return 999*x*x*x*x*x*x']:
        with pytest.raises(ValueError): domain.python_node(source)
    for source in ['int f(int x) { return system("ls"); }','int f(int x) { return x; } int g(int y){return y;}']:
        with pytest.raises(ValueError): domain.cpp_node(source)
    node,_=domain.cpp_node('long long f(long long x) {return x<0 ? -x : x;}')
    assert domain.behavior(node)==tuple(abs(x) for x in domain.INPUTS)
