"""Repaired scoring must reject unsafe programs, echoes and incomplete gate proofs."""
import ast
import importlib.util
import json
from pathlib import Path
import pytest

from bidir import gate_repair as repair
from bidir.domains import py_cpp, py_cpp_typed_v2 as typed, mt_noecho_v2, mt


def test_signed_literal_repair_accepts_real_behavior_and_preserves_original_scorer():
    source='long long f(int x) { return -2LL * x * x * x + 10LL * x + -14LL; }'
    with pytest.raises(ValueError):py_cpp.cpp_node(source)
    node,_=typed.cpp_node(source)
    values=py_cpp.behavior(ast.parse('-2*x*x*x+10*x-14',mode='eval').body)
    assert typed.compiled_behavior(source)==values==py_cpp.behavior(node)
    inst=dict(side_a='def f(x):\n    return -2*x*x*x+10*x-14',side_b=source,meta=dict(behavior=list(values)))
    assert typed.score_batch('forward',[source],[inst],{})[0]['strict']==1
    wrong=source.replace('10LL','11LL')
    assert typed.score_batch('forward',[wrong],[inst],{})[0]['strict']==0


@pytest.mark.parametrize('expression',['system(1LL)','1ULL+x','1LLx','1LL.0+x',
                                        '1001LL+x','1000LL*1000LL*1000LL*x'])
def test_literal_repair_keeps_bounded_allowlist(expression):
    with pytest.raises(ValueError):typed.cpp_node('long long f(long long x) { return '+expression+'; }')


@pytest.mark.parametrize('direction',['forward','reverse'])
def test_stricter_translation_rejects_echo_in_both_directions(monkeypatch,direction):
    monkeypatch.setattr(mt,'score_batch',lambda *args:[dict(strict=1,echo=1,criterion='original'),
                                                     dict(strict=1,echo=0,criterion='original')])
    rows=mt_noecho_v2.score_batch(direction,[],[],{})
    assert [row['strict'] for row in rows]==[0,1]


def test_repair_oracle_requires_complete_exact_gold_and_negative_controls():
    proof=dict(directions={d:{name:dict(strict=rate) for name,rate in
                             [('gold',1),('echo',0),('empty',0),('garbage',0)]}
                           for d in ['forward','reverse']})
    assert not repair.oracle_problems(proof)
    proof['directions']['forward']['gold']['strict']=.975
    assert repair.oracle_problems(proof)
    del proof['directions']['reverse']
    assert any('reverse' in problem for problem in repair.oracle_problems(proof))


def test_gate_admission_rejects_unsuccessful_job_and_tampered_trials(tmp_path,monkeypatch):
    monkeypatch.setattr(repair,'OUT',tmp_path)
    monkeypatch.setattr(repair,'code_hash',lambda:'version')
    monkeypatch.setattr(repair,'data_hash',lambda cell:'data')
    name='py_cpp_typed_v2';out=tmp_path/'gates'/'model';out.mkdir(parents=True)
    (out/'trials.jsonl').write_text('{}\n');(out/'oracle.json').write_text('{}')
    summary=dict(worker_sha256='version',data_sha256={name:'data'},finished_utc='2026-10-09',
                 trial_sha256=repair.sha(out/'trials.jsonl'),oracle_sha256=repair.sha(out/'oracle.json'),
                 gates={name:dict(passes_gate=True)})
    (out/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(repair,'status_success',lambda *args:False)
    assert repair.gate_passed(name,'model') is None
    monkeypatch.setattr(repair,'status_success',lambda *args:True)
    assert repair.gate_passed(name,'model') is True
    (out/'trials.jsonl').write_text('changed\n')
    assert repair.gate_passed(name,'model') is None


def test_d2t_schema_comes_only_from_training_relations():
    from bidir.config import DATA_DIR
    from bidir.domains import d2t
    from bidir.schema import read_pairs
    if not (DATA_DIR/'d2t_schema_v2/extractor_schema.json').exists():
        pytest.skip('requires prepared WebNLG repair data')
    doc=json.loads((DATA_DIR/'d2t_schema_v2/extractor_schema.json').read_text())
    expected={p for row in read_pairs(DATA_DIR/'d2t/train.jsonl') for _,p,_ in d2t.parse_triples(row.side_a)}
    assert doc['relations']==sorted(expected)
    assert doc['uses_test_targets'] is False


def test_repaired_contrastive_rejects_mismatched_negative_pool(monkeypatch):
    spec=importlib.util.spec_from_file_location('repair_trainer',
        Path(__file__).resolve().parents[1]/'scripts/110_gate_repair_train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    from types import SimpleNamespace
    def rows(path):
        return [SimpleNamespace(side_a=str(path.parent.name),side_b='b',subtask='json-yaml')]
    monkeypatch.setattr(trainer,'read_pairs',rows)
    with pytest.raises(ValueError,match='does not match'):trainer.install_candidates()


def test_repaired_contrastive_smoke_requires_success_and_signed_training(tmp_path,monkeypatch):
    monkeypatch.setattr(repair,'OUT',tmp_path)
    monkeypatch.setattr(repair,'DATA_DIR',tmp_path/'data')
    monkeypatch.setattr(repair,'code_hash',lambda:'current')
    monkeypatch.setattr(repair,'data_hash',lambda cell:'data')
    pool=tmp_path/'data/fmt';pool.mkdir(parents=True);(pool/'train.jsonl').write_text('{}')
    out=tmp_path/'contrastive_smokes/gemma3-12b/cl_fwd';out.mkdir(parents=True)
    training=dict(steps=2,direction_exposure=dict(contrastive_gradient_gate_passed=True))
    (out/'training_summary.json').write_text(json.dumps(training))
    summary=dict(worker_sha256='current',data_sha256='data',exit_code=0,steps=2,
        negative_pool_sha256=repair.sha(pool/'train.jsonl'),
        training_sha256=repair.sha(out/'training_summary.json'),peak_cuda_bytes=1,
        finished_utc='2026-10-09')
    (out/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(repair,'status_success',lambda *args:False)
    assert not repair.contrastive_smoke_passed('gemma3-12b','cl_fwd')
    monkeypatch.setattr(repair,'status_success',lambda *args:True)
    assert repair.contrastive_smoke_passed('gemma3-12b','cl_fwd')
    (out/'training_summary.json').write_text('{}')
    assert not repair.contrastive_smoke_passed('gemma3-12b','cl_fwd')
