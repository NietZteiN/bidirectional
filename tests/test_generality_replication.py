import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('generality_runner', Path(__file__).resolve().parents[1] / 'scripts/95_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def valid_null_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(runner, 'adapter_complete', lambda *args: True)
    monkeypatch.setattr('bidir.pipeline_state.evaluation_succeeded', lambda *args: True)
    run = tmp_path / '2026-10-08/units/gemma3-12b/generality_scale_pilot_s17'
    run.mkdir(parents=True)
    trial = run / 'trials.jsonl'
    trial.write_text('valid coverage audited by followup analysis\n')
    means = {arm: {'forward': .5, 'reverse': .5} for arm in ['base', 'sft', 'rev', 'mix50', 'replay']}
    means['sft'] = {'forward': .97, 'reverse': .7}  # valid improvement; no collapse
    report = dict(domain='units', model='gemma3-12b', seed=17, means=means,
                  trial_sha256=hashlib.sha256(trial.read_bytes()).hexdigest(), evaluation_layout='single_engine_pass')
    (run / 'followup_contrasts.json').write_text(json.dumps(report))
    (run / 'summary.json').write_text(json.dumps(dict(arms=list(means), n_instances=1000,
        adapter_effectiveness={arm: {'identical_rate': .2} for arm in means if arm != 'base'})))
    return run


def test_valid_null_and_adverse_pilots_are_promoted(valid_null_pilot):
    assert runner.generality_replication_valid('units', 'gemma3-12b')
    path = valid_null_pilot / 'followup_contrasts.json'
    report = json.loads(path.read_text())
    report['means']['sft']['reverse'] = 0
    path.write_text(json.dumps(report))
    assert runner.generality_replication_valid('units', 'gemma3-12b')


def test_partial_stale_withdrawn_and_ineffective_pilots_are_blocked(valid_null_pilot, monkeypatch):
    monkeypatch.setattr('bidir.pipeline_state.evaluation_succeeded', lambda *args: False)
    assert not runner.generality_replication_valid('units', 'gemma3-12b')

    monkeypatch.setattr('bidir.pipeline_state.evaluation_succeeded', lambda *args: True)
    marker = valid_null_pilot / 'WITHDRAWN.txt'
    marker.write_text('withdrawn')
    assert not runner.generality_replication_valid('units', 'gemma3-12b')
    marker.unlink()
    trial = valid_null_pilot / 'trials.jsonl'
    original = trial.read_text()
    trial.write_text('changed coverage')
    assert not runner.generality_replication_valid('units', 'gemma3-12b')
    trial.write_text(original)
    path = valid_null_pilot / 'summary.json'
    summary = json.loads(path.read_text())
    summary['adapter_effectiveness']['sft']['identical_rate'] = 1
    path.write_text(json.dumps(summary))
    assert not runner.generality_replication_valid('units', 'gemma3-12b')


def test_scale_admission_requires_own_objective_and_control_and_accepts_valid_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'RESULTS_DIR', tmp_path / 'results')
    statuses = tmp_path / 'runs/status'
    statuses.mkdir(parents=True)
    def write_proof(job, name, model, arm, rc=0):
        (statuses / f'{name}.{job}.json').write_text(json.dumps(dict(exit_code=rc)))
        out = runner.RESULTS_DIR / 'engineering_smokes' / f'{model}_{arm}_{job}'
        out.mkdir(parents=True)
        (out / 'profile.json').write_text(json.dumps(dict(exit_code=0)))
        (out / 'training_summary.json').write_text(json.dumps(dict(steps=2,
            direction_exposure=dict(contrastive_gradient_gate_passed=True, extra_ce_passes=6))))
    for job, name, model, arm in [
        (440645,'cl_accum_smoke_llama','llama32-3b','cl_fwd'),
        (440646,'ce_proxy_smoke_llama','llama32-3b','sft_extra_ce'),
        (440647,'cl_mix_smoke_gemma','gemma3-4b','cl_mix5'),
        (440648,'cl_shuffle_smoke_gemma','gemma3-4b','cl_shuffled')]:
        write_proof(job,name,model,arm)
    assert runner.contrastive_smokes_ready('llama32-3b')
    assert not runner.contrastive_smokes_ready('gemma3-12b')
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'contrastive_scale_smokes.json').write_text(json.dumps({'gemma3-12b':[
        dict(job=100,name='cl_large',arm='cl_fwd'), dict(job=101,name='ce_large',arm='sft_extra_ce')]}))
    write_proof(100,'cl_large','gemma3-12b','cl_fwd')
    assert not runner.contrastive_smokes_ready('gemma3-12b')
    write_proof(101,'ce_large','gemma3-12b','sft_extra_ce',rc=1)
    assert not runner.contrastive_smokes_ready('gemma3-12b')
    write_proof(102,'ce_large','gemma3-12b','sft_extra_ce')
    assert runner.contrastive_smokes_ready('gemma3-12b')
