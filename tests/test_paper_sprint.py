"""Protect the changed scope against autonomous re-admission and deadline races."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest
from bidir import paper_sprint as sprint


@pytest.fixture
def policy(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    path = tmp_path / 'policy.json'
    config = dict(enabled=True, finish_by_utc=(now+timedelta(hours=24)).isoformat(),
        production_jobs=['ev_mx_hf_fmt_llama32-3b_s17'],
        allowed_jobs=['ev_mx_hf_fmt_llama32-3b_s17'])
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sprint, 'POLICY', path)
    return now, path, config


def test_scope_and_deadline(policy):
    now, _, _ = policy
    assert sprint.allowed('ev_mx_hf_fmt_llama32-3b_s17', now)
    assert not sprint.allowed('ev_mx_hf_fmt_llama32-3b_s42', now)
    assert not sprint.allowed('tr_fmt_contract_v2_gemma3-12b_s17_contrastive_pilot', now)
    assert not sprint.allowed('ev_mx_hf_fmt_llama32-3b_s17', now+timedelta(hours=24))


def test_deferred_submit_never_calls_slurm_or_writes_script(policy, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/slurm/submit.py'
    spec = importlib.util.spec_from_file_location('sprint_submit', script)
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    monkeypatch.setattr(submit, 'SLURM_DIR', tmp_path/'slurm')
    def forbidden(*args, **kwargs):
        pytest.fail('deferred jobs must never reach sbatch')
    monkeypatch.setattr(submit.subprocess, 'run', forbidden)
    assert submit.submit('unused', 'tr_fmt_llama32-3b_s42', False) is None
    assert not submit.SLURM_DIR.exists()


def test_disabled_policy_restores_original_admission(policy):
    now, path, config = policy
    config['enabled'] = False
    path.write_text(json.dumps(config))
    assert sprint.allowed('tr_fmt_llama32-3b_s42', now)


def test_malformed_policy_fails_closed(policy):
    _, path, config = policy
    config['finish_by_utc'] = 'not a date'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        sprint.allowed('ev_mx_hf_fmt_llama32-3b_s17')


def test_collector_keeps_historical_denominator_and_only_renders_proof_valid_results(policy, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/114_mechanism_analysis.py'
    spec = importlib.util.spec_from_file_location('sprint_collector', script)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    (tmp_path/'paper/tables').mkdir(parents=True)
    record = dict(name='ev_mx_hf_fmt_llama32-3b_s17', domain='fmt', model='llama32-3b',
        mode='hf', seed=17, state='awaiting proof-valid result')
    document = dict(production_total=24, items=[record])
    collector.sprint_tables(document)
    assert document['production_total'] == 24
    assert document['sprint']['production_total'] == 1
    assert document['sprint']['production_complete'] == 0
    table = tmp_path/'paper/tables/mechanism_sprint.tex'
    assert 'SFT minus base' not in table.read_text()
    result = dict(contrasts=[dict(system=system, reference=reference, direction=direction,
        metric='gold_delta_nll_per_token', mean=mean, ci95=[mean-.1, mean+.1])
        for direction in ['forward','reverse']
        for system,reference,mean in [('sft','base',.3),('mix5','sft',-.2)]],
        gradient_states=[dict(system=arm,cosine=-.3) for arm in ['sft','replay','mix5']])
    analysis = tmp_path/'analysis.json'; analysis.write_text(json.dumps(dict(result=result)))
    record.update(state='complete', analysis=str(analysis))
    collector.sprint_tables(document)
    content = table.read_text()
    assert '+0.300' in content and '-0.200' in content
    assert 'forward' in content and 'reverse' in content
    assert document['sprint']['production_complete'] == 1
