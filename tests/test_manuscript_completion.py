"""Versioned-domain reporting retains unfavorable seeds and rejects unverified runs."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manuscript_completion', ROOT/'scripts/118_manuscript_completion.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def snapshot(tmp_path):
    records = []; sources = {}
    for seed in [17, 42, 1234]:
        run = tmp_path/f'repair_domain_pilot_s{seed}'
        run.mkdir()
        rows = [dict(system=a, direction=d, pair_id=str(i), strategy='simple',
                     strict=int(not (a=='mix50' and seed==42)))
                for a in report.ARMS for d in ['forward','reverse'] for i in range(2)]
        (run/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (run/'summary.json').write_text(json.dumps(dict(arms=report.ARMS, n_instances=2,
            adapter_effectiveness={a:dict(identical_rate=.1) for a in report.ARMS if a!='base'})))
        records.append(dict(run=str(run), domain='py_cpp_typed_v2', model='llama31-8b', seed=seed))
        sources[str(run/'trials.jsonl')] = report.results.sha(run/'trials.jsonl')
    return dict(followup_reports=records, sources=sources)


def test_reporting_keeps_adverse_seed_and_its_same_pass_controls(tmp_path, monkeypatch):
    data = snapshot(tmp_path)
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'evaluation_succeeded', lambda *args:True)
    sources = {}
    cells = report.repaired_campaigns(data, sources)
    assert [c['seed'] for c in cells] == [17, 42, 1234]
    adverse = cells[1]['means']
    assert adverse['mix50']['reverse'] == 0
    assert adverse['replay']['reverse'] == adverse['base']['reverse'] == 1
    assert len(sources) == 6
    rendered = report.revised_table(cells)
    assert '42' in rendered and '0.0' in rendered


def test_missing_success_and_duplicate_cells_cannot_enter_manuscript(tmp_path, monkeypatch):
    data = snapshot(tmp_path)
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'evaluation_succeeded', lambda *args:False)
    with pytest.raises(ValueError, match='successful evaluation'):
        report.repaired_campaigns(data, {})
    monkeypatch.setattr(report, 'evaluation_succeeded', lambda *args:True)
    data['followup_reports'].append(data['followup_reports'][0])
    with pytest.raises(ValueError, match='duplicate revised-domain campaign'):
        report.repaired_campaigns(data, {})
