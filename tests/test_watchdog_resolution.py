import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('watchdog',
    Path(__file__).resolve().parents[1] / 'runs/feeder/watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


@pytest.mark.parametrize('name',['ev_fmt_gemma3-4b_s17_contrastive_pilot','gr_gate_gemma3-12b'])
def test_later_success_clears_alarm_and_preserves_failure_evidence(tmp_path, monkeypatch, name):
    monkeypatch.setattr(watchdog, 'ROOT', tmp_path)
    monkeypatch.setattr(watchdog, 'QUAR', tmp_path / 'quarantine')
    monkeypatch.setattr(watchdog, 'FAILURES', tmp_path / 'failures')
    monkeypatch.setattr(watchdog, 'EVENTS', tmp_path / 'events')
    status_dir = tmp_path / 'runs/status'
    status_dir.mkdir(parents=True)
    failure = dict(id='100', name=name, why='quarantined')
    state = {'code_failures': [failure]}
    jobs = [('101', name, 'COMPLETED')]
    watchdog.resolve_runner_failures(state, jobs)
    assert 'resolved_by' not in failure
    (status_dir / f'{name}.101.json').write_text(json.dumps(dict(exit_code=0, job_id='101')))
    watchdog.QUAR.write_text(name.split('_', 1)[1] + '\n')
    watchdog.resolve_runner_failures(state, jobs)
    assert 'resolved_by' not in failure
    watchdog.QUAR.write_text('')
    watchdog.resolve_runner_failures(state, [('99', name, 'COMPLETED')])
    assert 'resolved_by' not in failure
    watchdog.resolve_runner_failures(state, jobs)
    assert failure['resolved_by']['job_id'] == '101'
    assert state['code_failures'][0]['why'] == 'quarantined'
    assert 'Failure evidence retained' in watchdog.FAILURES.read_text()
