"""Protect table priority, phase dependencies, adverse outcomes and missing-cell accounting."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


queue=module('tc_queue_test','119_table_completion.py')
audit=module('tc_audit_test','121_table_audit.py')
results=module('tc_results_test','117_main_results.py')


def test_one_phase_per_cell_and_bounded_retries(monkeypatch):
    cell=queue.worker.plan()['cells'][0]
    monkeypatch.setattr(queue.worker,'ready_phase',lambda c:'eval')
    assert queue.next_action(cell,{queue.worker.job_name(cell,'train'):'1'}, {})=='inflight'
    name=queue.worker.job_name(cell,'eval')
    attempts={name:['1','2','3']}
    assert queue.next_action(cell,{},attempts).startswith('blocked: bounded attempts')


def test_no_production_train_without_smoke_and_no_partial_eval(monkeypatch):
    cell=queue.worker.plan()['cells'][1]
    monkeypatch.setattr(queue.worker,'adapter_valid',lambda c,a: a not in c['new_arms'])
    monkeypatch.setattr(queue.worker,'completed',lambda c,p:False)
    assert queue.worker.ready_phase(cell)=='smoke'
    monkeypatch.setattr(queue.worker,'completed',lambda c,p:p=='smoke')
    assert queue.worker.ready_phase(cell)=='train'
    monkeypatch.setattr(queue.worker,'adapter_valid',lambda c,a:True)
    assert queue.worker.ready_phase(cell)=='eval_smoke'
    monkeypatch.setattr(queue.worker,'completed',lambda c,p:p=='eval_smoke')
    assert queue.worker.ready_phase(cell)=='eval'


def test_missing_field_scan_is_not_confused_by_negative_values_or_headers(tmp_path):
    (tmp_path/'example.tex').write_text('Task & SFT--base & Arm \\\\\nA & -2.3 & -- \\\\\nB & 0.0 & \\textit{n/a} \\\\\n')
    rows=audit.missing_fields(tmp_path)
    assert len(rows)==1 and rows[0]['row']==2 and rows[0]['column']==3


def test_normalization_calculates_null_and_adverse_cells_without_recovery_claim():
    assert audit.normalized_change(.5,.6,.55)==pytest.approx(50)
    assert audit.normalized_change(.5,.4,.3)==pytest.approx(-100)
    assert audit.normalized_change(.5,.5,.6) is None


def test_newest_auxiliary_pass_is_selected_without_using_effect_sign(tmp_path):
    cells=[]
    for date,score in [('2026-10-01',1),('2026-10-02',0)]:
        run=tmp_path/date;run.mkdir()
        (run/'summary.json').write_text(json.dumps(dict(finished_utc=date)))
        cells.append(dict(domain='code',model='llama32-3b',seed=17,run=str(run),means={'cft':{'reverse':score}}))
    selected=results.latest_auxiliary(cells)
    assert len(selected)==1 and selected[0]['means']['cft']['reverse']==0


def test_controller_exclusive_mode_does_not_run_broad_planners(tmp_path,monkeypatch):
    daemon=module('tc_daemon_test','../runs/feeder/daemon.py')
    monkeypatch.setattr(daemon,'ROOT',tmp_path)
    (tmp_path/'configs').mkdir()
    path=tmp_path/'configs/table_completion.json'
    path.write_text(json.dumps(dict(enabled=True,exclusive=True)))
    exclusive,stages=daemon.controller_stages()
    assert exclusive and [s[0] for s in stages]==['table_completion','followup_analysis','paper_snapshot']
    path.write_text(json.dumps(dict(enabled=False,exclusive=True)))
    exclusive,stages=daemon.controller_stages()
    assert not exclusive and 'runner' in [s[0] for s in stages]
