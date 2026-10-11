"""Validate the measured counts and paired uncertainty added to the manuscript."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task_presentation_test',ROOT/'scripts/122_task_presentation.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def trials():
    return [dict(system=a,direction=d,pair_id=str(i),cluster_id=str(i//2),strategy='simple',
                 strict=int(a!='sft'))
            for a in report.ARMS for d in ['forward','reverse'] for i in range(6)]


def test_bootstrap_preserves_constant_paired_difference():
    result=report.paired_intervals(trials(),reps=100)
    assert result['deltas']['mix1']['reverse']==dict(mean=1.,lo=1.,hi=1.)
    assert result['rates']['sft']['forward']==dict(mean=0.,lo=0.,hi=0.)


def test_bootstrap_rejects_partial_duplicate_or_inconsistent_clusters():
    rows=trials()
    with pytest.raises(ValueError,match='incomplete'):report.paired_intervals(rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):report.paired_intervals(rows+[rows[0]])
    rows[-1]['cluster_id']='different'
    with pytest.raises(ValueError,match='inconsistent'):report.paired_intervals(rows)


def test_dose_counts_remove_filtered_reverse_examples(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'adapter_dir',lambda *args:tmp_path/args[2])
    for a in ['sft','mix1','mix5']:
        p=tmp_path/a;p.mkdir()
        (p/'training_summary.json').write_text(json.dumps(dict(
            balance=dict(by_task=dict(rev=8)),lengths=dict(n_kept=99,dropped_by_task=dict(rev=2)))))
    sources={};counts=report.training_counts('code',sources)
    assert counts['train']==99 and counts['mix1']['reverse_pairs']==6
    assert len(sources)==3


def test_algebra_labels_follow_actual_forward_instruction():
    from bidir import domains
    pair=dict(side_a='A',side_b='B')
    assert domains.get('algebra').instruction('forward',pair).startswith('Expand')
    assert domains.get('algebra_rev').instruction('forward',pair).startswith('Factor')
    assert report.rates.TASKS['algebra']=='Polynomial expansion'
    assert report.rates.TASKS['algebra_rev']=='Polynomial factorization'


def test_diagnostic_subset_renders_after_scheduling_policy_disabled(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('diagnostic_render_test',ROOT/'scripts/114_mechanism_analysis.py')
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(dict(enabled=False,production_jobs=[])))
    monkeypatch.setattr(diagnostic.paper_sprint,'plan',lambda:None)
    monkeypatch.setattr(diagnostic.paper_sprint,'POLICY',policy)
    monkeypatch.setattr(diagnostic,'ROOT',tmp_path)
    (tmp_path/'paper/tables').mkdir(parents=True)
    diagnostic.sprint_tables(dict(items=[]))
    rendered=(tmp_path/'paper/tables/mechanism_sprint.tex').read_text()
    assert 'Likelihood, candidate selection' in rendered and 'Amendment54' not in rendered
    assert json.loads(policy.read_text())['enabled'] is False
