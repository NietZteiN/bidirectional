"""Guard against contaminated probes, incomplete pairing and unsafe automatic admission."""
import importlib.util
from pathlib import Path
import pytest
from bidir import paper_suite as suite

ROOT=Path(__file__).resolve().parents[1]


def module(name, script):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/script)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


worker=module('paper_worker_test','105_paper_worker.py')
planner=module('paper_planner_test','104_paper_finish.py')
evidence=module('paper_evidence_test','99_paper_evidence.py')
runner=module('paper_runner_test','95_runner.py')


def test_overlap_rejects_contained_questions_and_reports_source():
    question='What is the product of nine hundred and twenty three with forty seven?'
    index=suite.OverlapIndex([('replay/train', 'Question: '+question+' Answer: 43381')])
    assert index.hits(question)==['replay/train']
    assert index.hits('What is the product of eleven and thirteen?')==[]
    rows=[dict(pair_id='bad',question=question),dict(pair_id='good',question='Compute seventeen plus sixty five.')]
    kept,removed=suite.filter_rows(rows,index,['question'])
    assert [r['pair_id'] for r in kept]==['good']
    assert removed[0]['sources']==['replay/train']


def test_probe_endpoints_do_not_conflate_strict_and_flexible_gsm8k():
    doc={'answer':'Reasoning.\n#### 1,234'}
    assert worker.probe_score('gsm8k',doc,'Answer: 1234')['strict']==0
    assert worker.probe_score('gsm8k',doc,'Answer: 1234')['flexible']==1
    assert worker.probe_score('gsm8k',doc,'#### 1,234')['strict']==1
    assert worker.probe_score('gsm8k',doc,'#### 1235')['strict']==0


def rows():
    return [dict(endpoint=e,system=s,pair_id=f'p{i}',cluster_id=f'g{i//2}',strict=int(s=='mix5'))
            for e in ('forward','reverse') for s in ('base','sft','mix5') for i in range(6)]


def test_paired_coverage_rejects_missing_duplicate_and_cluster_mismatch():
    data=rows();systems={'base':None,'sft':'a','mix5':'b'}
    assert suite.validate_coverage(data,systems)=={'forward':6,'reverse':6}
    with pytest.raises(ValueError,match='incomplete'):suite.validate_coverage(data[:-1],systems)
    with pytest.raises(ValueError,match='duplicate'):suite.validate_coverage(data+[data[0]],systems)
    changed=[dict(r) for r in data];changed[0]['cluster_id']='unrelated'
    with pytest.raises(ValueError,match='cluster'):suite.validate_coverage(changed,systems)


def test_cluster_intervals_keep_training_seed_uncertainty_separate():
    data=rows();systems={'base':None,'sft':'a','mix5':'b'}
    result=suite.paired_intervals(data,systems,n_boot=100)
    comparison=next(r for r in result if r['a']=='mix5' and r['b']=='sft')
    assert comparison['n_pairs']==6 and comparison['n_clusters']==3
    assert comparison['delta_pp']==comparison['ci_lo']==comparison['ci_hi']==100
    assert 'equivalence' in comparison['multiplicity'] and 'verdict' not in comparison


def test_ood_gate_uses_all_unchanged_preconditions():
    gate={'forward':dict(rate=.1,format_fail=.15,echo_probe_strict=.02),
          'reverse':dict(rate=.5,format_fail=0,echo_probe_strict=0)}
    assert worker.transfer_gate_passed(gate)
    for field,bad in [('rate',.099),('format_fail',.151),('echo_probe_strict',.021)]:
        broken={key:dict(value) for key,value in gate.items()};broken['forward'][field]=bad
        assert not worker.transfer_gate_passed(broken)
    assert not worker.transfer_gate_passed({})


def test_frozen_manifest_detects_changed_test_or_training_source(tmp_path,monkeypatch):
    root=tmp_path/'root';data=root/'data';data.mkdir(parents=True)
    config=root/'plan.json';config.write_text('{}')
    original=root/'training.jsonl';original.write_text('training')
    held=data/'test.jsonl';held.write_text('held-out')
    suite.atomic_json(data/'manifest.json',dict(plan_sha256=suite.sha(config),
        files={'test.jsonl':suite.sha(held)},training_source_sha256={'training.jsonl':suite.sha(original)}))
    monkeypatch.setattr(suite,'DATA',data);monkeypatch.setattr(suite,'ROOT',root);monkeypatch.setattr(suite,'PLAN',config)
    suite.frozen_data();held.write_text('changed')
    with pytest.raises(ValueError,match='frozen paper corpus'):suite.frozen_data()
    held.write_text('held-out');original.write_text('changed')
    with pytest.raises(ValueError,match='local corpus'):suite.frozen_data()


def test_recipe_namespace_never_overwrites_original_adapters():
    cell={'domain':'units','model':'gemma3-12b','mix':'mix50'}
    path=suite.recipe_dir(cell,42,'lr_high','sft')
    assert 'paper_finish_v1' in path.parts and 'lr_high__sft'==path.name
    assert planner.name('train',cell,42)!=planner.name('recipe',cell,42)
    other=dict(cell,model='llama32-3b')
    assert suite.recipe_dir(cell,17,'rank_high','sft',True)!=suite.recipe_dir(other,17,'rank_high','sft',True)


def test_inventory_deduplicates_reruns_without_selecting_favorable_effect():
    old=dict(domain='units',model='gemma3-12b',seed=17,run='older',finished_utc='2026-10-07',collapse=True)
    new=dict(old,run='newer',finished_utc='2026-10-08',collapse=False)
    second=dict(new,seed=42,run='seed42')
    latest,superseded=evidence.latest_core_campaigns([new,old,second])
    assert len(latest)==2 and new in latest and old in superseded
    assert not latest[0]['collapse']


def test_completed_evaluation_cannot_promote_partial_trials(tmp_path,monkeypatch):
    cell=dict(domain='fmt',model='gemma3-4b',mix='mix5')
    out=tmp_path/'out';out.mkdir()
    root=tmp_path/'root';(root/'runs/status').mkdir(parents=True)
    config=tmp_path/'plan.json';config.write_text('{}')
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}')
    monkeypatch.setattr(planner,'ROOT',root)
    monkeypatch.setattr(suite,'PLAN',config);monkeypatch.setattr(suite,'DATA',data)
    monkeypatch.setattr(planner,'worker_code_hash',lambda:'worker')
    monkeypatch.setattr(suite,'result_dir',lambda *args,**kwargs:out)
    monkeypatch.setattr(suite,'systems_for',lambda *args,**kwargs:{'base':None,'sft':'adapter'})
    rows=[dict(endpoint='ifeval',system=s,pair_id=f'p{i}',cluster_id=f'p{i}',strict=1)
          for s in ('base','sft') for i in range(2)]
    suite.write_rows(out/'trials.jsonl',rows)
    summary=dict(plan_sha256=suite.sha(config),worker_sha256='worker',engineering_smoke=True,
        mode='probes',domain='fmt',model='gemma3-4b',seed=17,eligibility_passed=True,
        data_manifest_sha256=suite.sha(data/'manifest.json'),trial_sha256=suite.sha(out/'trials.jsonl'),
        systems={'base':None,'sft':'adapter'},n_instances_by_endpoint={'ifeval':2},
        adapter_effectiveness={'sft':dict(identical_rate=.5)},finished_utc='2026-10-08T00:00:00+00:00')
    suite.atomic_json(out/'summary.json',summary)
    name=planner.name('probes',cell,17,True)
    suite.atomic_json(root/f'runs/status/{name}.1.json',dict(exit_code=0,finished_utc='2026-10-08T00:01:00+00:00'))
    assert planner.completed('probes',cell,17,True)
    suite.write_rows(out/'trials.jsonl',rows[:-1]);summary['trial_sha256']=suite.sha(out/'trials.jsonl')
    suite.atomic_json(out/'summary.json',summary)
    assert not planner.completed('probes',cell,17,True)


def test_new_paper_jobs_are_counted_in_runner_capacity():
    assert runner.OURS.match('ev_paper_probes_fmt_gemma3-4b_s17_smoke')
    assert runner.OURS.match('tr_paper_train_units_gemma3-12b_s42')
    assert runner.OURS.match('tr_fmt_llama31-8b_s17_contrastive_pilot')
    assert not runner.OURS.match('tr_unrelated_project')


def test_wider_smoke_placement_keeps_unproven_production_and_bad_nodes_excluded():
    small=dict(domain='fmt',model='gemma3-4b',mix='mix5')
    parts,excluded=planner.placement('robust',small,True)
    assert set(parts.split(','))=={'h100','h200','a30'}
    assert {'g-01-01','g-02-01','g-03-01','g-06-01','g-08-06'}<=set(excluded.split(','))
    assert 'a30' not in planner.placement('robust',small,False)[0].split(',')
    mt=dict(domain='mt_de-en',model='llama32-3b',mix='mix5')
    assert 'a30' not in planner.placement('transfer',mt,True)[0].split(',')
    large=dict(domain='units',model='gemma3-12b',mix='mix50')
    assert planner.placement('train_smoke',large,True)[0]=='h100,h200'
    assert planner.placement('train',large,False)[0]=='h200'


def test_failed_transfer_boundary_requires_frozen_data_and_trial_hashes(tmp_path,monkeypatch):
    cell=dict(domain='mt_de-en',model='llama32-3b',mix='mix5')
    out=tmp_path/'out';out.mkdir()
    root=tmp_path/'root';(root/'runs/status').mkdir(parents=True)
    config=tmp_path/'plan.json';config.write_text('{}')
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}')
    monkeypatch.setattr(planner,'ROOT',root)
    monkeypatch.setattr(suite,'PLAN',config);monkeypatch.setattr(suite,'DATA',data)
    monkeypatch.setattr(planner,'worker_code_hash',lambda:'worker')
    monkeypatch.setattr(suite,'result_dir',lambda *args,**kwargs:out)
    trial=out/'trials.jsonl'
    suite.write_rows(trial,[dict(endpoint='opus/primary/forward',system='base',pair_id='p0',cluster_id='g0',strict=0)])
    summary=dict(mode='transfer',domain=cell['domain'],model=cell['model'],seed=17,
        plan_sha256=suite.sha(config),worker_sha256='worker',engineering_smoke=False,
        eligibility_passed=False,gate={'forward':dict(rate=0,format_fail=0,echo_probe_strict=0)},
        data_manifest_sha256=suite.sha(data/'manifest.json'),trial_sha256=suite.sha(trial),
        systems={'base':None},n_instances_by_endpoint={'opus/primary/forward':1},
        finished_utc='2026-10-08T00:00:00+00:00')
    suite.atomic_json(out/'summary.json',summary)
    name=planner.name('transfer',cell,17)
    suite.atomic_json(root/f'runs/status/{name}.1.json',dict(exit_code=0,finished_utc='2026-10-08T00:01:00+00:00'))
    assert planner.completed('transfer',cell,17)
    trial.write_text('')
    assert not planner.completed('transfer',cell,17)
    summary['trial_sha256']=suite.sha(trial);suite.atomic_json(out/'summary.json',summary)
    assert not planner.completed('transfer',cell,17)
