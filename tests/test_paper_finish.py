"""Guard against contaminated probes, incomplete pairing and unsafe automatic admission."""
import importlib.util
import json
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
synthesis=module('paper_synthesis_test','106_paper_synthesis.py')


def test_probe_table_keeps_adverse_seeds_and_separate_endpoints():
    comparisons=[('sft','base'),('mix5','sft'),('replay','sft')]
    results=[dict(summary=dict(mode='probes',eligibility_passed=True,domain='fmt',
        model='gemma3-4b',seed=seed),contrasts=dict(contrasts=[dict(a=a,b=b,
        endpoint=endpoint,delta_pp=delta if endpoint=='gsm8k' else 3)
        for endpoint in ['gsm8k','ifeval'] for a,b in comparisons]))
        for seed,delta in [(17,20),(42,-40)]]
    text=synthesis.publication_probe_table(results)
    assert '[-40.0, +20.0]' in text
    assert '[+3.0, +3.0]' in text
    assert 'gsm8k & 2' in text and 'ifeval & 2' in text
    with pytest.raises(ValueError,match='duplicate probe'):
        synthesis.publication_probe_table(results+[results[0]])


def test_overflow_requires_real_free_gpu_cpu_memory_and_healthy_node():
    node='NodeName=g-04-02 State=MIXED CPUEfctv=64 CPUAlloc=32 RealMemory=512000 AllocMem=299008 CfgTRES=cpu=64,gres/gpu=4 AllocTRES=cpu=32,gres/gpu=3'
    assert planner.spare_h100_slots(node)==1
    assert planner.spare_h100_slots(node.replace('AllocTRES=cpu=32,gres/gpu=3','AllocTRES=cpu=32,gres/gpu=4'))==0
    assert planner.spare_h100_slots(node.replace('State=MIXED','State=MIXED+DRAIN'))==0
    assert planner.spare_h100_slots(node.replace('CPUAlloc=32','CPUAlloc=60'))==0
    assert planner.spare_h100_slots(node.replace('AllocMem=299008','AllocMem=500000'))==0


@pytest.mark.parametrize('name',['tr_repair_pack','gr_gate_gemma3-4b'])
def test_overflow_admits_owned_ready_repair_jobs_only(tmp_path,monkeypatch,name):
    from types import SimpleNamespace
    monkeypatch.setattr(planner,'ROOT',tmp_path)
    (tmp_path/'runs/feeder').mkdir(parents=True)
    updates=[]
    node='State=MIXED CPUEfctv=64 CPUAlloc=32 RealMemory=512000 AllocMem=299008 CfgTRES=cpu=64,gres/gpu=4 AllocTRES=cpu=32,gres/gpu=3'
    def run(argv,**kwargs):
        if argv[0]=='squeue':
            return SimpleNamespace(stdout=(f'100|{name}|h100,h200|juno-pri|QOSMaxJobsPerUserLimit|(null)|{tmp_path}\n'
                f'101|gr_dependent|h100,h200|juno-pri|QOSMaxJobsPerUserLimit|afterok:1|{tmp_path}\n'
                '102|gr_foreign|h100,h200|juno-pri|QOSMaxJobsPerUserLimit|(null)|/another/project\n'))
        if argv[1:3]==['show','node']:
            return SimpleNamespace(stdout=node if argv[3]=='g-04-02' else node.replace('gres/gpu=3','gres/gpu=4'))
        if argv[1:3]==['show','job']:
            assert argv[3]=='100'
            return SimpleNamespace(stdout=f'JobState=PENDING WorkDir={tmp_path} JobName={name} Dependency=(null) Reason=QOSMaxJobsPerUserLimit ExcNodeList=(null)')
        updates.append(argv)
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(planner.subprocess,'run',run)
    planner.admit_h100_overflow()
    assert len(updates)==1
    assert 'JobId=100' in updates[0] and 'Partition=h100' in updates[0] and 'QOS=normal' in updates[0]


def test_repair_status_heartbeat_does_not_rebuild_evidence_bundle(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('artifact',ROOT/'scripts/107_paper_artifact.py')
    artifact=importlib.util.module_from_spec(spec);spec.loader.exec_module(artifact)
    status=tmp_path/'gate_repair_status.json'
    status.write_text(json.dumps(dict(updated_utc='first',items=[dict(state='submitted')])))
    monkeypatch.setattr(artifact,'ROOT',tmp_path)
    monkeypatch.setattr(artifact,'BUNDLE',tmp_path/'bundle')
    monkeypatch.setattr(artifact,'selected_paths',lambda:[status])
    artifact.build()
    manifest=artifact.BUNDLE/'MANIFEST.json';before=manifest.read_bytes()
    status.write_text(json.dumps(dict(updated_utc='second',items=[dict(state='submitted')])))
    artifact.build()
    assert manifest.read_bytes()==before
    status.write_text(json.dumps(dict(updated_utc='third',items=[dict(state='complete')])))
    artifact.build()
    assert manifest.read_bytes()!=before


def test_publication_reporting_uses_registered_probe_and_null_cell_panels():
    probe=dict(mode='probes',domain='units',model='gemma3-12b')
    robust=dict(probe,mode='robust')
    recipe=dict(probe,mode='recipe')
    assert synthesis.publication_cell(probe)['mix']=='mix5'
    assert synthesis.publication_cell(robust)['mix']=='mix50'
    assert synthesis.publication_cell(recipe)['mix']=='mix50'
    for mode,key in [('probes','probe'),('robust','robust'),('transfer','transfer'),('recipe','recipe')]:
        for cell in suite.plan()[key+'_panel']:
            assert synthesis.publication_cell(dict(cell,mode=mode))==cell


def test_publication_ranges_keep_same_pass_baselines_and_adverse_seed_outcomes():
    def campaign(seed, base, sft, mix):
        return dict(summary=dict(mode='robust', domain='fmt', model='gemma3-4b',
            seed=seed, eligibility_passed=True, means={
                'original/primary/forward': dict(base=base, sft=sft, mix5=mix),
                'original/primary/reverse': dict(base=base, sft=sft, mix5=mix)}))
    data=[campaign(17, .8, .1, .9), campaign(42, .2, .5, .4)]
    row=synthesis.publication_seed_ranges(data, 'robust')[0]
    assert row['seeds']==[17, 42]
    assert row['ranges'][0]==pytest.approx((-70, 30))
    assert row['ranges'][2]==pytest.approx((-10, 80))
    with pytest.raises(ValueError, match='duplicate'):
        synthesis.publication_seed_ranges(data+[data[0]], 'robust')
    # A failed eligibility gate cannot become a tuned comparison in the summary.
    data[1]['summary']['eligibility_passed']=False
    assert synthesis.publication_seed_ranges(data, 'robust')[0]['seeds']==[17]


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
