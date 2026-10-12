"""Project-scoped priority, start-triggered preemption, and recorded-hold resumption."""
import importlib.util
import json
import os
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pj_priority_tests',ROOT/'scripts/128_paired_judgment_priority.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def job(jid,name,state='PENDING',reason='Priority',root=ROOT):
    return dict(job_id=jid,name=name,job_state=[state],state_reason=reason,user_id=os.getuid(),
                current_working_directory=str(root),command=str(root/'runs/slurm'/(name+'.sbatch')),
                tres_req_str='cpu=8,gres/gpu=1',tres_per_node='gres/gpu:1',requeue=True)


def test_only_owned_project_gpu_jobs_are_eligible():
    assert guard.project_gpu(job(1,'tr_x'))
    assert not guard.project_gpu({**job(1,'tr_x'),'user_id':os.getuid()+1})
    assert not guard.project_gpu(job(1,'tr_x',root=ROOT.parent/'obtune'))
    assert not guard.project_gpu({**job(1,'hold-node'),'tres_req_str':'cpu=8','tres_per_node':''})
    assert not guard.project_gpu({**job(1,'tr_x'),'command':str(ROOT.parent/'node/hold.sbatch')})


def test_start_trigger_and_preexisting_holds():
    assert guard.operation(job(1,'tr_x'),False)=='hold'
    assert guard.operation(job(1,'tr_x','RUNNING'),False) is None
    assert guard.operation(job(1,'tr_x','RUNNING'),True)=='requeuehold'
    for reason in ['JobHeldUser','JobHeldAdmin']:
        assert guard.operation(job(1,'tr_x',reason=reason),True) is None
    with pytest.raises(ValueError,match='cannot be requeued'):
        guard.operation({**job(1,'tr_x','RUNNING'),'requeue':False},True)


def test_priority_stage_excludes_all_background_submitters(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('pj_priority_daemon',ROOT/'runs/feeder/daemon.py')
    daemon=importlib.util.module_from_spec(spec);spec.loader.exec_module(daemon)
    monkeypatch.setattr(daemon,'ROOT',tmp_path)
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/paired_judgment_priority.json').write_text('{"enabled":true}')
    exclusive,stages=daemon.controller_stages()
    assert exclusive
    assert [s[0] for s in stages]==['paired_priority','paired_judgment','followup_analysis','paper_snapshot']


def test_lifecycle_pauses_on_start_keeps_phase_gaps_exclusive_and_releases_only_our_holds(tmp_path,monkeypatch):
    feeder=tmp_path/'runs/feeder';feeder.mkdir(parents=True)
    configs=tmp_path/'configs';configs.mkdir()
    policy=configs/'paired_judgment_priority.json'
    monkeypatch.setattr(guard,'ROOT',tmp_path);monkeypatch.setattr(guard,'F',feeder);monkeypatch.setattr(guard,'POLICY',policy)
    plan=dict(seed=17,cells=[dict(domain=d,model=m) for d in ['mt_en-de','sql'] for m in ['3b','8b']])
    (configs/'paired_judgment.json').write_text(json.dumps(plan));policy.write_text('{"enabled":true}')
    (configs/'background_campaign.json').write_text('{"enabled":true}')
    pj=job(1,'pj_smoke_mt_en-de_3b_s17',root=tmp_path)
    other=job(2,'tr_other','RUNNING',root=tmp_path)
    pending=job(3,'ev_other',root=tmp_path)
    preheld=job(4,'tr_preheld',reason='JobHeldUser',root=tmp_path)
    unrelated=job(5,'tr_external',root=tmp_path.parent/'other')
    jobs=[pj,other,pending,preheld,unrelated];calls=[]
    def command(argv):
        if argv[0]=='squeue':return json.dumps(dict(jobs=jobs))
        calls.append(argv)
        j=next(j for j in jobs if str(j['job_id'])==argv[2])
        j['job_state']=['PENDING'];j['state_reason']='Priority' if argv[1]=='release' else 'JobHeldUser'
        return ''
    monkeypatch.setattr(guard,'command',command)
    assert not guard.reconcile()['started']
    assert calls==[['scontrol','hold','3']]
    pj['job_state']=['RUNNING']
    assert guard.reconcile()['started']
    assert ['scontrol','requeuehold','2'] in calls
    jobs.remove(pj)  # Between smoke and production: remain exclusive.
    assert guard.reconcile()['started'] and json.loads(policy.read_text())['enabled']
    status=dict(total=4,complete=4,rows=[dict(cell=c,phase='complete') for c in plan['cells']])
    (feeder/'paired_judgment_status.json').write_text(json.dumps(status))
    assert not guard.reconcile()['active']
    assert sorted(c[2] for c in calls if c[1]=='release')==['2','3']
    assert preheld['state_reason']=='JobHeldUser' and unrelated['state_reason']=='Priority'
    assert not json.loads(policy.read_text())['enabled']
    assert json.loads((configs/'background_campaign.json').read_text())['enabled']


def test_completed_counter_alone_cannot_release_work():
    plan=dict(cells=[dict(domain='sql',model='3b'),dict(domain='sql',model='8b')])
    assert not guard.complete(dict(total=2,complete=2,rows=[]),plan)
    rows=[dict(cell=c,phase='complete') for c in plan['cells']]
    assert guard.complete(dict(total=2,complete=2,rows=rows),plan)
    rows[1]['phase']='blocked'
    assert not guard.complete(dict(total=2,complete=2,rows=rows),plan)
