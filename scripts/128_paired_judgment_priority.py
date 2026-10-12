#!/usr/bin/env python
"""Give PJ-5 exclusive project GPU priority, preserving other jobs for resumption."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
F=ROOT/'runs/feeder'
POLICY=ROOT/'configs/paired_judgment_priority.json'


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default


def names(plan):
    return {f"pj_{p}_{c['domain']}_{c['model']}_s{plan['seed']}"
            for c in plan['cells'] for p in ['smoke','train','eval_smoke','eval']}


def project_gpu(job):
    return (job.get('user_id')==os.getuid()
            and Path(job.get('current_working_directory','')).resolve()==ROOT
            and Path(job.get('command','')).parent==ROOT/'runs/slurm'
            and 'gres/gpu' in (job.get('tres_req_str','')+job.get('tres_per_node','')))


def identity(job):
    return {k:job.get(k) for k in ['job_id','name','user_id','current_working_directory','command']}


def operation(job, started):
    state=job.get('job_state',[])
    if 'PENDING' in state and job.get('state_reason') not in ['JobHeldUser','JobHeldAdmin']:
        return 'hold'
    if started and 'RUNNING' in state:
        if not job.get('requeue'):raise ValueError(f"Job {job['job_id']} cannot be requeued safely")
        return 'requeuehold'
    return None


def complete(status, plan):
    expected={(c['domain'],c['model']) for c in plan['cells']}
    actual={(r['cell']['domain'],r['cell']['model']) for r in status.get('rows',[])
            if r.get('phase')=='complete'}
    return (bool(expected) and status.get('total')==len(expected)
            and status.get('complete')==len(expected) and actual==expected)


def command(argv):
    return subprocess.run(argv,check=True,capture_output=True,text=True,timeout=30).stdout


def reconcile():
    if not read(POLICY).get('enabled'):return {'active':False}
    with (F/'.paired_judgment_priority.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'active':True,'busy':True}
        if not read(POLICY).get('enabled'):return {'active':False}
        plan=read(ROOT/'configs/paired_judgment.json');priority=names(plan)
        jobs=json.loads(command(['squeue','--json','-u',os.environ['USER']]))['jobs']
        ours={str(j['job_id']):j for j in jobs if project_gpu(j)}
        path=F/'paired_judgment_priority_journal.json'
        journal=read(path)
        if not journal:
            initial=read(F/'paired_judgment_priority_initial.json')
            journal={'created_utc':datetime.now(timezone.utc).isoformat(),
                     'started':bool(initial.get('priority_running')),'held':{},
                     'background_before':initial.get('background_before',read(ROOT/'configs/background_campaign.json'))}
            before={str(j['job_id']):j for j in initial.get('before',[]) if project_gpu(j)}
            for action in initial.get('actions',[]):
                jid=action['job_id']
                if action['state'] in ['requested','succeeded'] and jid in before:
                    journal['held'][jid]={'identity':identity(before[jid]),'operation':action['command'][1],
                                         'state':'held' if action['state']=='succeeded' else 'requested'}
            atomic_json(path,journal)
        running=[j['job_id'] for j in ours.values() if j['name'] in priority and 'RUNNING' in j['job_state']]
        journal['started']=journal['started'] or bool(running)
        done=complete(read(F/'paired_judgment_status.json'),plan)
        actions=[]
        if done:
            # Release only holds made by this policy, with the original job identity.
            for jid,record in journal['held'].items():
                job=ours.get(jid)
                if not job or record.get('state')=='released':continue
                if identity(job)!=record['identity']:raise ValueError('Held job identity changed: '+jid)
                if job.get('state_reason')=='JobHeldAdmin':raise ValueError('Administrator hold: '+jid)
                if job.get('state_reason')=='JobHeldUser':
                    command(['scontrol','release',jid]);actions.append(['release',jid])
                record['state']='released';atomic_json(path,journal)
            atomic_json(ROOT/'configs/background_campaign.json',journal['background_before'])
            atomic_json(POLICY,{**read(POLICY),'enabled':False,'completed_utc':datetime.now(timezone.utc).isoformat()})
        else:
            for jid,job in ours.items():
                if job['name'] in priority:continue
                op=operation(job,journal['started'])
                # Recover an interrupted command before inferring ownership of a hold.
                record=journal['held'].get(jid)
                if record and identity(job)!=record['identity']:raise ValueError('Job identity changed: '+jid)
                if op:
                    journal['held'][jid]={'identity':identity(job),'operation':op,'state':'requested'}
                    atomic_json(path,journal)
                    command(['scontrol',op,jid]);actions.append([op,jid])
                    journal['held'][jid]['state']='held'
                    atomic_json(path,journal)
                elif record and job.get('state_reason')=='JobHeldUser':record['state']='held'
        journal['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(path,journal)
        status={'updated_utc':journal['updated_utc'],'active':not done,'started':journal['started'],
                'priority_running':running,'recorded_holds':len(journal['held']),'actions':actions,'errors':[]}
        atomic_json(F/'paired_judgment_priority_status.json',status)
        return status


if __name__=='__main__':
    print(json.dumps(reconcile(),indent=2),flush=True)
