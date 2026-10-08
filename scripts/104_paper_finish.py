#!/usr/bin/env python
"""Idempotently admit frozen paper-completion smokes and dependent production packs."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import paper_suite as suite
from bidir.pipeline_state import atomic_json,evaluation_succeeded


def name(mode,cell,seed,smoke=False):
    prefix='tr' if mode in ('train','train_smoke') else 'ev'
    return f"{prefix}_paper_{mode}_{cell['domain']}_{cell['model']}_s{seed}"+('_smoke' if smoke else '')


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def worker_code_hash():
    spec=importlib.util.spec_from_file_location('paper_worker',ROOT/'scripts/105_paper_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    return worker.code_hash()


def completed(mode,cell,seed,smoke=False):
    out=suite.result_dir('train' if mode in ('train','train_smoke') else mode,cell,seed,smoke)
    summary=read(out/'summary.json')
    if (summary.get('plan_sha256')!=suite.sha(suite.PLAN) or
        summary.get('worker_sha256')!=worker_code_hash() or
        summary.get('engineering_smoke')!=smoke):return False
    job_name=name(mode,cell,seed,smoke)
    if mode in ('train','train_smoke'):
        statuses=[read(p) for p in (ROOT/'runs/status').glob(job_name+'.*.json')]
        success=any(d.get('exit_code')==0 and d.get('finished_utc','')>=summary.get('finished_utc','') for d in statuses)
        return success and not summary.get('failed')
    if not evaluation_succeeded(out,ROOT/'runs/status',job_name):return False
    if summary.get('data_manifest_sha256')!=suite.sha(suite.DATA/'manifest.json'):return False
    trial=out/'trials.jsonl'
    if not trial.exists() or summary.get('trial_sha256')!=suite.sha(trial):return False
    if (summary.get('mode')!=mode or summary.get('domain')!=cell['domain'] or
        summary.get('model')!=cell['model'] or summary.get('seed')!=seed):return False
    if summary.get('eligibility_passed') is False:
        if smoke or mode!='transfer' or not summary.get('gate'):return False
        try:
            return (summary.get('systems')=={'base':None} and
                    suite.validate_coverage(suite.read_rows(trial),{'base':None})==summary.get('n_instances_by_endpoint'))
        except (OSError,KeyError,ValueError):return False
    try:
        systems=suite.systems_for(mode,cell,seed,smoke)
        if summary.get('systems')!=systems:return False
        coverage=suite.validate_coverage(suite.read_rows(trial),systems)
        if coverage!=summary.get('n_instances_by_endpoint'):return False
    except (OSError,KeyError,ValueError):return False
    effects=summary.get('adapter_effectiveness',{})
    return set(effects)==set(systems)-{'base'} and all(0<=v.get('identical_rate',1)<.999 for v in effects.values())


def submit(mode,cell,seed,smoke,dep=None,dry=False):
    job_name=name(mode,cell,seed,smoke)
    duration='00:30:00' if smoke else '06:00:00'
    partition,excluded=placement(mode,cell,smoke)
    argv=['scripts/105_paper_worker.py','--mode',mode,'--domain',cell['domain'],
          '--model',cell['model'],'--seed',str(seed)]
    if smoke and mode!='train_smoke':argv.append('--smoke')
    cmd=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',job_name,'--partition',partition,
         '--exclude',excluded,'--time',duration,'--mem','64G','--cpus','8']
    if dep:cmd+=['--dependency','afterok:'+str(dep)]
    if dry:cmd+=['--dry-run']
    proc=subprocess.run(cmd+['--argv']+argv,capture_output=True,text=True,cwd=ROOT,timeout=45)
    print(proc.stdout,flush=True)
    if proc.returncode:raise RuntimeError(proc.stderr or proc.stdout)
    if dry:return 'DRY'
    for line in proc.stdout.splitlines():
        if line.startswith('submitted '):return line.split()[1]
    raise RuntimeError('submitter returned no job ID')


def placement(mode,cell,smoke):
    """Broaden short engineering checks; keep unproven A30 production excluded."""
    excluded={'g-06-01','g-08-06'}
    partition='h200' if cell['model']=='gemma3-12b' and not smoke else 'h100,h200'
    small=cell['model'] in {'llama32-3b','gemma3-4b'}
    a30_workload=mode=='probes' or (cell['domain']=='fmt' and mode in {'robust','recipe','train_smoke'})
    if smoke and small and a30_workload:
        partition='h100,h200,a30'
        excluded.update({'g-01-01','g-02-01','g-03-01'})
    return partition,','.join(sorted(excluded))


def schedule(max_new=4,dry=False):
    if not (suite.DATA/'manifest.json').exists():
        print('paper finish: frozen data preparation pending');return 0
    suite.frozen_data()
    queue=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T'],
                         capture_output=True,text=True,check=True,timeout=30).stdout
    live={fields[1]:fields[0] for line in queue.splitlines() if len(fields:=line.split('|'))==3}
    quarantined={line.split()[0] for line in (ROOT/'runs/feeder/quarantine.txt').read_text().splitlines() if line.strip()}
    all_items=[];cfg=suite.plan()
    for mode,key in [('probes','probe'),('robust','robust'),('transfer','transfer'),('recipe','recipe')]:
        for cell in cfg[key+'_panel']:
            if mode=='recipe':all_items.append(('train_smoke',cell,17,True))
            all_items.append((mode,cell,17,True))
            for seed in cfg['seeds']:
                if mode=='recipe':all_items.append(('train',cell,seed,False))
                all_items.append((mode,cell,seed,False))
    report=[]; launched=0
    enabled=(ROOT/'runs/feeder/paper_finish.enabled').exists()
    for mode,cell,seed,smoke in all_items:
        job_name=name(mode,cell,seed,smoke);state='ready';dep=None
        if completed(mode,cell,seed,smoke):state='complete'
        elif job_name in live:state='submitted'
        elif job_name.split('_',1)[1] in quarantined:state='quarantined'
        elif not smoke and not enabled:state='awaiting debug enablement'
        else:
            try:
                if mode not in ('train','train_smoke','recipe'):
                    suite.systems_for(mode,cell,seed)
                if mode=='recipe':
                    prereq='train_smoke' if smoke else 'train'
                    prerequisite=name(prereq,cell,seed,smoke)
                    if completed(prereq,cell,seed,smoke):
                        suite.systems_for(mode,cell,seed,smoke)
                    elif prerequisite in live:dep=live[prerequisite]
                    else:state='awaiting recipe train'
                if not smoke:
                    checks=[('recipe',cell,17,True)] if mode=='train' else [(mode,cell,17,True)]
                    if not all(completed(*check) for check in checks):state='awaiting successful allocation-local smoke'
                if state=='ready' and launched<max_new:
                    jid=submit(mode,cell,seed,smoke,dep,dry)
                    live[job_name]=jid;launched+=1;state='submitted' if not dry else 'dry-run'
            except FileNotFoundError as exc:state='awaiting adapters: '+str(exc)
        report.append(dict(mode=mode,cell=cell,seed=seed,smoke=smoke,state=state,job_name=job_name,
                           job_id=live.get(job_name)))
    if not dry:
        atomic_json(ROOT/'runs/feeder/paper_finish_status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            production_enabled=enabled,launched=launched,items=report))
    counts={state:sum(r['state']==state for r in report) for state in sorted({r['state'] for r in report})}
    print('paper finish:',json.dumps(counts),flush=True)
    return launched


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new',type=int,default=4);parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:suite.prepare_data();return 0
    lock=ROOT/'runs/feeder/.paper_finish.lock'
    with lock.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('paper finish scheduling already active');return 0
        schedule(args.max_new,args.dry_run)
    if not args.dry_run:
        subprocess.run([sys.executable,str(ROOT/'scripts/106_paper_synthesis.py')],cwd=ROOT,check=True,timeout=120)
        # Build the final local evidence bundle once production campaigns have closed.
        report=read(ROOT/'runs/feeder/paper_finish_status.json')
        items=[item for item in report.get('items',[]) if not item['smoke']]
        if items and all(item['state']=='complete' for item in items):
            subprocess.run([sys.executable,str(ROOT/'scripts/107_paper_artifact.py'),'--build'],
                           cwd=ROOT,check=True,timeout=120)
    return 0


if __name__=='__main__':raise SystemExit(main())
