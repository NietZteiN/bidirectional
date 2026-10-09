#!/usr/bin/env python
"""Idempotent Amendment52 admission; production requires a complete matching smoke."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx
from bidir.pipeline_state import atomic_json


def launch(job,dry=False):
    partition='h200' if job['model']=='gemma3-12b' else 'h100,h200'
    duration=('01:30:00' if job['mode']=='hf' else '01:00:00') if job['smoke'] else '08:00:00'
    argv=['scripts/113_mechanism_worker.py','--domain',job['domain'],'--model',job['model'],
          '--seed',str(job['seed']),'--mode',job['mode']]+(['--smoke'] if job['smoke'] else [])
    command=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',job['name'],
        '--partition',partition,'--exclude','g-06-01,g-08-06','--time',duration,'--cpus','8','--mem','64G']
    if dry:command.append('--dry-run')
    result=subprocess.run(command+['--argv']+argv,cwd=ROOT,capture_output=True,text=True,check=True,timeout=45)
    print(result.stdout,flush=True)
    if dry:return 'DRY'
    jid=next(line.split()[1] for line in result.stdout.splitlines() if line.startswith('submitted '))
    # Explicitly put explanatory diagnostics behind the user's priority-one domain/CL panel.
    subprocess.run(['scontrol','update','JobId='+jid,'Nice=1000'],check=True,capture_output=True,timeout=30)
    return jid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-new',type=int,default=30);args=p.parse_args()
    if args.prepare:mx.prepare();return 0
    marker=ROOT/'runs/feeder/mechanism.enabled'
    if not marker.exists() and not args.dry_run:print('mechanism diagnostics await tested enablement');return 0
    if marker.exists():
        enabled=json.loads(marker.read_text())
        if enabled['worker_sha256']!=mx.code_hash() or enabled['panel_sha256']!=mx.sha(mx.DATA/'manifest.json'):
            raise ValueError('mechanism enablement is stale')
    with (ROOT/'runs/feeder/.mechanism_queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('mechanism scheduler already active');return 0
        raw=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],capture_output=True,text=True,
                           check=True,timeout=30).stdout
        live={line.split('|')[1]:line.split('|')[0] for line in raw.splitlines() if '|' in line}
        quarantine=ROOT/'runs/feeder/quarantine.txt'
        blocked=set(quarantine.read_text().split()) if quarantine.exists() else set()
        report=[];count=0
        for job in mx.items():
            jid=live.get(job['name']);state='ready'
            if mx.completed(job):state='complete'
            elif jid:state='submitted'
            elif job['name'].split('_',1)[1] in blocked:state='quarantined'
            elif not job['smoke'] and not mx.completed(mx.smoke_for(job)):state='awaiting matching smoke'
            if state=='ready' and count<args.max_new:
                jid=launch(job,args.dry_run);state='submitted';count+=1
            report.append(dict(job,state=state,job_id=jid))
        if not args.dry_run:atomic_json(ROOT/'runs/feeder/mechanism_status.json',
            dict(updated_utc=datetime.now(timezone.utc).isoformat(),items=report))
        print('Mechanism states:',json.dumps({s:sum(j['state']==s for j in report) for s in sorted({j['state'] for j in report})}))
    return 0


if __name__=='__main__':raise SystemExit(main())
