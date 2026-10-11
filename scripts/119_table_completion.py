#!/usr/bin/env python
"""Prioritize every missing numerical table entry; admit only frozen table work."""
import argparse
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
spec=importlib.util.spec_from_file_location('table_worker',ROOT/'scripts/120_table_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
F=ROOT/'runs/feeder'


def next_action(cell, live, attempts):
    # A cell owns one phase at a time; never evaluate partially trained arms.
    if any(worker.job_name(cell,p) in live for p in ['smoke','train','eval_smoke','eval']):return 'inflight'
    phase=worker.ready_phase(cell)
    if phase in ['smoke','train','eval_smoke','eval']:
        n=len(attempts.get(worker.job_name(cell,phase),[]))
        if n>=worker.plan()['max_attempts']:return 'blocked: bounded attempts exhausted for '+phase
    return phase


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--max-new',type=int,default=16)
    args=ap.parse_args()
    if not worker.plan()['enabled']:print('table completion disabled');return 0
    with (F/'.table_completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T'],
                             text=True,capture_output=True,check=True,timeout=30).stdout
        live={r.split('|')[1]:r.split('|')[0] for r in queue.splitlines() if len(r.split('|'))==3}
        journal=F/'table_completion_admission.json'
        history=json.loads(journal.read_text()) if journal.exists() else dict(attempts={})
        attempts=history['attempts'];rows=[];launched=0
        for cell in worker.plan()['cells']:
            phase=next_action(cell,live,attempts)
            row=dict(cell=cell,phase=phase);rows.append(row)
            if phase not in ['smoke','train','eval_smoke','eval'] or launched>=args.max_new:continue
            name=worker.job_name(cell,phase)
            # Do not automatically spend a fresh retry on a deterministic code/scientific failure.
            statuses=[json.loads(p.read_text()) for p in sorted((ROOT/'runs/status').glob(name+'.*.json'))]
            if statuses and statuses[-1].get('exit_code') not in [0,124,137,143]:
                row['phase']='blocked: failed phase requires diagnosis';continue
            duration=('01:00:00' if phase in ['smoke','eval_smoke'] else
                      '12:00:00' if phase=='train' else '04:00:00')
            cmd=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',name,
                 '--partition','h100,h200','--exclude','g-06-01,g-08-06',
                 '--time',duration,'--mem','64G','--cpus','8']
            if args.dry_run:cmd+=['--dry-run']
            cmd+=['--argv','scripts/120_table_worker.py','--cell',worker.key(cell),'--phase',phase]
            result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=60)
            print(result.stdout,flush=True)
            if result.stderr:print(result.stderr,file=sys.stderr,flush=True)
            if result.returncode:raise RuntimeError('table-completion submission failed')
            if not args.dry_run:
                jid=next(line.split()[1] for line in result.stdout.splitlines() if line.startswith('submitted '))
                attempts.setdefault(name,[]).append(jid)
                atomic_json(journal,history);row['job']=jid;row['phase']='submitted '+phase
            launched+=1
        status=dict(updated_utc=datetime.now(timezone.utc).isoformat(),exclusive=worker.plan()['exclusive'],
                    rows=rows,launched=launched,complete=sum(r['phase']=='complete' for r in rows),total=len(rows))
        if not args.dry_run:atomic_json(F/'table_completion_status.json',status)
        print(json.dumps(status,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
