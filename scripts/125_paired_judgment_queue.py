#!/usr/bin/env python
"""Admit one proof-gated phase per paired-judgment cell, with bounded retries."""
import argparse
from datetime import datetime, timezone
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
spec=importlib.util.spec_from_file_location('pj_queue_worker',ROOT/'scripts/124_paired_judgment_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
F=ROOT/'runs/feeder'


def next_action(cell, live, history):
    if any(worker.job_name(cell,p) in live for p in ['smoke','train','eval_smoke','eval']):return 'inflight'
    phase=worker.ready_phase(cell)
    if phase in ['smoke','train','eval_smoke','eval']:
        name=worker.job_name(cell,phase)
        if len(history.get(name,[]))>=worker.plan()['max_attempts']:return 'blocked: attempts exhausted'
        statuses=[worker.read(p) for p in sorted((ROOT/'runs/status').glob(name+'.*.json'))]
        if statuses and statuses[-1].get('exit_code') not in [0,124,137,143]:
            return 'blocked: failure requires diagnosis'
    return phase


def validate_preflight(proof):
    expected={(worker.key(c),a) for c in worker.plan()['cells'] for a in worker.plan()['new_arms']}
    actual={(worker.key(row),row['arm']) for row in proof['rows']}
    if actual!=expected or len(proof['rows'])!=len(expected):raise ValueError('incomplete data preflight')
    if proof['preflight_sha256']!=worker.sha(ROOT/'scripts/127_paired_judgment_preflight.py'):
        raise ValueError('stale preflight implementation')
    for row in proof['rows']:
        if row['signature']!=worker.signature(row):raise ValueError('stale data preflight')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--max-new',type=int,default=4)
    args=p.parse_args()
    if not worker.plan()['enabled']:return 0
    proof=worker.read(F/'paired_judgment_data_preflight.json')
    validate_preflight(proof)
    with (F/'.paired_judgment_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raw=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],text=True,timeout=30)
        live={line.split('|')[1]:line.split('|')[0] for line in raw.splitlines()}
        journal=F/'paired_judgment_admission.json'
        history=worker.read(journal) if journal.exists() else {'attempts':{}}
        rows=[];launched=0
        for cell in worker.plan()['cells']:
            phase=next_action(cell,live,history['attempts'])
            row=dict(cell=cell,phase=phase);rows.append(row)
            if phase not in ['smoke','train','eval_smoke','eval'] or launched>=args.max_new:continue
            name=worker.job_name(cell,phase)
            duration={'smoke':'00:45:00','train':'12:00:00','eval_smoke':'01:00:00','eval':'04:00:00'}[phase]
            # SQL execution/round-trip scoring exceeded 64 GiB in the completed table panel.
            memory='192G' if cell['domain']=='sql' and phase in ['eval_smoke','eval'] else '64G'
            cmd=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',name,
                 '--partition','h100,h200','--exclude','g-06-01,g-08-06',
                 '--time',duration,'--mem',memory,'--cpus','8']
            if args.dry_run:cmd+=['--dry-run']
            cmd+=['--argv','scripts/124_paired_judgment_worker.py','--cell',worker.key(cell),'--phase',phase]
            result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True,timeout=60)
            print(result.stdout,flush=True)
            if not args.dry_run:
                jid=next(line.split()[1] for line in result.stdout.splitlines() if line.startswith('submitted '))
                history['attempts'].setdefault(name,[]).append(jid);atomic_json(journal,history)
                row.update(job=jid,phase='submitted '+phase)
            launched+=1
        report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),rows=rows,launched=launched,
                    complete=sum(r['phase']=='complete' for r in rows),total=len(rows))
        if not args.dry_run:atomic_json(F/'paired_judgment_status.json',report)
        print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
