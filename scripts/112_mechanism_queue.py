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
from bidir import mechanism_suite as mx, mechanism_text_graph as text_graph
from bidir.pipeline_state import atomic_json
from bidir import paper_sprint


def launch(job,dry=False):
    partition='h200' if job['model']=='gemma3-12b' else 'h100,h200'
    # Full HF production was validated on H100 for both Gemma sizes during the sprint.
    # Retain that eligibility after reopening the registered seeds.
    if job['mode']=='hf':partition='h100,h200'
    duration=('01:30:00' if job['mode']=='hf' else '01:00:00') if job['smoke'] else '08:00:00'
    repaired=text_graph.applies(job)
    if repaired:duration='00:20:00' if job['smoke'] else '04:00:00'
    # All four fixed seed17 HF cells finished in under ten minutes. Two hours preserves
    # generous headroom with unchanged samples/controls and improves backfill eligibility.
    if job['mode']=='hf' and not job['smoke']:duration='02:00:00'
    job_name=text_graph.name(job) if repaired else job['name']
    worker='scripts/116_mechanism_text_graph_worker.py' if repaired else 'scripts/113_mechanism_worker.py'
    argv=[worker,'--domain',job['domain'],'--model',job['model'],
          '--seed',str(job['seed']),'--mode',job['mode']]+(['--smoke'] if job['smoke'] else [])
    command=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',job_name,
        '--partition',partition,'--exclude','g-06-01,g-08-06','--time',duration,'--cpus','8','--mem','64G']
    if dry:command.append('--dry-run')
    result=subprocess.run(command+['--argv']+argv,cwd=ROOT,capture_output=True,text=True,check=True,timeout=45)
    print(result.stdout,flush=True)
    if dry:return 'DRY'
    jid=next(line.split()[1] for line in result.stdout.splitlines() if line.startswith('submitted '))
    # Explicitly put explanatory diagnostics behind the user's priority-one domain/CL panel.
    nice='0' if paper_sprint.plan() else '1000'
    subprocess.run(['scontrol','update','JobId='+jid,'Nice='+nice],check=True,capture_output=True,timeout=30)
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
    repair_marker=ROOT/'runs/feeder/mechanism_text_graph.enabled'
    if repair_marker.exists():
        repair=json.loads(repair_marker.read_text())
        if repair['worker_sha256']!=text_graph.code_hash() or repair['panel_sha256']!=mx.sha(mx.DATA/'manifest.json'):
            raise ValueError('text-graph repair enablement is stale')
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
            repaired=text_graph.applies(job)
            job_name=text_graph.name(job) if repaired else job['name']
            complete=text_graph.completed if repaired else mx.completed
            jid=live.get(job_name);state='ready'
            if complete(job):state='complete'
            elif paper_sprint.deferred_reason(job_name):
                state=paper_sprint.deferred_reason(job_name)
                if jid and not args.dry_run:
                    # Pending work cannot start after the sprint deadline. Running selected
                    # diagnostics retain their bounded walltime and finish writing evidence.
                    pending=subprocess.run(['squeue','-h','-j',jid,'-t','PENDING','-o','%i'],
                        capture_output=True,text=True,check=True,timeout=30).stdout.strip()
                    if pending:subprocess.run(['scancel',jid],check=True,timeout=30)
            elif jid:state='submitted'
            elif job_name.split('_',1)[1] in blocked:state='quarantined'
            elif repaired and not (ROOT/'runs/feeder/mechanism_text_graph.enabled').exists():state='awaiting tested text-graph repair'
            elif not job['smoke'] and not complete(mx.smoke_for(job)):state='awaiting matching smoke'
            if state=='ready' and count<args.max_new:
                jid=launch(job,args.dry_run);state='submitted';count+=1
            report.append(dict(job,state=state,job_id=jid,**(text_graph.public_status(job) if repaired else {})))
        if not args.dry_run:atomic_json(ROOT/'runs/feeder/mechanism_status.json',
            dict(updated_utc=datetime.now(timezone.utc).isoformat(),items=report))
        print('Mechanism states:',json.dumps({s:sum(j['state']==s for j in report) for s in sorted({j['state'] for j in report})}))
    return 0


if __name__=='__main__':raise SystemExit(main())
