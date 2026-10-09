#!/usr/bin/env python
"""Idempotent, proof-gated admission for Amendment51 repair diagnostics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import gate_repair as repair, paper_suite as suite
from bidir.pipeline_state import atomic_json


def items():
    cfg=repair.plan()
    return ([dict(mode='gate',model=model,name='gr_gate_'+model) for model in cfg['models']]
        +[dict(mode='cl_smoke',model=cfg['contrastive_model'],arm=arm,
               name=f"gr_cl_smoke_{cfg['contrastive_model']}_{arm}") for arm in cfg['contrastive_smoke_arms']]
        +[dict(mode='transfer',model=model,seed=17,smoke=True,name='gr_transfer_'+model+'_s17_smoke')
          for model in cfg['transfer_models']]
        +[dict(mode='d2t',model='granite',name='gr_d2t_schema_oracle')]
        +[dict(mode='d2t_gate',model=cfg['d2t_model'],name='gr_d2t_gate_'+cfg['d2t_model'])]
        +[dict(mode='legacy',model=model,name='gr_legacy_'+model) for model in cfg['legacy_models']]
        +[dict(mode='transfer',model=model,seed=seed,smoke=False,name=f'gr_transfer_{model}_s{seed}')
          for model in cfg['transfer_models'] for seed in cfg['transfer_seeds']])


def output(item):
    if item['mode'] in ['gate','legacy','d2t_gate']:
        bucket={'gate':'gates','legacy':'legacy','d2t_gate':'d2t_gates'}[item['mode']]
        return repair.OUT/bucket/item['model']
    if item['mode']=='d2t':return repair.OUT/'d2t_schema_v2'
    if item['mode']=='cl_smoke':return repair.OUT/'contrastive_smokes'/item['model']/item['arm']
    return repair.OUT/'transfer'/item['model']/f"s{item['seed']}"/('smoke' if item['smoke'] else 'production')


def completed(item):
    if item['mode']=='cl_smoke':return repair.contrastive_smoke_passed(item['model'],item['arm'])
    out=output(item)
    try:
        summary=json.loads((out/'summary.json').read_text())
        if summary['worker_sha256']!=repair.code_hash() or not repair.status_success(item['name'],summary):
            return False
        if item['mode'] in ['gate','legacy','d2t_gate']:
            cells=(['d2t_schema_v2'] if item['mode']=='d2t_gate' else
                   repair.plan()['cells'] if item['mode']=='gate' else repair.plan()['legacy_cells'])
            return (all(summary['data_sha256'][cell]==repair.data_hash(cell) for cell in cells)
                    and summary['trial_sha256']==repair.sha(out/'trials.jsonl')
                    and summary['oracle_sha256']==repair.sha(out/'oracle.json'))
        if item['mode']=='d2t':return summary['data_sha256']==repair.data_hash('d2t_schema_v2')
        if summary['data_manifest_sha256']!=suite.sha(suite.DATA/'manifest.json'):
            return False
        if summary['trial_sha256']!=repair.sha(out/'trials.jsonl'):
            return False
        if summary.get('eligibility_passed'):
            if any(e['identical_rate']>=.999 for e in summary.get('adapter_effectiveness',{}).values()):
                return False
            if not item['smoke']:
                contrasts=json.loads((out/'contrasts.json').read_text())
                if contrasts['trial_sha256']!=summary['trial_sha256']:return False
        return summary.get('repair_plan_sha256')==repair.sha(repair.PLAN)
    except (OSError,KeyError,ValueError):return False


def launch(item,dry):
    large=item['model'] in ['gemma3-12b','llama31-8b','granite'] or item['mode']=='d2t_gate'
    partition='h200' if large else 'h100,h200'
    duration='04:00:00' if item['mode'] in ['gate','legacy','d2t_gate'] else '02:00:00'
    if item.get('smoke'):duration='00:30:00'
    argv=['scripts/109_gate_repair_worker.py','--mode',item['mode'],'--model',item['model']]
    if item['mode']=='transfer':argv+=['--seed',str(item['seed'])]+(['--smoke'] if item['smoke'] else [])
    if item['mode']=='cl_smoke':
        argv+=['--arm',item['arm']];duration='00:30:00'
    command=[sys.executable,str(ROOT/'scripts/slurm/submit.py'),'--name',item['name'],
             '--partition',partition,'--exclude','g-06-01,g-08-06','--time',duration,'--cpus','8','--mem','64G']
    if dry:command.append('--dry-run')
    result=subprocess.run(command+['--argv']+argv,capture_output=True,text=True,timeout=45,check=True,cwd=ROOT)
    print(result.stdout,flush=True)
    if dry:return 'DRY'
    return next(line.split()[1] for line in result.stdout.splitlines() if line.startswith('submitted '))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-new',type=int,default=4)
    args=parser.parse_args()
    if args.prepare:repair.prepare();return 0
    if not (ROOT/'runs/feeder/gate_repair.enabled').exists() and not args.dry_run:
        print('gate repairs await tested enablement');return 0
    with (ROOT/'runs/feeder/.gate_repair.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('gate repair scheduling already active');return 0
        raw=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],capture_output=True,text=True,check=True,timeout=30).stdout
        live={line.split('|')[1]:line.split('|')[0] for line in raw.splitlines() if '|' in line}
        report=[];launched=0
        for item in items():
            state='ready';job=live.get(item['name'])
            if completed(item):state='complete'
            elif job:state='submitted'
            elif item['mode']=='d2t_gate':
                oracle=dict(mode='d2t',model='granite',name='gr_d2t_schema_oracle')
                if not completed(oracle):state='awaiting extractor oracle'
                elif not json.loads((output(oracle)/'summary.json').read_text())['passes_oracle']:
                    state='blocked by extractor oracle'
            elif item['mode']=='transfer' and not item['smoke']:
                smoke=dict(item,seed=17,smoke=True,name='gr_transfer_'+item['model']+'_s17_smoke')
                if not completed(smoke):state='awaiting matching smoke'
            if state=='ready' and launched<args.max_new:
                # Preserve watchdog quarantine; never endlessly retry a technical failure.
                quarantine=ROOT/'runs/feeder/quarantine.txt'
                if quarantine.exists() and item['name'].split('_',1)[1] in quarantine.read_text().split():state='quarantined'
                else:job=launch(item,args.dry_run);launched+=1;state='submitted'
            report.append(dict(item,state=state,job_id=job))
        if not args.dry_run:
            atomic_json(ROOT/'runs/feeder/gate_repair_status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),items=report))
        print('Gate repair states:',json.dumps({state:sum(i['state']==state for i in report) for state in sorted({i['state'] for i in report})}))
    return 0


if __name__=='__main__':raise SystemExit(main())
