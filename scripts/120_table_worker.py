#!/usr/bin/env python
"""GPU checks and production work for the frozen table-completion panel."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bidir.config import DATA_DIR, RESULTS_DIR, RUNS_DIR, load_config, resolve_model
from bidir.pipeline_state import atomic_json, evaluation_succeeded
from bidir.train import adapter_dir

PLAN = ROOT/'configs/table_completion.json'
STATE = RUNS_DIR/'table_completion_v1'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    return json.loads(PLAN.read_text())


def key(cell):
    return cell['domain']+'_'+cell['model']+'_s'+str(plan()['seed'])


def job_name(cell, phase):
    if phase == 'eval':
        return 'ev_'+key(cell)+'_'+cell['tag']
    return 'tc_'+phase+'_'+key(cell)


def recipe(cell):
    return 'train/table_completion_aux.yaml' if cell['tag'].startswith('attrib_') else 'train/_base_lora.yaml'


def signature(cell):
    paths = [Path(__file__), PLAN, ROOT/'configs/train/_base_lora.yaml',
             ROOT/'configs/train/table_completion_aux.yaml', ROOT/'configs/models.yaml',
             ROOT/'configs/domains'/f"{cell['domain']}.yaml"]
    paths += [ROOT/'src/bidir'/name for name in ['train.py','evaluate.py','arms.py','mixture.py',
               'losses.py','collators.py','prompts.py','schema.py']]
    paths += list((ROOT/'src/bidir').rglob('*.py'))
    paths += [ROOT/'scripts/95_runner.py']
    paths += [DATA_DIR/cell['domain']/(s+'.jsonl') for s in ['train','val','test']]
    return hashlib.sha256(json.dumps({str(p):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()


def adapter_valid(cell, arm, out=None, smoke=False):
    out = out or adapter_dir(cell['domain'],cell['model'],arm,32,plan()['seed'])
    try:
        if (out/'WITHDRAWN.txt').exists() or not (out/'final/adapter_model.safetensors').exists(): return False
        manifest = json.loads((out/'run_manifest.json').read_text())
        training = json.loads((out/'training_summary.json').read_text())
        if not manifest.get('finished_utc') or not manifest.get('adapter',{}).get('sha256'): return False
        if (training['domain'],training['model'],training['arm'],training['seed']) != (
                cell['domain'],cell['model'],arm,plan()['seed']): return False
        if training['steps'] <= 0 or (smoke and training['steps'] != plan()['smoke_steps']): return False
        if training['effective_batch'] != 64: return False
        tokens = (training.get('direction_exposure') or {}).get('supervised_tokens_by_direction',{})
        counts = training.get('lengths',{}).get('supervised_tokens_by_task',{})
        if arm == 'unlikelihood': return tokens.get('conflict_unlikelihood',0)>0
        if arm == 'roundtrip': return tokens.get('roundtrip_reverse_ce',0)>0
        if arm == 'cft': return counts.get('pos',0)>0 and counts.get('neg',0)>0
        if arm.startswith('mix') and arm[3:].isdigit():
            return training['balance']['by_task'].get('rev',0)>0 and counts.get('rev',0)>0
        return True
    except (OSError,KeyError,ValueError): return False


def receipt_path(cell, phase):
    return STATE/key(cell)/(phase+'.json')


def completed(cell, phase):
    try:
        r = json.loads(receipt_path(cell,phase).read_text())
        if r['signature'] != signature(cell): return False
        success = any(json.loads(p.read_text()).get('exit_code') == 0 and
                      json.loads(p.read_text()).get('finished_utc','') >= r['finished_utc']
                      for p in (ROOT/'runs/status').glob(job_name(cell,phase)+'.*.json'))
        if not success: return False
        if phase in ['smoke','train']:
            return all(adapter_valid(cell,a,Path(d),phase=='smoke') and
                       sha(Path(d)/'run_manifest.json')==r['manifest_sha256'][a]
                       for a,d in r['adapters'].items())
        run = Path(r['run'])
        if not evaluation_succeeded(run,ROOT/'runs/status',job_name(cell,phase)): return False
        report = load_module('table_rates',ROOT/'scripts/117_main_results.py')
        means = report.campaign_rates(run,r['trial_sha256'],{})
        return set(means)==set(cell['arms'])
    except (OSError,KeyError,ValueError): return False


def ready_phase(cell):
    baseline = set(cell['arms'])-set(cell['new_arms'])-{'base'}
    if not all(adapter_valid(cell,a) for a in baseline): return 'blocked: missing or withdrawn baseline'
    missing = [a for a in cell['new_arms'] if not adapter_valid(cell,a)]
    if missing:
        if not completed(cell,'smoke'): return 'smoke'
        return 'train'
    if not completed(cell,'eval_smoke'): return 'eval_smoke'
    if not completed(cell,'eval'): return 'eval'
    return 'complete'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',required=True)
    parser.add_argument('--phase',choices=['smoke','train','eval_smoke','eval'],required=True)
    args=parser.parse_args()
    cell=next(c for c in plan()['cells'] if key(c)==args.cell)
    runner=load_module('table_original_gate',ROOT/'scripts/95_runner.py')
    if runner.gate_passed(cell['domain'],cell['model']) is not True:
        raise ValueError('unchanged original base gate has not passed')
    initial_signature=signature(cell)
    r=dict(signature=initial_signature,phase=args.phase,cell=cell,job=os.environ.get('SLURM_JOB_ID'))
    if args.phase in ['smoke','train']:
        if args.phase=='train' and not completed(cell,'smoke'):
            raise ValueError('training requires a current successful GPU smoke')
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable on allocated node')
        r['gpu']=torch.cuda.get_device_name(0)
        r['adapters']={};r['manifest_sha256']={}
        for arm in cell['new_arms']:
            production=adapter_dir(cell['domain'],cell['model'],arm,32,plan()['seed'])
            if args.phase=='smoke' and adapter_valid(cell,arm): continue
            out=(STATE/key(cell)/'smoke_adapters'/arm if args.phase=='smoke' else production)
            if not adapter_valid(cell,arm,out,args.phase=='smoke'):
                if (out/'WITHDRAWN.txt').exists(): raise ValueError('withdrawn output; preserve and investigate')
                cmd=[sys.executable,'-m','bidir.train','--domain',cell['domain'],'--model',cell['model'],
                     '--arm',arm,'--seed',str(plan()['seed']),'--rank','32','--train-config',recipe(cell),'--out',str(out)]
                if args.phase=='smoke':cmd+=['--max-steps',str(plan()['smoke_steps'])]
                subprocess.run(cmd,cwd=ROOT,check=True)
            if not adapter_valid(cell,arm,out,args.phase=='smoke'):
                raise ValueError('training output lacks steps, completion hash, batch or realized objective exposure')
            r['adapters'][arm]=str(out);r['manifest_sha256'][arm]=sha(out/'run_manifest.json')
    else:
        if not all(adapter_valid(cell,a) for a in cell['arms'] if a!='base'):
            raise ValueError('evaluation requires every complete, effective-exposure adapter')
        if args.phase=='eval' and not completed(cell,'eval_smoke'):
            raise ValueError('production evaluation requires a current shared-pass smoke')
        cmd=[sys.executable,'-m','bidir.evaluate','--domain',cell['domain'],'--model',cell['model'],
             '--seed',str(plan()['seed']),'--arms',','.join(cell['arms']),'--tag',cell['tag']]
        if args.phase=='eval_smoke':
            out=RESULTS_DIR/'table_completion_v1/eval_smokes'/key(cell)
            cmd+=['--limit',str(plan()['eval_smoke_pairs']),'--out',str(out)]
        else:
            # Freeze the output path before launch, including across a UTC midnight.
            out=RESULTS_DIR/datetime.now(timezone.utc).strftime('%Y-%m-%d')/cell['domain']/cell['model']/f"{cell['tag']}_s{plan()['seed']}"
            cmd+=['--out',str(out)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        r['run']=str(out);r['trial_sha256']=sha(out/'trials.jsonl')
        rates=load_module('table_checked_rates',ROOT/'scripts/117_main_results.py')
        means=rates.campaign_rates(out,r['trial_sha256'],{})
        if set(means)!=set(cell['arms']): raise ValueError('shared evaluation arm set changed')
    if signature(cell)!=initial_signature: raise ValueError('protocol, code or data changed during work')
    r['finished_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(receipt_path(cell,args.phase),r)
    print(json.dumps(r,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
