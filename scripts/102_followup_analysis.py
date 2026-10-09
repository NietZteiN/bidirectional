#!/usr/bin/env python
"""Refresh valid full-weight, attribution and revised-domain paired outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.config import RESULTS_DIR
from bidir.pipeline_state import atomic_json,evaluation_succeeded
from bidir.train import adapter_dir
spec=importlib.util.spec_from_file_location('paired_analysis',ROOT/'scripts/50_contrasts.py')
paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)

def validate(rows,systems,n):
    coverage={};clusters={}
    for row in rows:
        if row['strategy']!='simple' or row['system'] not in systems:continue
        key=(row['system'],row['direction']);pid=row['pair_id']
        target=coverage.setdefault(key,set())
        if pid in target:raise ValueError('duplicate trial')
        target.add(pid)
        if row['strict'] not in [0,1]:raise ValueError('nonbinary strict')
        cluster=row.get('cluster_id') or pid
        if pid in clusters and clusters[pid]!=cluster:raise ValueError('inconsistent cluster')
        clusters[pid]=cluster
    common=coverage.get(('base','reverse'),set())
    if len(common)!=n:raise ValueError('incomplete base')
    if any(coverage.get((s,d),set())!=common for s in systems for d in ['forward','reverse']):
        raise ValueError('incomplete/mismatched coverage')

def main():
    count=0
    for summary_file in sorted(RESULTS_DIR.glob('*/*/*/*/summary.json')):
        run=summary_file.parent;tag=run.name.rsplit('_s',1)[0]
        if not (tag in ['fullft_campaign','fullft_projection_repair','explicit_domain_pilot',
                        'generality_scale_pilot','d2t_repaired_pilot',
                        'repair_domain_pilot','repair_legacy_pilot']
                or tag.startswith('attrib_')):continue
        if (run/'WITHDRAWN.txt').exists():continue
        summary=json.loads(summary_file.read_text());domain=summary['domain'];model=summary['model'];seed=summary['seed']
        names=[f'ev_{domain}_{model}_s{seed}_{tag}',f'ev_{tag}_{domain}_{model}_s{seed}']
        methods=set(summary['arms'])&{'cft','unlikelihood','roundtrip'}
        names += [f'at_ev_{a}_{domain}_{model}_s{seed}' for a in methods]
        if tag=='fullft_projection_repair':names.append(f'ev_fullft_repair_{domain}_s{seed}')
        if not any(evaluation_succeeded(run,ROOT/'runs/status',name) for name in names):continue
        exposure={};invalid=False
        for arm in methods:
            root=adapter_dir(domain,model,arm,32,seed)
            if not (root/'training_summary.json').exists() or (root/'WITHDRAWN.txt').exists():invalid=True;break
            d=json.loads((root/'training_summary.json').read_text())
            tokens=(d.get('direction_exposure') or {}).get('supervised_tokens_by_direction',{})
            counts=(d.get('lengths') or {}).get('supervised_tokens_by_task',{})
            valid=(tokens.get('conflict_unlikelihood',0)>0 if arm=='unlikelihood' else
                   tokens.get('roundtrip_reverse_ce',0)>0 if arm=='roundtrip' else
                   counts.get('pos',0)>0 and counts.get('neg',0)>0)
            exposure[arm]=dict(valid=valid,direction_tokens=tokens,task_tokens=counts,
                               train_runtime_s=d.get('train_runtime_s'),steps=d.get('steps'))
            if not valid:invalid=True
        if invalid:continue
        trial=run/'trials.jsonl';digest=hashlib.sha256(trial.read_bytes()).hexdigest()
        code_hash=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'scripts/50_contrasts.py').read_bytes()).hexdigest()
        out=run/'followup_contrasts.json'
        if out.exists():
            previous=json.loads(out.read_text())
            if previous.get('trial_sha256')==digest and previous.get('analysis_sha256')==code_hash:continue
        rows=[json.loads(line) for line in trial.open()];systems=set(summary['arms'])
        validate(rows,systems,summary['n_instances'])
        if any(summary.get('adapter_effectiveness',{}).get(s,{}).get('identical_rate',1)>=.999
               for s in systems-{'base'}):raise ValueError(f'ineffective system: {run}')
        comparisons=list(paired.DEFAULT_CONTRASTS)+[(a,b) for a in methods for b in ['sft','mix5']]
        comparisons += [('fullft_sft','base'),('fullft_mix5','fullft_sft'),('fullft_sft','sft'),
                        ('fullft_mix5','mix5'),('repaired_tau1','fullft_sft')]
        contrasts=[]
        for a,b in comparisons:
            if a in systems and b in systems:
                for direction in ['forward','reverse']:
                    result=paired.paired_delta(rows,a,b,'strict',direction,n_boot=2000)
                    result.pop('verdict',None);result.pop('margin_pp',None)
                    contrasts.append(result)
        means={s:{d:sum(r['strict'] for r in rows if r['system']==s and r['direction']==d and r['strategy']=='simple')/summary['n_instances']
                  for d in ['forward','reverse']} for s in systems}
        atomic_json(out,dict(run=str(run),domain=domain,model=model,seed=seed,means=means,
            trial_sha256=digest,analysis_sha256=code_hash,confidence=.95,n_boot=2000,
            multiplicity='unadjusted exploratory intervals; no equivalence verdict',
            evaluation_layout=summary.get('evaluation_layout','single_engine_pass'),
            engine_restart_caveat='interpret full-weight differences against measured decoding floor',
            auxiliary_exposure=exposure,contrasts=contrasts))
        count+=1
    print(f'followup analysis: {count} reports refreshed');return 0
if __name__=='__main__':raise SystemExit(main())
