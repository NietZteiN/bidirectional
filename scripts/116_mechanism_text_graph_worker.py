#!/usr/bin/env python
"""Versioned Gemma text-gradient worker; keep the registered data/objectives/steps fixed."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx,mechanism_text_graph as fix
from bidir.pipeline_state import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',required=True);p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--mode',choices=['hf'],default='hf')
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('worker requires a compute GPU allocation')
    cell=next(c for c in mx.plan()['cells'] if c['domain']==args.domain and c['model']==args.model)
    job=mx.item(cell,args.seed,'hf',args.smoke)
    if job not in mx.items() or not fix.applies(job):raise ValueError('unregistered repaired job')
    marker=json.loads((ROOT/'runs/feeder/mechanism_text_graph.enabled').read_text())
    code=fix.code_hash();panel=mx.sha(mx.DATA/'manifest.json')
    if marker['worker_sha256']!=code or marker['panel_sha256']!=panel:raise ValueError('stale repair enablement')
    if not args.smoke and not fix.completed(mx.smoke_for(job)):raise ValueError('matching repaired smoke required')
    out=fix.output(job);out.mkdir(parents=True,exist_ok=True);adapters=mx.adapter_inventory(job)
    atomic_json(out/'started.json',dict(protocol=fix.PROTOCOL,job=job,worker_sha256=code,panel_sha256=panel,
        adapters=adapters,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat()))
    spec=importlib.util.spec_from_file_location('frozen_mechanism_worker',ROOT/'scripts/113_mechanism_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    with fix.text_only_peft() as captures:
        details=worker.hf(job,adapters,out)
        excluded=[]
        for parameters in captures:
            excluded += [dict(name=name,numel=param.numel(),requires_grad=param.requires_grad)
                         for name,param in parameters() if fix.excluded(name)]
    if not excluded or any(p['requires_grad'] for p in excluded):raise ValueError('vision factors were not frozen')
    atomic_json(out/'text_graph_projection.json',dict(excluded_parameters=excluded,
        excluded_numel=sum(p['numel'] for p in excluded),subspace='text LoRA A/B only',
        strict_text_autograd=True,allow_unused=False,
        reason='Text-only inputs never invoke the vision tower or multimodal projector.'))
    if fix.code_hash()!=code or mx.sha(mx.DATA/'manifest.json')!=panel:raise ValueError('repair source/data changed during job')
    if mx.adapter_inventory(job)!=adapters:raise ValueError('source checkpoint changed')
    mx.selected(job['domain'])
    files=['likelihood.jsonl','steps.jsonl','gradients.json','contrasts.json','text_graph_projection.json']
    atomic_json(out/'summary.json',dict(protocol=fix.PROTOCOL,job=job,valid=True,
        worker_sha256=code,original_protocol_sha256=mx.code_hash(),panel_sha256=panel,adapters=adapters,
        slurm_job_id=os.environ['SLURM_JOB_ID'],file_sha256={f:mx.sha(out/f) for f in files},
        gradient_subspace='text-only LoRA factors; disconnected vision factors excluded',
        finished_utc=datetime.now(timezone.utc).isoformat(),**details))
    print('TEXT GRAPH COMPLETE',fix.name(job),json.dumps(details),flush=True)
    return 0


if __name__=='__main__':
    try:rc=main()
    except BaseException:traceback.print_exc();rc=1
    from bidir import engine
    engine.shutdown_and_exit(rc)
