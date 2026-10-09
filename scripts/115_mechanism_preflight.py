#!/usr/bin/env python
"""Reproduce Amendment52 checkpoint/tokenizer/norm preflight; optionally enable queue."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx,domains,prompts
from bidir.config import resolve_model
from bidir.pipeline_state import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse',action='store_true',help='verify a completed current preflight instead of repeating it')
    p.add_argument('--enable',action='store_true');p.add_argument('--test-log',type=Path)
    args=p.parse_args();cfg=mx.plan();mx.prepare()
    path=ROOT/'runs/feeder/mechanism_preflight.json'
    if args.reuse:
        report=json.loads(path.read_text())
        if report['worker_sha256']!=mx.code_hash() or report['panel_sha256']!=mx.sha(mx.DATA/'manifest.json'):
            raise ValueError('preflight source/data changed')
        for row in report['adapters']:
            job=mx.item({k:v for k,v in row.items() if k in ['domain','model','role']},row['seed'],'hf')
            if row['arms']!=mx.adapter_inventory(job):raise ValueError('preflight checkpoint changed')
        for cell in cfg['cells']:mx.selected(cell['domain'])
    else:
        from transformers import AutoTokenizer
        spec=importlib.util.spec_from_file_location('mx_preflight_worker',ROOT/'scripts/113_mechanism_worker.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        report=dict(worker_sha256=mx.code_hash(),panel_sha256=mx.sha(mx.DATA/'manifest.json'),cells=[],adapters=[])
        for cell in cfg['cells']:
            tok=AutoTokenizer.from_pretrained(resolve_model(cell['model'])['hf_id'],local_files_only=True)
            test=mx.selected(cell['domain']);val=mx.selected(cell['domain'],'val');mod=domains.get(cell['domain']);longest=0
            for direction in ['forward','reverse']:
                shuffled=worker.shuffled_targets(test,direction)
                for inst in test[:cfg['likelihood_n']]:
                    for answer in [prompts.completion_for(inst,direction),prompts.input_for(inst,direction),shuffled[inst['pair_id']]]:
                        ids,_=mx.completion_ids(tok,prompts.build_messages(inst,direction,mod,'simple'),answer,cfg['max_seq_len'])
                        longest=max(longest,len(ids))
                for inst in val:
                    ids,_=mx.completion_ids(tok,prompts.build_messages(inst,direction,mod,'simple'),
                        prompts.completion_for(inst,direction),cfg['max_seq_len']);longest=max(longest,len(ids))
            for seed in cfg['seeds']:
                report['adapters'].append(dict(cell,seed=seed,arms=mx.adapter_inventory(mx.item(cell,seed,'hf'))))
            source=mx.adapter_inventory(mx.item(cell,17,'layers'))['sft']['path']
            variants=worker.make_variants(source,mx.output(mx.item(cell,17,'layers',True)),cell['model'])
            report['cells'].append(dict(cell,max_seq_len=longest,variants_checked=len(variants)))
            print('PREFLIGHT PASS',cell,longest,flush=True)
        if mx.code_hash()!=report['worker_sha256']:raise ValueError('source changed during preflight')
        atomic_json(path,report)
    if args.enable:
        if not args.test_log:raise ValueError('--enable requires a passing pytest --test-log')
        log=args.test_log.read_text()
        if not re.search(r'\b\d+ passed(?:, \d+ warnings)? in [\d.]+s',log) or re.search(r'\b\d+ (failed|errors?)\b',log):
            raise ValueError('test log does not show a successful completed pytest run')
        atomic_json(ROOT/'runs/feeder/mechanism.enabled',dict(worker_sha256=mx.code_hash(),panel_sha256=mx.sha(mx.DATA/'manifest.json'),
            preflight_sha256=mx.sha(path),test_log_sha256=mx.sha(args.test_log),preflight_script_sha256=mx.sha(__file__),
            tested_utc=datetime.now(timezone.utc).isoformat()))
    print('Mechanism preflight verified; queue '+('enabled' if args.enable else 'unchanged'))
    return 0


if __name__=='__main__':raise SystemExit(main())
