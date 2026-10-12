#!/usr/bin/env python
"""Verify real judgment data, tokenizer lengths, baseline bytes and matched budgets."""
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import arms,domains,prompts
from bidir.config import DATA_DIR,load_config,resolve_model
from bidir.mixture import build_mixture
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs
from bidir.train import adapter_dir,measure_lengths,_effective_train_knobs


def main():
    from transformers import AutoTokenizer
    from obtune import prompts as oprompts
    from obtune.provenance import sha256_dir
    spec=importlib.util.spec_from_file_location('pj_preflight_worker',ROOT/'scripts/124_paired_judgment_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    results=[]
    for domain in sorted({c['domain'] for c in worker.plan()['cells']}):
        reference=build_mixture(arms.resolve(worker.plan()['replacement_arm']),domain,'train',worker.plan()['seed'])
        pool=read_pairs(DATA_DIR/domain/'train.jsonl')
        normal=worker.judgment_rows(reference,pool,domain,worker.plan()['seed'])
        shuffled=worker.judgment_rows(reference,pool,domain,worker.plan()['seed'],True)
        if [r.model_dump(exclude={'task'}) for r in normal]!=[r.model_dump(exclude={'task'}) for r in shuffled]:
            raise ValueError('shuffled control inputs changed')
        for cell in [c for c in worker.plan()['cells'] if c['domain']==domain]:
            initial=worker.signature(cell);model=cell['model'];cfg=resolve_model(model)
            tok=AutoTokenizer.from_pretrained(cfg['hf_id'],local_files_only=True)
            if tok.pad_token is None:tok.pad_token=tok.eos_token
            baseline=worker.read(adapter_dir(domain,model,'sft',32,worker.plan()['seed'])/'training_summary.json')
            for arm in ['sft','mix5']:
                directory=adapter_dir(domain,model,arm,32,worker.plan()['seed'])
                manifest=worker.read(directory/'run_manifest.json')
                if (directory/'WITHDRAWN.txt').exists() or sha256_dir(directory/'final')!=manifest['adapter']['sha256']:
                    raise ValueError('baseline checkpoint changed')
            knobs=_effective_train_knobs(load_config(worker.plan()['recipe']),cfg)
            for arm,rows in zip(worker.plan()['new_arms'],[normal,shuffled]):
                mod=domains.get(domain)
                examples=[oprompts.to_trl_example(worker.judgment_example(r.task,r.model_dump())
                    if r.task in ('pos','neg') else prompts.build_example(r.model_dump(),mod),tok) for r in rows]
                keep,stats=measure_lengths(examples,[r.task for r in rows],tok,knobs['max_seq_len'])
                if stats['n_kept']!=baseline['lengths']['n_kept'] or not all(stats['supervised_tokens_by_task'].get(t,0)>0 for t in ['fwd','pos','neg']):
                    raise ValueError('token filtering changes budget or removes a judgment class')
                if worker.signature(cell)!=initial:raise ValueError('protocol changed during preflight')
                results.append(dict(cell,arm=arm,signature=initial,lengths=stats,baseline_steps=baseline['steps']))
                print(domain,model,arm,'kept',stats['n_kept'],'dropped',stats['n_dropped'],flush=True)
    atomic_json(ROOT/'runs/feeder/paired_judgment_data_preflight.json',dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),rows=results,
        preflight_sha256=worker.sha(__file__),worker_sha256=worker.sha(worker.__file__),
        scope='Real data, baseline hashes and token filtering; GPU training/generation still require separate checks.'))
    return 0


if __name__=='__main__':raise SystemExit(main())
