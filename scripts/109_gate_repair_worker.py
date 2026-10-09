#!/usr/bin/env python
"""GPU-local audited repair gates and fresh stricter OPUS campaigns."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import domains, engine, prompts, gate_repair as repair, paper_suite as suite
from bidir.config import DATA_DIR, load_config
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs


def module(name, file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def gate_batch(model, legacy=False, d2t=False):
    cfg = repair.plan()
    cells = ['d2t_schema_v2'] if d2t else (cfg['legacy_cells'] if legacy else cfg['cells'])
    audit = module('repair_audit','16_audit_criteria.py')
    out = repair.OUT/('d2t_gates' if d2t else ('legacy' if legacy else 'gates'))/model
    out.mkdir(parents=True,exist_ok=True)
    frozen_data = {cell:repair.data_hash(cell) for cell in cells}
    proofs, gates, rows = {}, {}, []
    resident = None
    for cell in cells:
        domain = domains.get(cell)
        local = load_config(f'domains/{cell}.yaml')
        proof = audit.audit_criterion(cell,cfg['oracle_n'])
        problems = audit.verdict(proof) if legacy else repair.oracle_problems(proof)
        proofs[cell] = proof
        if problems:
            gates[cell] = dict(passes_gate=False,reason='criterion oracle failed',problems=problems)
            continue
        if resident is None:
            resident = engine.get_engine(model,dict(max_model_len=4096,max_loras=8,gpu_memory_utilization=.45 if d2t else .75,seed=17))
        pool = [pair.model_dump() for pair in read_pairs(DATA_DIR/cell/'test.jsonl')]
        instances = random.Random(cfg['gate_seed']).sample(pool,min(cfg['gate_n'],len(pool)))
        directions = {}
        for direction in ['forward','reverse']:
            messages = [prompts.build_messages(inst,direction,domain) for inst in instances]
            raw, tokens = engine.generate(resident,messages,[None]*len(messages),local.get('sampling',{}))
            if len(raw)!=len(instances) or len(tokens)!=len(instances):
                raise RuntimeError('repair gate generation coverage mismatch')
            outputs = [prompts.extract_answer(output) for output in raw]
            scored = domain.score_batch(direction,outputs,instances,local)
            if len(scored)!=len(instances) or any(row.get('strict') not in (0,1) for row in scored):
                raise RuntimeError('repair gate scoring coverage mismatch')
            for inst, output, original, count, metrics in zip(instances,outputs,raw,tokens,scored):
                rows.append(dict(cell=cell,model=model,direction=direction,pair_id=inst['pair_id'],
                    output=output,output_raw=original,n_gen_tokens=count,**metrics))
            directions[direction] = dict(rate=sum(r['strict'] for r in scored)/len(scored),
                format_fail=sum(bool(r.get('off_target') or r.get('empty_output')) for r in scored)/len(scored),
                echo=sum(r.get('echo',0) for r in scored)/len(scored))
        passed = all(row['rate']>=cfg['min_base_rate'] and row['format_fail']<=cfg['max_format_fail']
                     for row in directions.values())
        gates[cell] = dict(passes_gate=passed,n=len(instances),directions=directions)
        print('GATE',cell,model,json.dumps(gates[cell]),flush=True)
        suite.write_rows(out/'trials.jsonl',rows)
    atomic_json(out/'oracle.json',proofs)
    suite.write_rows(out/'trials.jsonl',rows)
    if frozen_data != {cell:repair.data_hash(cell) for cell in cells}:
        raise RuntimeError('repair data changed during the gate')
    atomic_json(out/'summary.json',dict(protocol=cfg['protocol'],model=model,gates=gates,
        worker_sha256=repair.code_hash(),data_sha256=frozen_data,
        oracle_sha256=repair.sha(out/'oracle.json'),
        trial_sha256=repair.sha(out/'trials.jsonl') if (out/'trials.jsonl').exists() else None,
        finished_utc=datetime.now(timezone.utc).isoformat()))
    # A failed scientific gate is a completed diagnostic, not a retryable infrastructure failure.
    return 0


def d2t_oracle():
    cfg = repair.plan()
    audit = module('repair_d2t_audit','16_audit_criteria.py')
    out = repair.OUT/'d2t_schema_v2';out.mkdir(parents=True,exist_ok=True)
    proof = audit.audit_criterion('d2t_schema_v2',cfg['d2t_oracle_n'])
    problems = repair.oracle_problems(proof)
    atomic_json(out/'summary.json',dict(protocol=cfg['protocol'],oracle=proof,problems=problems,
        passes_oracle=not problems,worker_sha256=repair.code_hash(),
        data_sha256=repair.data_hash('d2t_schema_v2'),finished_utc=datetime.now(timezone.utc).isoformat()))
    print('D2T schema oracle:',json.dumps(dict(problems=problems,proof=proof)),flush=True)
    return 0


def transfer(model, seed, smoke=False):
    worker = module('repair_transfer_worker','105_paper_worker.py')
    from bidir.domains import mt_noecho_v2
    original_get = domains.get
    def repaired_domain(cell):
        found = original_get(cell)
        return mt_noecho_v2 if cell=='mt_de-en' else found
    domains.get = repaired_domain
    out = repair.OUT/'transfer'/model/f's{seed}'/('smoke' if smoke else 'production')
    # Reuse the frozen data, resident-engine pairing and effectiveness guard in
    # an isolated process. Only scorer, proof hash and result namespace change.
    suite.result_dir = lambda *args,**kwargs:out
    worker.code_hash = repair.code_hash
    cell = next(cell for cell in suite.plan()['transfer_panel'] if cell['model']==model)
    rc = worker.generation('transfer',cell,seed,smoke)
    summary = json.loads((out/'summary.json').read_text())
    summary.update(protocol=repair.plan()['protocol'],repair_plan_sha256=repair.sha(repair.PLAN))
    atomic_json(out/'summary.json',summary)
    return rc


def contrastive_smoke(model,arm):
    import torch
    trainer=module('repair_trainer','110_gate_repair_train.py')
    out=repair.OUT/'contrastive_smokes'/model/arm
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    rc=trainer.single(['--domain',repair.plan()['contrastive_cell'],'--model',model,'--arm',arm,
        '--train-config','train/contrastive_smoke.yaml','--max-steps','2','--out',str(out)])
    training=json.loads((out/'training_summary.json').read_text())
    atomic_json(out/'summary.json',dict(protocol=repair.plan()['protocol'],model=model,arm=arm,
        worker_sha256=repair.code_hash(),data_sha256=repair.data_hash(repair.plan()['contrastive_cell']),
        negative_pool_sha256=repair.sha(DATA_DIR/'fmt/train.jsonl'),
        training_sha256=repair.sha(out/'training_summary.json'),steps=training['steps'],
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_s=time.monotonic()-started,
        exit_code=rc,finished_utc=datetime.now(timezone.utc).isoformat()))
    return rc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['gate','legacy','d2t','d2t_gate','transfer','cl_smoke'],required=True)
    parser.add_argument('--model');parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--arm')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('repair worker requires a compute allocation')
    repair.code_hash()
    if args.mode=='d2t':return d2t_oracle()
    if args.mode=='transfer':return transfer(args.model,args.seed,args.smoke)
    if args.mode=='cl_smoke':return contrastive_smoke(args.model,args.arm)
    return gate_batch(args.model,args.mode=='legacy',args.mode=='d2t_gate')


if __name__=='__main__':
    try:rc=main()
    except BaseException:
        traceback.print_exc();rc=1
    engine.shutdown_and_exit(rc)
