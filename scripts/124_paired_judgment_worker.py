#!/usr/bin/env python
"""Isolated paired-judgment baseline; existing workers, arms and scorers stay frozen."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bidir import arms, domains, prompts
from bidir.config import DATA_DIR, RUNS_DIR, RESULTS_DIR
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs
from bidir.train import adapter_dir

PLAN = ROOT/'configs/paired_judgment.json'
STATE = RUNS_DIR/'paired_judgment_v1'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    return read(PLAN)


def key(cell):
    return cell['domain']+'_'+cell['model']+'_s'+str(plan()['seed'])


def job_name(cell, phase):
    return 'pj_'+phase+'_'+key(cell)


def signature(cell):
    paths = [Path(__file__), PLAN, ROOT/'configs/models.yaml',
             ROOT/'configs/train/_base_lora.yaml', ROOT/'configs'/plan()['recipe'],
             ROOT/'configs/domains'/(cell['domain']+'.yaml'), ROOT/'scripts/95_runner.py']
    paths += list((ROOT/'src/bidir').rglob('*.py'))
    if cell['domain']=='sql':
        from bidir.domains import sql
        paths += list(sql.TEST_SUITE.glob('*.py'))
    paths += [DATA_DIR/cell['domain']/(s+'.jsonl') for s in ['train','val','test']]
    for arm in ['sft','mix5']:
        folder=adapter_dir(cell['domain'],cell['model'],arm,32,plan()['seed'])
        paths += [folder/'run_manifest.json',folder/'training_summary.json']
    return hashlib.sha256(json.dumps({str(p):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()


def output_adapter(cell, arm, smoke=False):
    return STATE/key(cell)/('smoke_adapters' if smoke else 'adapters')/arm


def receipt_path(cell, phase):
    return STATE/key(cell)/(phase+'.json')


def normalized(text):
    return ' '.join(text.split()).casefold()


def sql_negative(source, candidate, timeout_s):
    """Only credit a valid same-database query that fails the frozen equality check."""
    from bidir.domains import sql
    db = sql.db_path(source.meta['db_id'])
    deadline = time.monotonic()+timeout_s
    try:
        with closing(sqlite3.connect('file:'+db+'?mode=ro',uri=True)) as con:
            con.execute('PRAGMA query_only=ON')
            con.set_progress_handler(lambda: int(time.monotonic()>deadline), 1000)
            for query in [source.side_b, candidate.side_b]:
                con.execute(query).fetchall()
    except sqlite3.Error:
        return False
    if str(sql.TEST_SUITE) not in sys.path:sys.path.insert(0,str(sql.TEST_SUITE))
    from exec_eval import eval_exec_match
    try:
        return not bool(eval_exec_match(db=db,p_str=candidate.side_b,g_str=source.side_b,
                                       plug_value=False,keep_distinct=False,progress_bar_for_each_datapoint=False))
    except Exception:
        return False  # A scorer error cannot manufacture a negative label.


def judgment_rows(rows, pool, domain, seed, shuffled=False, checker=None):
    """Use mix5's selected pair IDs; replace each reverse row with one judgment."""
    if any(p.split!='train' for p in pool):raise ValueError('negative pool must contain training pairs only')
    selected = sorted((r for r in rows if r.task=='rev'),key=lambda r:r.pair_id)
    if len(selected)<2:
        raise ValueError('too few selected pairs for both judgment classes')
    rng = random.Random(seed)
    rng.shuffle(selected)
    grouped = defaultdict(list)
    known = defaultdict(set)
    for p in pool:
        grouped[p.subtask].append(p)
        known[(p.subtask,normalized(p.side_a))].add(normalized(p.side_b))
    result=[]
    cfg=plan()
    negative_needed=(len(selected)+1)//2
    negative_count=0
    for row in selected:
        is_positive=True
        candidate=row
        if negative_count<negative_needed:
            options=list(grouped[row.subtask]);rng.shuffle(options)
            candidate=None
            for other in options[:cfg['negative_candidates']]:
                if normalized(other.side_a)==normalized(row.side_a):continue
                if normalized(other.side_b) in known[(row.subtask,normalized(row.side_a))] | {normalized(row.side_b)}:continue
                if domain=='sql' and not (checker or sql_negative)(row,other,cfg['sql_query_timeout_s']):continue
                candidate=other;break
            if candidate is None:
                candidate=row
            else:
                is_positive=False
                negative_count+=1
        meta={**row.meta,'judgment_candidate':candidate.side_b,
              'judgment_candidate_pair_id':candidate.pair_id,
              'judgment_candidate_split':candidate.split,
              'judgment_positive':is_positive}
        result.append(row.model_copy(update={'task':'pos' if is_positive else 'neg','meta':meta}))
    if negative_count!=negative_needed:raise ValueError('no checked negative for enough selected pairs')
    if shuffled:
        labels=[r.task for r in result]
        random.Random(seed+591).shuffle(labels)
        result=[r.model_copy(update={'task':label}) for r,label in zip(result,labels)]
        if all(r.task==('pos' if r.meta['judgment_positive'] else 'neg') for r in result):
            raise ValueError('shuffled control did not change any label')
    output=[r for r in rows if r.task=='fwd']+result
    random.Random(seed).shuffle(output)
    assert len(output)==len(rows) and not any(r.task=='rev' for r in output)
    return output


def judgment_example(task, row):
    details={'source':row['side_a'],'candidate':row['meta']['judgment_candidate']}
    if row['domain']=='sql':
        details['database_schema']=row['meta']['schema']
        instruction='Does the SQL candidate answer the source question using this database schema?'
    else:
        instruction='Does the German candidate express the meaning of the English source?'
    return {'prompt':[{'role':'system','content':prompts.SYSTEM},
                      {'role':'user','content':instruction+' Reply Yes or No only.\n'+json.dumps(details,ensure_ascii=False)}],
            'completion':[{'role':'assistant','content':'Yes' if task=='pos' else 'No'}]}


def register_arms():
    for arm in plan()['new_arms']:
        arms.ARMS[arm]=arms.ArmSpec(arm,('fwd','pos','neg'),matched_to='mix5',
                                  role='5% replacement with paired judgments; shuffled-label control')


def fit(cell, arm, out, smoke):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    from bidir import train
    register_arms()
    original_build=train.build_mixture
    original_example=prompts.build_example
    accounting={}
    def build(spec, domain, split, seed, **kwargs):
        if domain!=cell['domain'] or spec.name!=arm:raise ValueError('unregistered training cell')
        reference=original_build(arms.resolve(plan()['replacement_arm']),domain,split,seed,**kwargs)
        pool=read_pairs(DATA_DIR/domain/'train.jsonl')
        rows=judgment_rows(reference,pool,domain,seed,arm.endswith('_shuffle'))
        if split=='train' and smoke:
            per_class=plan()['smoke_class_rows']
            groups={t:[r for r in rows if r.task==t] for t in ['fwd','pos','neg']}
            need={'pos':per_class,'neg':per_class,'fwd':plan()['smoke_rows']-2*per_class}
            if any(len(groups[t])<n for t,n in need.items()):raise ValueError('smoke lacks each required class')
            rows=[r for t,n in need.items() for r in groups[t][:n]]
            random.Random(seed).shuffle(rows)
        accounting[split]={'counts':dict(Counter(r.task for r in rows)),
                           'row_sha256':hashlib.sha256(json.dumps([r.model_dump() for r in rows],sort_keys=True).encode()).hexdigest()}
        if split=='train':
            rows_file=out/'training_rows.jsonl';rows_file.parent.mkdir(parents=True,exist_ok=True)
            rows_file.write_text(''.join(json.dumps(r.model_dump(),ensure_ascii=False)+'\n' for r in rows))
        return rows
    train.build_mixture=build
    prompts.build_example=lambda row,mod: judgment_example(row['task'],row) if row['task'] in ('pos','neg') else original_example(row,mod)
    argv=['--domain',cell['domain'],'--model',cell['model'],'--arm',arm,
          '--seed',str(plan()['seed']),'--rank','32','--train-config',plan()['recipe'],'--out',str(out)]
    if smoke:argv+=['--max-steps',str(plan()['smoke_steps'])]
    initial=signature(cell)
    rc=train.main(argv)
    if rc:raise RuntimeError('trainer failed')
    summary=read(out/'training_summary.json')
    if smoke and (summary['steps']!=plan()['smoke_steps'] or summary['lengths']['n_kept']!=plan()['smoke_rows']):
        raise ValueError('smoke must consume all 256 rows over four effective batches')
    if signature(cell)!=initial:raise ValueError('protocol changed during training')
    atomic_json(out/'paired_judgment_protocol.json',dict(signature=initial,arm=arm,smoke=smoke,
        rows=accounting,training_rows_sha256=sha(out/'training_rows.jsonl'),
        manifest_sha256=sha(out/'run_manifest.json'),finished_utc=datetime.now(timezone.utc).isoformat()))
    return 0


def adapter_valid(cell, arm, smoke=False):
    out=output_adapter(cell,arm,smoke)
    try:
        if (out/'WITHDRAWN.txt').exists():return False
        proof=read(out/'paired_judgment_protocol.json');summary=read(out/'training_summary.json');manifest=read(out/'run_manifest.json')
        counts=summary['lengths']['supervised_tokens_by_task']
        from obtune.provenance import sha256_dir
        return (proof['signature']==signature(cell) and proof['arm']==arm and proof['smoke']==smoke
                and proof['manifest_sha256']==sha(out/'run_manifest.json')
                and proof['training_rows_sha256']==sha(out/'training_rows.jsonl')
                and manifest.get('finished_utc') and manifest['adapter'].get('sha256')
                and sha256_dir(out/'final')==manifest['adapter']['sha256']
                and (out/'final/adapter_model.safetensors').exists()
                and summary['arm']==arm and summary['domain']==cell['domain'] and summary['model']==cell['model']
                and summary['seed']==plan()['seed'] and math.isfinite(summary['train_loss'])
                and summary['steps']>0 and summary['effective_batch']==64
                and all(counts.get(t,0)>0 for t in ['fwd','pos','neg'])
                and summary['balance']['by_task'].get('rev',0)==0
                and (not smoke or summary['steps']==plan()['smoke_steps']))
    except (OSError,KeyError,ValueError):return False


def completed(cell, phase):
    try:
        r=read(receipt_path(cell,phase))
        if r['signature']!=signature(cell):return False
        statuses=list((ROOT/'runs/status').glob(job_name(cell,phase)+'.*.json'))
        if not any(read(p).get('exit_code')==0 and read(p).get('finished_utc','')>=r['finished_utc'] for p in statuses):return False
        if phase in ['smoke','train']:
            return all(adapter_valid(cell,a,phase=='smoke') and
                       sha(output_adapter(cell,a,phase=='smoke')/'run_manifest.json')==r['manifest_sha256'][a]
                       for a in plan()['new_arms'])
        rates=module('pj_rates','117_main_results.py')
        means=rates.campaign_rates(r['run'],r['trial_sha256'],{})
        summary=read(Path(r['run'])/'summary.json')
        expected=plan()['eval_smoke_pairs'] if phase=='eval_smoke' else len(read_pairs(DATA_DIR/cell['domain']/'test.jsonl'))
        return (set(means)==set(plan()['evaluation_arms']) and summary['n_instances']==expected
                and all(adapter_valid(cell,a) for a in plan()['new_arms']))
    except (OSError,KeyError,ValueError):return False


def ready_phase(cell):
    gate=module('pj_gate','95_runner.py')
    if gate.gate_passed(cell['domain'],cell['model']) is not True:return 'blocked: original capability gate'
    for arm in ['sft','mix5']:
        folder=adapter_dir(cell['domain'],cell['model'],arm,32,plan()['seed'])
        if (folder/'WITHDRAWN.txt').exists() or not gate.adapter_complete(cell['domain'],cell['model'],arm,plan()['seed']):
            return 'blocked: missing baseline '+arm
    for phase in ['smoke','train','eval_smoke','eval']:
        if not completed(cell,phase):return phase
    return 'complete'


def evaluate(cell, out, smoke):
    from bidir import evaluate as evaluator
    register_arms()
    original=evaluator.adapter_dir
    evaluator.adapter_dir=lambda domain,model,arm,rank,seed,root='adapters': (
        output_adapter(cell,arm) if arm in plan()['new_arms'] else original(domain,model,arm,rank,seed,root))
    argv=['--domain',cell['domain'],'--model',cell['model'],'--seed',str(plan()['seed']),
          '--arms',','.join(plan()['evaluation_arms']),'--out',str(out),'--tag','paired_judgment_v1']
    if smoke:argv+=['--limit',str(plan()['eval_smoke_pairs'])]
    return evaluator.main(argv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',required=True);p.add_argument('--phase',choices=['smoke','train','eval_smoke','eval'],required=True)
    p.add_argument('--fit-arm');p.add_argument('--out',type=Path);p.add_argument('--evaluate-child',action='store_true')
    args=p.parse_args();cell=next(c for c in plan()['cells'] if key(c)==args.cell)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('worker requires Slurm')
    if args.fit_arm:
        if args.fit_arm not in plan()['new_arms'] or args.phase not in ['smoke','train']:raise ValueError('unregistered fit')
        return fit(cell,args.fit_arm,args.out,args.phase=='smoke')
    if args.evaluate_child:return evaluate(cell,args.out,args.phase=='eval_smoke')
    initial=signature(cell)
    required={'train':'smoke','eval_smoke':'train','eval':'eval_smoke'}
    if args.phase in required and not completed(cell,required[args.phase]):raise ValueError('missing prerequisite '+required[args.phase])
    if module('pj_worker_gate','95_runner.py').gate_passed(cell['domain'],cell['model']) is not True:raise ValueError('original gate failed')
    receipt=dict(signature=initial,cell=cell,phase=args.phase)
    command=[sys.executable,str(Path(__file__).resolve()),'--cell',args.cell,'--phase',args.phase]
    if args.phase in ['smoke','train']:
        receipt['manifest_sha256']={}
        for arm in plan()['new_arms']:
            out=output_adapter(cell,arm,args.phase=='smoke')
            if not adapter_valid(cell,arm,args.phase=='smoke'):
                if (out/'WITHDRAWN.txt').exists():raise ValueError('withdrawn adapter; preserve for diagnosis')
                subprocess.run(command+['--fit-arm',arm,'--out',str(out)],cwd=ROOT,check=True)
            if not adapter_valid(cell,arm,args.phase=='smoke'):raise ValueError('training exposure/completion failed')
            if args.phase=='train':
                baseline=read(adapter_dir(cell['domain'],cell['model'],'sft',32,plan()['seed'])/'training_summary.json')
                actual=read(out/'training_summary.json')
                if actual['steps']!=baseline['steps'] or actual['lengths']['n_kept']!=baseline['lengths']['n_kept']:
                    raise ValueError('training-example/optimizer-step matching failed')
            receipt['manifest_sha256'][arm]=sha(out/'run_manifest.json')
    else:
        if not all(adapter_valid(cell,a) for a in plan()['new_arms']):raise ValueError('incomplete new adapters')
        out=RESULTS_DIR/'paired_judgment_v1'/cell['domain']/cell['model']/('smoke_s17' if args.phase=='eval_smoke' else 'production_s17')
        subprocess.run(command+['--evaluate-child','--out',str(out)],cwd=ROOT,check=True)
        receipt.update(run=str(out),trial_sha256=sha(out/'trials.jsonl'))
        means=module('pj_checked_rates','117_main_results.py').campaign_rates(out,receipt['trial_sha256'],{})
        if set(means)!=set(plan()['evaluation_arms']):raise ValueError('incorrect evaluation arm set')
    if signature(cell)!=initial:raise ValueError('protocol changed during work')
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(receipt_path(cell,args.phase),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    return 0


if __name__=='__main__':
    from bidir import engine
    import traceback
    try:rc=main()
    except BaseException:
        traceback.print_exc();rc=1
    engine.shutdown_and_exit(rc)
