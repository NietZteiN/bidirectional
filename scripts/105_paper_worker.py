#!/usr/bin/env python
"""Allocation-local paper-completion evaluation and restartable recipe workers."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bidir import paper_suite as suite
from bidir.config import load_config, resolve_thresholds
from bidir.pipeline_state import atomic_json
_WORKER_HASH = None


def find_cell(mode, domain, model):
    key = {'probes':'probe','robust':'robust','transfer':'transfer','recipe':'recipe',
           'train':'recipe','train_smoke':'recipe'}[mode] + '_panel'
    return next(c for c in suite.plan()[key] if c['domain']==domain and c['model']==model)


def code_hash():
    global _WORKER_HASH
    if _WORKER_HASH is not None: return _WORKER_HASH
    from lm_eval.tasks.ifeval import utils
    paths = [Path(__file__), ROOT/'src/bidir/paper_suite.py', ROOT/'src/bidir/prompts.py',
             ROOT/'src/bidir/engine.py', ROOT/'src/bidir/train.py', Path(utils.__file__)]
    _WORKER_HASH = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    return _WORKER_HASH


def train_pack(cell, seed, smoke):
    cfg = suite.plan(); variants = ['rank_high'] if smoke else cfg['recipe_variants']
    failed=[]; results=[]
    for variant in variants:
        knobs = cfg['recipe_variants'][variant]
        for arm in (['sft'] if smoke else ['sft',cell['mix']]):
            out=suite.recipe_dir(cell,seed,variant,arm,smoke)
            if suite.complete_adapter(out):
                results.append(dict(variant=variant,arm=arm,status='skipped')); continue
            out.mkdir(parents=True,exist_ok=True)
            recipe=load_config('train/_base_lora.yaml')
            recipe['train']['lr']=knobs['lr']
            if smoke:
                recipe['train'].update(per_device_batch=1,grad_accum=4)
            config_file=out/'resolved_recipe.json'; atomic_json(config_file,recipe)
            cmd=[sys.executable,'-m','bidir.train','--domain',cell['domain'],'--model',cell['model'],
                '--arm',arm,'--seed',str(seed),'--rank',str(knobs['rank']),
                '--train-config',str(config_file),'--out',str(out)]
            if smoke: cmd+=['--max-steps','2']
            start=time.monotonic(); print('TRAIN',variant,arm,flush=True)
            rc=subprocess.run(cmd,cwd=ROOT).returncode
            results.append(dict(variant=variant,arm=arm,exit_code=rc,runtime_s=time.monotonic()-start))
            if rc or not suite.complete_adapter(out): failed.append(f'{variant}/{arm}')
            else:
                summary=json.loads((out/'training_summary.json').read_text())
                adapter=json.loads((out/'final/adapter_config.json').read_text())
                if (adapter['r']!=knobs['rank'] or adapter['lora_alpha']!=2*knobs['rank']
                    or summary['effective_batch']!=(4 if smoke else 64)
                    or summary['steps']!=(2 if smoke else summary['steps']) or summary['steps']<=0):
                    failed.append(f'{variant}/{arm}:invalid recipe/batch/steps')
    path=suite.result_dir('train',cell,seed,smoke)
    atomic_json(path/'summary.json',dict(mode='train',cell=cell,seed=seed,engineering_smoke=smoke,
        plan_sha256=suite.sha(suite.PLAN),worker_sha256=code_hash(),results=results,
        failed=failed,finished_utc=datetime.now(timezone.utc).isoformat()))
    if failed: raise RuntimeError(f'paper recipe pack failed: {failed}')
    return 0


def probe_score(task, document, output):
    if task=='ifeval':
        from lm_eval.tasks.ifeval.utils import process_results
        result=process_results(document,[output])
        return dict(strict=int(result['prompt_level_strict_acc']),
                    loose=int(result['prompt_level_loose_acc']),
                    instruction_strict=result['inst_level_strict_acc'],
                    instruction_loose=result['inst_level_loose_acc'])
    target=document['answer'].split('####')[-1].strip().replace(',','').replace('$','').rstrip('.')
    strict=re.findall(r'#### (\-?[0-9\.\,]+)',output)
    flexible=re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)',output)
    clean=lambda x:x.replace(',','').replace('$','').rstrip('.')
    answer=clean(strict[0]) if strict else None
    flex=clean(''.join(flexible[-1])) if flexible else None
    return dict(strict=int(answer==target),flexible=int(flex==target),target=target,
                extracted_strict=answer,extracted_flexible=flex)


def probe_messages(task, doc, demos):
    if task=='ifeval': user=doc['prompt']
    else:
        blocks=[f"Question: {r['question']}\nAnswer: {r['answer']}" for r in demos]
        user='\n\n'.join(blocks+[f"Question: {doc['question']}\nAnswer:"])
    return [{'role':'system','content':suite.NEUTRAL_SYSTEM},{'role':'user','content':user}]


def requests(mode, cell, systems, smoke):
    from bidir import domains, prompts
    from bidir.domains._common import cluster_key
    cfg=suite.plan(); limit=cfg['smoke_limit'] if smoke else None
    domain=domains.get(cell['domain']); reqs=[]
    if mode=='probes':
        demos=suite.read_rows(suite.DATA/'gsm8k_demos.jsonl')
        for task in cfg['probes']:
            documents=suite.read_rows(suite.DATA/f'{task}.jsonl')
            if limit: documents=documents[:limit]
            for s,adapter in systems.items():
                for doc in documents:
                    reqs.append(dict(endpoint=task,system=s,adapter=adapter,doc=doc,
                        pair_id=doc['pair_id'],cluster_id=doc['pair_id'],
                        messages=probe_messages(task,doc,demos)))
        return reqs
    if mode=='transfer': datasets={'opus':suite.read_rows(suite.DATA/'opus.jsonl')}
    else:
        datasets={'original':suite.read_rows(suite.DATA.parent/cell['domain']/'test.jsonl')}
        if mode=='robust' and cell['domain']=='units':
            datasets['units_magnitude']=suite.read_rows(suite.DATA/'units_magnitude.jsonl')
        if mode=='robust' and cell['domain']=='py_cpp':
            datasets['py_cpp_piecewise_quadratic']=suite.read_rows(suite.DATA/'py_cpp_piecewise_quadratic.jsonl')
    for dataset, documents in datasets.items():
        if limit: documents=documents[:limit]
        variants=cfg['prompt_variants'] if mode=='robust' and dataset=='original' else {'primary':'{instruction}'}
        for s,adapter in systems.items():
            for variant,template in variants.items():
                for direction in ['forward','reverse']:
                    for doc in documents:
                        msgs=prompts.build_messages(doc,direction,domain)
                        msgs[-1]['content']=template.format(instruction=msgs[-1]['content'])
                        reqs.append(dict(endpoint=f'{dataset}/{variant}/{direction}',system=s,
                            adapter=adapter,doc=doc,pair_id=doc['pair_id'],direction=direction,
                            cluster_id=getattr(domain,'cluster_key',cluster_key)(doc),messages=msgs))
    return reqs


def score_requests(mode, cell, reqs, outputs):
    from bidir import domains, prompts
    domain=domains.get(cell['domain'])
    dcfg=resolve_thresholds(load_config(f"domains/{cell['domain']}.yaml"),cell['model'])
    groups=defaultdict(list); rows=[None]*len(reqs)
    for i,request in enumerate(reqs): groups[request['endpoint']].append(i)
    for endpoint,indices in groups.items():
        print('SCORE',endpoint,len(indices),flush=True)
        if mode=='probes':
            scored=[probe_score(endpoint,reqs[i]['doc'],outputs[i]) for i in indices]
            cleaned=[outputs[i] for i in indices]
        else:
            cleaned=[prompts.extract_answer(outputs[i]) for i in indices]
            scored=domain.score_batch(reqs[indices[0]]['direction'],cleaned,[reqs[i]['doc'] for i in indices],dcfg)
        for i,output,metrics in zip(indices,cleaned,scored):
            r=reqs[i]
            rows[i]=dict(endpoint=endpoint,system=r['system'],pair_id=r['pair_id'],cluster_id=r['cluster_id'],
                direction=r.get('direction'),output_raw=outputs[i],output=output,
                prompt_sha256=hashlib.sha256(json.dumps(r['messages'],sort_keys=True).encode()).hexdigest(),**metrics)
    return rows


def oracle(mode, cell, reqs):
    """Exercise real scoring functions before generating measured outputs."""
    from bidir import domains, prompts
    if mode=='probes':
        gsm=[r['doc'] for r in reqs if r['endpoint']=='gsm8k' and r['system']=='base'][:12]
        if not all(probe_score('gsm8k',d,d['answer'])['strict']==1 for d in gsm):
            raise RuntimeError('GSM8K gold scorer failed')
        ifeval=[r['doc'] for r in reqs if r['endpoint']=='ifeval' and r['system']=='base']
        # Preflight EVERY retained instruction to surface missing NLTK resources and invalid validators.
        for d in ifeval:
            probe_score('ifeval',d,'Oracle preflight response. This is a deliberately generic response.')
            if probe_score('ifeval',d,'')['strict']:
                raise RuntimeError('IFEval empty-output oracle failed')
        return {'gsm8k_gold':True,'ifeval_validators_preflighted':len(ifeval)}
    domain=domains.get(cell['domain'])
    dcfg=resolve_thresholds(load_config(f"domains/{cell['domain']}.yaml"),cell['model'])
    reports={}
    for endpoint in dict.fromkeys(r['endpoint'] for r in reqs):
        candidates=[r for r in reqs if r['endpoint']==endpoint and r['system']=='base'][:12]
        direction=candidates[0]['direction']; docs=[r['doc'] for r in candidates]
        gold=[prompts.completion_for(d,direction) for d in docs]
        controls=[*gold,*(['']*len(docs)),*[prompts.input_for(d,direction) for d in docs],
                  *(['invalid output !@#$']*len(docs))]
        scored=domain.score_batch(direction,controls,docs*4,dcfg); n=len(docs)
        gold_scores,empty_scores,echo_scores,garbage=[scored[k*n:(k+1)*n] for k in range(4)]
        rate=lambda scores:sum(s['strict'] for s in scores)/len(scores)
        reports[endpoint]=dict(gold=rate(gold_scores),empty=rate(empty_scores),echo=rate(echo_scores),garbage=rate(garbage))
        # Translation's model/language-based gold ceiling is recorded, not silently changed.
        if ((not cell['domain'].startswith('mt_') and rate(gold_scores)!=1)
            or rate(empty_scores) or rate(garbage) or (direction=='reverse' and rate(echo_scores))):
            raise RuntimeError(f'scorer oracle failed: {endpoint}: {reports[endpoint]}')
    return reports


def generation(mode, cell, seed, smoke):
    from bidir import engine as eng
    from bidir.evaluate import adapter_effectiveness
    manifest=suite.frozen_data(); systems=suite.systems_for(mode,cell,seed,smoke)
    reqs=requests(mode,cell,systems,smoke)
    out=suite.result_dir(mode,cell,seed,smoke); out.mkdir(parents=True,exist_ok=True)
    # Audit persisted before any scored generation; failed audit cannot be mistaken for results.
    audit=oracle(mode,cell,reqs); atomic_json(out/'oracle.json',audit)
    engine=eng.get_engine(cell['model'],dict(max_model_len=8192,max_loras=max(8,len(systems)-1),
        max_cpu_loras=32,max_lora_rank=64,gpu_memory_utilization=.75))
    raw=[None]*len(reqs); tokens=[None]*len(reqs)
    batches=defaultdict(list)
    for i,r in enumerate(reqs): batches[r['endpoint']].append(i)
    gate={}
    # OOD eligibility is measured before tuned generation, with the same resident engine.
    if mode=='transfer':
        from bidir import domains,prompts
        domain=domains.get(cell['domain'])
        dcfg=resolve_thresholds(load_config(f"domains/{cell['domain']}.yaml"),cell['model'])
        indices=[i for i,r in enumerate(reqs) if r['system']=='base']
        outputs,nt=eng.generate(engine,[reqs[i]['messages'] for i in indices],[None]*len(indices),
            load_config(f"domains/{cell['domain']}.yaml").get('sampling',{}))
        base_rows=score_requests(mode,cell,[reqs[i] for i in indices],outputs)
        for i,o,n in zip(indices,outputs,nt): raw[i]=o;tokens[i]=n
        for endpoint in batches:
            group=[r for r in base_rows if r['endpoint']==endpoint]
            documents=[r['doc'] for r in reqs if r['endpoint']==endpoint and r['system']=='base']
            direction=endpoint.rsplit('/',1)[-1]
            echo=domain.score_batch(direction,[prompts.input_for(d,direction) for d in documents],documents,dcfg)
            gate[endpoint]=dict(rate=sum(r['strict'] for r in group)/len(group),
                format_fail=sum(bool(r.get('off_target') or r.get('empty_output')) for r in group)/len(group),
                echo_probe_strict=sum(r['strict'] for r in echo)/len(echo))
        # Unchanged ordinary gate conjunction (scripts/15_base_gate.py), frozen OOD tau.
        if not smoke and not transfer_gate_passed(gate):
            suite.write_rows(out/'trials.jsonl',base_rows)
            atomic_json(out/'summary.json',dict(mode=mode,domain=cell['domain'],model=cell['model'],seed=seed,
                engineering_smoke=smoke,eligibility_passed=False,gate=gate,plan_sha256=suite.sha(suite.PLAN),
                worker_sha256=code_hash(),data_manifest_sha256=suite.sha(suite.DATA/'manifest.json'),
                trial_sha256=suite.sha(out/'trials.jsonl'),systems={'base':None},
                n_instances_by_endpoint=suite.validate_coverage(base_rows,{'base':None}),
                oracle=audit,finished_utc=datetime.now(timezone.utc).isoformat()))
            subprocess.run([sys.executable,str(ROOT/'scripts/94_archive_trials.py')],cwd=ROOT,check=True)
            print('OOD eligibility failed; boundary result retained, no tuned generations',flush=True)
            return 0
    for endpoint,indices in batches.items():
        indices=[i for i in indices if raw[i] is None]
        if not indices: continue
        sampling=(dict(temperature=0,top_p=1,seed=17,max_tokens=suite.plan()['probes'][endpoint]['max_tokens'])
                  if mode=='probes' else load_config(f"domains/{cell['domain']}.yaml").get('sampling',{}))
        if endpoint=='gsm8k': sampling['stop']=['Question:','</s>','<|im_end|>']
        print('GENERATE',endpoint,len(indices),flush=True)
        outputs,nt=eng.generate(engine,[reqs[i]['messages'] for i in indices],[reqs[i]['adapter'] for i in indices],sampling)
        if len(outputs)!=len(indices): raise RuntimeError('generation coverage mismatch')
        for i,o,n in zip(indices,outputs,nt): raw[i]=o;tokens[i]=n
    rows=score_requests(mode,cell,reqs,raw)
    for row,nt in zip(rows,tokens): row['n_gen_tokens']=int(nt)
    suite.write_rows(out/'trials.jsonl',rows)
    coverage=suite.validate_coverage(rows,systems)
    # Reuse the established effectiveness guard while retaining endpoint identities.
    effect_rows=[dict(r,direction=r['endpoint'],strategy='simple') for r in rows]
    effects=adapter_effectiveness(effect_rows)
    failures=[s for s,e in effects.items() if e['identical_rate']>=.999]
    means={endpoint:{s:sum(r['strict'] for r in rows if r['endpoint']==endpoint and r['system']==s)/n
                     for s in systems} for endpoint,n in coverage.items()}
    summary=dict(mode=mode,domain=cell['domain'],model=cell['model'],seed=seed,engineering_smoke=smoke,
        eligibility_passed=True,gate=gate,systems=systems,n_instances_by_endpoint=coverage,means=means,
        adapter_effectiveness=effects,plan_sha256=suite.sha(suite.PLAN),worker_sha256=code_hash(),
        data_manifest_sha256=suite.sha(suite.DATA/'manifest.json'),trial_sha256=suite.sha(out/'trials.jsonl'),
        engine_version=engine.version(),oracle=audit,finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'summary.json',summary)
    if failures: raise RuntimeError(f'adapter effectiveness failed: {failures}; trials preserved')
    if not smoke:
        atomic_json(out/'contrasts.json',dict(trial_sha256=summary['trial_sha256'],
            contrasts=suite.paired_intervals(rows,systems),means=means,
            plan_sha256=summary['plan_sha256'],worker_sha256=summary['worker_sha256']))
    subprocess.run([sys.executable,str(ROOT/'scripts/94_archive_trials.py')],cwd=ROOT,check=True)
    print('COMPLETE',mode,cell,seed,'smoke' if smoke else 'production',flush=True)
    return 0


def transfer_gate_passed(gate):
    return bool(gate) and all(g['rate']>=.10 and g['format_fail']<=.15 and
                              g['echo_probe_strict']<=.02 for g in gate.values())


def main():
    code_hash()  # Freeze the implementation that this process actually loaded.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['probes','robust','transfer','recipe','train','train_smoke'])
    parser.add_argument('--domain',required=True); parser.add_argument('--model',required=True)
    parser.add_argument('--seed',type=int,default=17); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); suite.frozen_data()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('paper worker requires a compute allocation')
    cell=find_cell(args.mode,args.domain,args.model)
    if args.mode in ('train','train_smoke'): return train_pack(cell,args.seed,args.mode=='train_smoke')
    return generation(args.mode,cell,args.seed,args.smoke)


if __name__=='__main__':
    from bidir.engine import shutdown_and_exit
    try: shutdown_and_exit(main())
    except Exception:
        import traceback
        traceback.print_exc(); shutdown_and_exit(1)
