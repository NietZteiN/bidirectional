#!/usr/bin/env python
"""Amendment52 GPU worker: paired likelihood/gradients or norm-controlled layers."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx, domains, prompts, paper_suite as suite
from bidir.config import load_config,resolve_model,resolve_thresholds
from bidir.pipeline_state import atomic_json


def frozen_job(args):
    cfg=mx.plan()
    cell=next(c for c in cfg['cells'] if c['domain']==args.domain and c['model']==args.model)
    job=mx.item(cell,args.seed,args.mode,args.smoke)
    if job not in mx.items():raise ValueError('unregistered job')
    marker=json.loads((ROOT/'runs/feeder/mechanism.enabled').read_text())
    if marker['worker_sha256']!=mx.code_hash() or marker['panel_sha256']!=mx.sha(mx.DATA/'manifest.json'):
        raise ValueError('enablement does not match tested code/data')
    if not args.smoke and not mx.completed(mx.smoke_for(job)):
        raise ValueError('production requires a successful matching full smoke receipt')
    return job


def oracle(domain,model,out):
    mod=domains.get(domain)
    cfg=resolve_thresholds(load_config('domains/'+domain+'.yaml'),model)
    instances=mx.selected(domain)[:8]
    proof=dict(domain=domain,model=model,n=len(instances),directions={})
    problems=[]
    for direction in ['forward','reverse']:
        probes=dict(gold=[prompts.completion_for(i,direction) for i in instances],
            echo=[prompts.input_for(i,direction) for i in instances],
            empty=['']*len(instances),garbage=['invalid output !@#$']*len(instances))
        scored=mod.score_batch(direction,[o for outputs in probes.values() for o in outputs],
                              instances*len(probes),cfg)
        if len(scored)!=len(instances)*len(probes):raise ValueError('oracle coverage mismatch')
        proof['directions'][direction]={}
        for index,(name,outputs) in enumerate(probes.items()):
            block=scored[index*len(instances):(index+1)*len(instances)]
            rate=sum(int(r['strict'] and not r.get('echo',0)) for r in block)/len(instances)
            proof['directions'][direction][name]=dict(strict_noecho=rate)
            if rate!=(1 if name=='gold' else 0):problems.append(direction+'/'+name)
    atomic_json(out/'oracle.json',proof)
    if problems:raise ValueError('frozen criterion oracle failed: '+str(problems))


def make_variants(source,out,model):
    from safetensors.torch import load_file,save_file
    import torch
    tensors={name:t.to(torch.bfloat16) for name,t in
             load_file(str(Path(source)/'adapter_model.safetensors')).items()}
    # vLLM's auto LoRA dtype follows the bf16 base; match norms after that cast.
    config=json.loads((Path(source)/'adapter_config.json').read_text())
    energies=mx.layer_energies(tensors,config)
    variants=mx.control_coefficients(energies,int(resolve_model(model)['n_layers']),mx.plan()['random_layer_seeds'])
    systems={};receipts={};total=sum(energies.values())
    for label,coeff in variants.items():
        dst=out/'variants'/label;dst.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(source)/'adapter_config.json',dst/'adapter_config.json')
        changed={}
        for name,tensor in tensors.items():
            match=mx.LAYER.search(name)
            changed[name]=(tensor.float()*coeff[int(match.group(1))]).to(tensor.dtype) if 'lora_B' in name else tensor
        save_file(changed,str(dst/'adapter_model.safetensors'),metadata={'format':'pt'})
        actual=sum(mx.layer_energies(changed,config).values())
        expected=sum(energies[l]*c*c for l,c in coeff.items())
        error=abs(math.sqrt(actual)-math.sqrt(expected))/max(math.sqrt(expected),math.sqrt(total)*1e-12)
        if error>mx.plan()['norm_rtol']:raise ValueError('saved control norm differs from matched target')
        edit={n:(changed[n].float()-t.float()) if 'lora_B' in n else t for n,t in tensors.items()}
        # layer_energies rejects a zero total; zero-edit controls are explicitly recorded.
        edit_energy=0. if all(c==1 for c in coeff.values()) else sum(mx.layer_energies(edit,config).values())
        receipts[label]=dict(coefficients=coeff,retained_delta_norm=math.sqrt(actual),
            target_retained_delta_norm=math.sqrt(expected),edit_delta_norm=math.sqrt(edit_energy),
            norm_relative_error=error,weight_sha256=mx.sha(dst/'adapter_model.safetensors'))
        systems[label]=str(dst)
    atomic_json(out/'variant_manifest.json',dict(source_adapter=source,original_delta_norm=math.sqrt(total),
        layer_energy=energies,variants=receipts,matching='retained delta-W Frobenius norm only',norm_tensor_dtype='bfloat16 (vLLM auto LoRA dtype)'))
    return systems


def layers(job,adapters,out):
    from bidir import engine
    cfg=mx.plan();mod=domains.get(job['domain'])
    dcfg=resolve_thresholds(load_config('domains/'+job['domain']+'.yaml'),job['model'])
    instances=mx.selected(job['domain'])[:cfg['smoke_generation_n'] if job['smoke'] else cfg['generation_n']]
    oracle(job['domain'],job['model'],out)
    systems={'base':None,**{a:v['path'] for a,v in adapters.items()}}
    systems.update(make_variants(adapters['sft']['path'],out,job['model']))
    resident=engine.get_engine(job['model'],dict(dcfg.get('engine',{}),max_model_len=cfg['max_seq_len'],
        max_cpu_loras=64,max_loras=8,gpu_memory_utilization=.85))
    rows=[];by={}
    for system,adapter in systems.items():
        for direction in ['forward','reverse']:
            messages=[prompts.build_messages(i,direction,mod,'simple') for i in instances]
            raw,tokens=engine.generate(resident,messages,[adapter]*len(messages),dcfg.get('sampling',{}))
            if len(raw)!=len(instances) or len(tokens)!=len(instances):raise ValueError('generation coverage mismatch')
            outputs=[prompts.extract_answer(r) for r in raw]
            scored=mod.score_batch(direction,outputs,instances,dcfg)
            if len(scored)!=len(instances) or any(r.get('strict') not in (0,1) for r in scored):
                raise ValueError('scoring coverage mismatch')
            by[(system,direction)]={}
            for i,r,output,count,s in zip(instances,raw,outputs,tokens,scored):
                # Amendment52 defines an explicit no-copy predicate in BOTH MT directions.
                # Raw original strict is preserved rather than silently re-labelled.
                strict_noecho=int(s['strict'] and not s.get('echo',0))
                row=dict(system=system,direction=direction,pair_id=i['pair_id'],cluster_id=i['pair_id'],
                    strategy='simple',output_raw=r,output=output,n_gen_tokens=count,**s,
                    strict_noecho=strict_noecho)
                rows.append(row);by[(system,direction)][i['pair_id']]=row
            suite.write_rows(out/'trials.jsonl',rows)
            print('GENERATION',system,direction,len(instances),flush=True)
    expected=len(systems)*2*len(instances)
    if len(rows)!=expected:raise ValueError('incomplete panel')
    # Fresh base and original SFT/replay/mix5 must each visibly load. Edited controls may
    # legitimately recover base behavior: record equality without censoring that outcome.
    effect={}
    for system in systems:
        if system=='base':continue
        identical=sum(by[(system,d)][i['pair_id']]['output_raw']==by[('base',d)][i['pair_id']]['output_raw']
                      for d in ['forward','reverse'] for i in instances)/(2*len(instances))
        effect[system]=dict(identical_rate=identical)
        if system in adapters and identical>=.999:raise ValueError('source adapter failed effectiveness guard: '+system)
    contrasts=[]
    comparisons=[(a,'base') for a in adapters]+[(a,'sft') for a in systems if a.startswith('band')]
    comparisons += [(f'band{b}_remove',f'band{b}_{c}') for b in range(4)
                    for c in ['scale',*[f'random{s}' for s in cfg['random_layer_seeds']]]]
    for system,reference in comparisons:
        for direction in ['forward','reverse']:
            values=[by[(system,direction)][i['pair_id']]['strict_noecho']-
                    by[(reference,direction)][i['pair_id']]['strict_noecho'] for i in instances]
            contrasts.append(dict(system=system,reference=reference,direction=direction,
                metric='strict_noecho',**mx.paired_interval(values,reps=cfg['bootstrap_reps'])))
    atomic_json(out/'contrasts.json',dict(exploratory=job['smoke'],comparisons=contrasts,
        interpretation='All fixed bands/controls reported; intervals within a cell/seed, no best-band selection.'))
    atomic_json(out/'adapter_effectiveness.json',effect)
    return dict(n_pairs=len(instances),n_systems=len(systems),generation_rows=expected,
        candidate_rows=0,gradient_states=0,source_adapter_effectiveness=effect)


def shuffled_targets(instances,direction):
    """Same-subtask evaluation candidates only; never used for fitting/gradients."""
    targets={}
    for inst in instances:
        gold=prompts.completion_for(inst,direction)
        others=[x for x in instances if x['subtask']==inst['subtask'] and
                prompts.completion_for(x,direction)!=gold]
        if not others:raise ValueError('no distinct same-subtask candidate for '+inst['pair_id'])
        offset=int.from_bytes(__import__('hashlib').sha256(inst['pair_id'].encode()).digest()[:8],'big')%len(others)
        targets[inst['pair_id']]=prompts.completion_for(others[offset],direction)
    return targets


def token_loss(model,encoded,grad=False):
    import torch
    import torch.nn.functional as F
    ids,labels=encoded
    x=torch.tensor([ids],device=model.device); y=torch.tensor(labels[1:],device=model.device)
    with (nullcontext() if grad else torch.no_grad()):
        logits=model(input_ids=x,use_cache=False).logits[0,:-1]
        losses=F.cross_entropy(logits.float(),y,ignore_index=-100,reduction='none')
        keep=y!=-100; total=losses[keep].sum();n=int(keep.sum())
    if n<1 or not torch.isfinite(total):raise ValueError('non-finite/empty completion loss')
    return total/n,total,n


def hf(job,adapters,out):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    cfg=mx.plan();mod=domains.get(job['domain']);mconfig=resolve_model(job['model'])
    torch.manual_seed(cfg['selection_seed'])
    tokenizer=AutoTokenizer.from_pretrained(mconfig['hf_id'])
    base=AutoModelForCausalLM.from_pretrained(mconfig['hf_id'],dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map='cuda')
    model=PeftModel.from_pretrained(base,adapters['sft']['path'],adapter_name='sft',is_trainable=True)
    for arm,receipt in adapters.items():
        if arm!='sft':model.load_adapter(receipt['path'],adapter_name=arm,is_trainable=True)
    model.eval();model.config.use_cache=False
    test=mx.selected(job['domain']);val=mx.selected(job['domain'],'val')
    n=cfg['smoke_likelihood_n'] if job['smoke'] else cfg['likelihood_n']
    nc=cfg['smoke_calibration_n'] if job['smoke'] else cfg['calibration_n']
    nh=cfg['smoke_step_test_n'] if job['smoke'] else cfg['step_test_n']
    candidates={d:shuffled_targets(test,d) for d in ['forward','reverse']}
    def encode(inst,direction,answer=None):
        return mx.completion_ids(tokenizer,prompts.build_messages(inst,direction,mod,'simple'),
            prompts.completion_for(inst,direction) if answer is None else answer,cfg['max_seq_len'])
    rows=[];means={}
    for arm in cfg['systems']:
        if arm!='base':model.set_adapter(arm)
        with (model.disable_adapter() if arm=='base' else nullcontext()):
            for direction in ['forward','reverse']:
                for inst in test[:n]:
                    answers=dict(gold=prompts.completion_for(inst,direction),
                        echo=prompts.input_for(inst,direction),shuffled=candidates[direction][inst['pair_id']])
                    for candidate,answer in answers.items():
                        mean,total,tokens=token_loss(model,encode(inst,direction,answer))
                        rows.append(dict(system=arm,direction=direction,pair_id=inst['pair_id'],
                            candidate=candidate,n_tokens=tokens,nll_per_token=float(mean),nll_total=float(total)))
                print('LIKELIHOOD',arm,direction,n,flush=True)
                suite.write_rows(out/'likelihood.jsonl',rows)
    if len(rows)!=len(cfg['systems'])*2*n*3:raise ValueError('candidate coverage incomplete')
    # Numerically verify every trained adapter has an effect, independently of accuracy.
    base_losses={(r['direction'],r['pair_id'],r['candidate']):r['nll_total'] for r in rows if r['system']=='base'}
    for arm in adapters:
        differences=[abs(r['nll_total']-base_losses[(r['direction'],r['pair_id'],r['candidate'])])
                     for r in rows if r['system']==arm]
        if max(differences)<=1e-5:raise ValueError('adapter numerically ineffective: '+arm)
    gradients=[];steps=[]
    for arm in cfg['gradient_systems']:
        model.set_adapter(arm)
        for name,param in model.named_parameters():
            param.requires_grad_('lora_' in name and f'.{arm}.' in name)
        active=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if not active:raise ValueError('no active LoRA parameters')
        # Float32 adapter parameters avoid bf16 rounding turning small interventions into no-ops.
        for name,p in active:p.data=p.data.float()
        ps=[p for name,p in active]
        saved=[p.detach().clone() for p in ps]
        calibration={d:[encode(i,d) for i in val[:nc]] for d in ['forward','reverse']}
        heldout={d:[encode(i,d) for i in test[:nh]] for d in ['forward','reverse']}
        def losses(batch):return [float(token_loss(model,e)[0]) for e in batch]
        before={d:losses(heldout[d]) for d in heldout}
        g={}
        for direction in ['forward','reverse']:
            accum=[torch.zeros_like(p,device='cpu',dtype=torch.float32) for p in ps]
            for enc in calibration[direction]:
                loss,_,_=token_loss(model,enc,True)
                pieces=torch.autograd.grad(loss,ps)
                for a,v in zip(accum,pieces):a.add_(v.detach().float().cpu(),alpha=1/nc)
            g[direction]=accum
        norms={d:math.sqrt(sum(float((v.double()**2).sum()) for v in vs)) for d,vs in g.items()}
        dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(g['forward'],g['reverse']))
        if any(not math.isfinite(v) or v<=0 for v in norms.values()) or not math.isfinite(dot):
            raise ValueError('non-finite/zero gradient')
        gradients.append(dict(system=arm,n_calibration=nc,parameter_names=[name for name,p in active],
            parameter_numel=sum(p.numel() for p in ps),forward_norm=norms['forward'],reverse_norm=norms['reverse'],
            dot=dot,cosine=dot/(norms['forward']*norms['reverse']),
            reverse_change_per_unit_forward_descent=-dot/norms['forward'],
            forward_change_per_unit_reverse_descent=-dot/norms['reverse']))
        random_gen=torch.Generator(device='cpu').manual_seed(cfg['selection_seed'])
        directions=dict(forward=g['forward'],reverse=g['reverse'],
            mixed=[(a+b)/2 for a,b in zip(g['forward'],g['reverse'])],
            random=[torch.randn(p.shape,generator=random_gen,dtype=torch.float32) for p in ps])
        param_norm=math.sqrt(sum(float((p.detach().double()**2).sum()) for p in ps))
        if param_norm<=0:raise ValueError('zero parameter norm')
        for label,vs in directions.items():
            norm=math.sqrt(sum(float((v.double()**2).sum()) for v in vs))
            if norm<=0:raise ValueError('zero step direction')
            for fraction in cfg['step_fractions']:
                requested=fraction*param_norm
                with torch.no_grad():
                    for p,original,v in zip(ps,saved,vs):p.copy_(original-requested*v.to(p.device)/norm)
                actual=math.sqrt(sum(float(((p.detach()-s).double()**2).sum()) for p,s in zip(ps,saved)))
                if actual<=0 or abs(actual-requested)/requested>.01:raise ValueError('small update rounded away')
                for direction in ['forward','reverse']:
                    after=losses(heldout[direction])
                    for inst,b,a in zip(test[:nh],before[direction],after):
                        steps.append(dict(system=arm,step_direction=label,fraction=fraction,direction=direction,
                            pair_id=inst['pair_id'],baseline_nll=b,updated_nll=a,delta_nll=a-b,
                            requested_parameter_norm=requested,actual_parameter_norm=actual))
                with torch.no_grad():
                    for p,s in zip(ps,saved):p.copy_(s)
                if not all(torch.equal(p,s) for p,s in zip(ps,saved)):raise ValueError('adapter not restored exactly')
                suite.write_rows(out/'steps.jsonl',steps)
        print('GRADIENTS',arm,gradients[-1]['cosine'],flush=True)
    suite.write_rows(out/'likelihood.jsonl',rows);suite.write_rows(out/'steps.jsonl',steps)
    atomic_json(out/'gradients.json',dict(states=gradients,parameterization='Local float32 LoRA A/B coordinates',
        interpretation='Gradients fit validation gold only; all equal-norm ephemeral steps evaluated on held-out pairs.'))
    contrasts=[]
    for arm in cfg['gradient_systems']:
        for label in cfg['step_directions']:
            for fraction in cfg['step_fractions']:
                for direction in ['forward','reverse']:
                    deltas=[r['delta_nll'] for r in steps if (r['system'],r['step_direction'],r['fraction'],r['direction'])==
                            (arm,label,fraction,direction)]
                    contrasts.append(dict(system=arm,step_direction=label,fraction=fraction,direction=direction,
                        metric='heldout_delta_nll',**mx.paired_interval(deltas,reps=cfg['bootstrap_reps'])))
    atomic_json(out/'contrasts.json',dict(exploratory=job['smoke'],comparisons=contrasts))
    expected=len(cfg['gradient_systems'])*len(cfg['step_directions'])*len(cfg['step_fractions'])*2*nh
    if len(steps)!=expected:raise ValueError('step coverage incomplete')
    return dict(n_pairs=n,n_systems=len(cfg['systems']),candidate_rows=len(rows),gradient_states=len(gradients),
        step_rows=len(steps),generation_rows=0,calibration_n=nc,step_test_n=nh,
        generation_engine='separate layer mode vLLM; likelihood/steps use HF SDPA')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--mode',choices=['hf','layers'],required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('GPU worker requires a compute allocation')
    job=frozen_job(args);out=mx.output(job);out.mkdir(parents=True,exist_ok=True)
    source_hash=mx.code_hash();panel_hash=mx.sha(mx.DATA/'manifest.json')
    mx.selected(job['domain']);adapters=mx.adapter_inventory(job)
    atomic_json(out/'started.json',dict(job=job,worker_sha256=source_hash,panel_sha256=panel_hash,
        adapters=adapters,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat()))
    details=(hf if args.mode=='hf' else layers)(job,adapters,out)
    if mx.code_hash()!=source_hash or mx.sha(mx.DATA/'manifest.json')!=panel_hash:
        raise ValueError('protocol changed during job')
    if mx.adapter_inventory(job)!=adapters:raise ValueError('adapter source changed during job')
    mx.selected(job['domain'])
    filenames=['likelihood.jsonl','steps.jsonl','gradients.json','contrasts.json'] if args.mode=='hf' else [
        'trials.jsonl','variant_manifest.json','contrasts.json','oracle.json','adapter_effectiveness.json']
    atomic_json(out/'summary.json',dict(protocol=mx.plan()['protocol'],job=job,valid=True,
        worker_sha256=source_hash,panel_sha256=panel_hash,adapters=adapters,
        slurm_job_id=os.environ['SLURM_JOB_ID'],file_sha256={f:mx.sha(out/f) for f in filenames},
        finished_utc=datetime.now(timezone.utc).isoformat(),**details))
    print('MECHANISM COMPLETE',job['name'],json.dumps(details),flush=True)
    return 0


if __name__=='__main__':
    try:rc=main()
    except BaseException:traceback.print_exc();rc=1
    from bidir import engine
    engine.shutdown_and_exit(rc)
