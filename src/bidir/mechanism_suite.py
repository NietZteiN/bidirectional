"""Frozen inputs, controls and completion proofs for Amendment52."""
import hashlib
import json
import math
import random
import re
from pathlib import Path
from bidir.config import PROJECT_ROOT as ROOT, DATA_DIR, RESULTS_DIR, load_config, ensure_obtune
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs

PLAN = ROOT/'configs/mechanism_panel.json'
DATA = DATA_DIR/'mechanism_panel'
OUT = RESULTS_DIR/'mechanism_explanation_v1'
LAYER = re.compile(r'\.layers\.(\d+)\.')


def plan(): return json.loads(PLAN.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hash():
    files = [Path(__file__), PLAN, ROOT/'scripts/113_mechanism_worker.py',
        ROOT/'src/bidir/prompts.py', ROOT/'src/bidir/engine.py', ROOT/'src/bidir/evaluate.py',
        ROOT/'src/bidir/config.py', ROOT/'src/bidir/schema.py', ROOT/'configs/models.yaml',
        ROOT/'scripts/16_audit_criteria.py']
    files += list((ROOT/'src/bidir/domains').glob('*.py'))
    configs = {c['domain']:load_config('domains/'+c['domain']+'.yaml') for c in plan()['cells']}
    ensure_obtune()
    import obtune.prompts, obtune.eval_vllm
    files += [Path(obtune.prompts.__file__), Path(obtune.eval_vllm.__file__)]
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(set(files)))+
                          json.dumps(configs,sort_keys=True).encode()).hexdigest()


def prepare():
    from bidir import domains
    from bidir.domains._common import content_key
    cfg=plan(); selected={}
    for cell in cfg['cells']:
        d=cell['domain']
        if d in selected: continue
        mod=domains.get(d); key=getattr(mod,'content_key',content_key)
        pools={s:read_pairs(DATA_DIR/d/(s+'.jsonl')) for s in ['train','val','test']}
        keys={s:{key(p) for p in ps} for s,ps in pools.items()}
        if keys['test'] & (keys['train'] | keys['val']):
            raise ValueError('test content overlaps train/validation: '+d)
        rng=random.Random(cfg['selection_seed'])
        test=rng.sample(pools['test'][cfg['exclude_test_prefix']:],cfg['generation_n'])
        val=rng.sample(pools['val'],cfg['calibration_n'])
        selected[d]=dict(source_sha256={s:sha(DATA_DIR/d/(s+'.jsonl')) for s in pools},
            test_ids=[p.pair_id for p in test], calibration_ids=[p.pair_id for p in val],
            test_overlap_train_val=0, excluded_test_prefix=cfg['exclude_test_prefix'])
    document=dict(protocol=cfg['protocol'],plan_sha256=sha(PLAN),domains=selected)
    path=DATA/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=document:
        raise ValueError('frozen panel differs; version the protocol instead of overwriting it')
    DATA.mkdir(parents=True,exist_ok=True)
    if not path.exists(): atomic_json(path,document)
    return document


def selected(domain,split='test'):
    manifest=json.loads((DATA/'manifest.json').read_text())
    if manifest['plan_sha256']!=sha(PLAN):raise ValueError('panel plan changed')
    entry=manifest['domains'][domain]
    for s,digest in entry['source_sha256'].items():
        if sha(DATA_DIR/domain/(s+'.jsonl'))!=digest:raise ValueError('panel data changed: '+domain+'/'+s)
    pool={p.pair_id:p.model_dump() for p in read_pairs(DATA_DIR/domain/(split+'.jsonl'))}
    return [pool[i] for i in entry['test_ids' if split=='test' else 'calibration_ids']]


def items():
    cfg=plan(); out=[]
    for index in cfg['smoke_cells']:
        for mode in cfg['modes']:
            out.append(item(cfg['cells'][index],17,mode,True))
    for cell in cfg['cells']:
        for seed in cfg['seeds']:
            for mode in cfg['modes']:out.append(item(cell,seed,mode,False))
    return out


def item(cell,seed,mode,smoke=False):
    return dict(cell,seed=seed,mode=mode,smoke=smoke,
        name=f"ev_mx_{mode}_{cell['domain']}_{cell['model']}_s{seed}"+('_smoke' if smoke else ''))


def smoke_for(job):
    cfg=plan()
    cell=next(cfg['cells'][i] for i in cfg['smoke_cells'] if cfg['cells'][i]['model']==job['model'])
    return item(cell,17,job['mode'],True)


def output(job):
    return OUT/job['domain']/job['model']/f"s{job['seed']}"/job['mode']/('smoke' if job['smoke'] else 'production')


def adapter_inventory(job,freeze=True):
    from bidir.train import adapter_dir
    ensure_obtune()
    from obtune.provenance import sha256_dir
    inventory={}
    arms=plan()['systems'][1:] if job['mode']=='hf' else plan()['generation_systems'][1:]
    for arm in arms:
        run=adapter_dir(job['domain'],job['model'],arm,plan()['rank'],job['seed'])
        manifest=run/'run_manifest.json'; final=run/'final'
        doc=json.loads(manifest.read_text())
        if not doc.get('finished_utc') or not doc['adapter'].get('sha256'):
            raise ValueError('incomplete adapter: '+str(run))
        digest=sha256_dir(final) if freeze else doc['adapter']['sha256']
        if digest!=doc['adapter']['sha256']:raise ValueError('adapter bytes differ from training receipt: '+str(run))
        inventory[arm]=dict(path=str(final),manifest=str(manifest),manifest_sha256=sha(manifest),adapter_sha256=digest,
            weight_stat={f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in final.glob('*.safetensors')})
    return inventory


def completed(job):
    out=output(job)
    try:
        receipt=json.loads((out/'summary.json').read_text())
        if not receipt['valid'] or receipt['worker_sha256']!=code_hash():return False
        if receipt['panel_sha256']!=sha(DATA/'manifest.json'):return False
        if receipt['job']!=job:return False
        selected(job['domain'])
        if any(sha(out/name)!=digest for name,digest in receipt['file_sha256'].items()):return False
        if receipt['adapters']!=adapter_inventory(job,freeze=False):return False
        for p in (ROOT/'runs/status').glob(job['name']+'.*.json'):
            state=json.loads(p.read_text())
            if (state.get('exit_code')==0 and state.get('finished_utc','')>=receipt['finished_utc']
                and str(state.get('job_id'))==str(receipt['slurm_job_id'])):return True
    except (OSError,ValueError,KeyError,TypeError):pass
    return False


def layer_energies(tensors,config):
    """Actual effective delta-W squared Frobenius norms; never form a dense BA."""
    import torch
    if config.get('use_dora') or config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('unsupported adapter scaling profile')
    rank=config['r']; scale=config['lora_alpha']/(math.sqrt(rank) if config.get('use_rslora') else rank)
    energies={}
    for name,B in tensors.items():
        if 'lora_B' not in name:continue
        match=LAYER.search(name)
        if not match:raise ValueError('LoRA tensor has no transformer layer: '+name)
        A=tensors[name.replace('lora_B','lora_A')].double(); B=B.double()
        energy=float(torch.sum((B.T@B)*(A@A.T)))*scale**2
        if not math.isfinite(energy) or energy<0:raise ValueError('invalid LoRA energy')
        layer=int(match.group(1));energies[layer]=energies.get(layer,0)+energy
    if not energies or sum(energies.values())<=0:raise ValueError('zero adapter delta')
    return energies


def control_coefficients(energies,n_layers,random_seeds=(101,202),groups=4):
    """Match retained delta norm. Record edit norm separately; it is not matched."""
    total=sum(energies.values()); variants={}
    bounds=[round(i*n_layers/groups) for i in range(groups+1)]
    for band,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        remove=sum(e for l,e in energies.items() if lo<=l<hi)
        variants[f'band{band}_remove']={l:float(not lo<=l<hi) for l in energies}
        variants[f'band{band}_scale']={l:math.sqrt(max(0,1-remove/total)) for l in energies}
        for seed in random_seeds:
            order=list(sorted(energies));random.Random(seed).shuffle(order);left=remove
            coef={l:1. for l in energies}
            for l in order:
                if left<=0:break
                amount=min(left,energies[l]);left-=amount
                coef[l]=math.sqrt(max(0,1-amount/energies[l])) if energies[l]>0 else 1.
            variants[f'band{band}_random{seed}']=coef
    return variants


def completion_ids(tokenizer,messages,answer,max_len=4096):
    from obtune.prompts import render_chat,render_full
    prefix=tokenizer(render_chat(messages,tokenizer),add_special_tokens=False)['input_ids']
    full=tokenizer(render_full(messages,tokenizer,answer),add_special_tokens=False)['input_ids']
    if full[:len(prefix)]!=prefix:raise ValueError('completion chat tokens are not a prompt prefix')
    if len(full)>max_len:raise ValueError(f'sequence length {len(full)} exceeds {max_len}; no silent truncation')
    if len(full)<=len(prefix) or not prefix:raise ValueError('empty completion or prompt')
    labels=[-100]*len(prefix)+full[len(prefix):]
    return full,labels


def paired_interval(values,seed=52017,reps=2000):
    import numpy as np
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('invalid paired contrast')
    rng=np.random.default_rng(seed)
    means=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(reps)])
    return dict(n=len(a),mean=float(a.mean()),ci95=[float(x) for x in np.quantile(means,[.025,.975])])
