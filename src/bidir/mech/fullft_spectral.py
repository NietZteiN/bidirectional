"""Stream full-checkpoint matrix deltas through full-spectrum hard thresholding.

Only attention/MLP projection matrices are filtered. Embeddings, normalization,
heads and other tensors retain tuned values. No claim of an IID noise model is
made: the Donoho-Gavish coefficient is an exploratory intervention here.
"""
import argparse
import json
from pathlib import Path
import shutil

from bidir.mech.spectral import omega
from bidir.pipeline_state import atomic_json

PROJECTIONS = ('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight',
               'gate_proj.weight','up_proj.weight','down_proj.weight')


def filter_delta(delta, tau_scale):
    import torch
    if delta.ndim != 2 or not torch.isfinite(delta).all():
        raise ValueError('delta must be a finite matrix')
    if tau_scale <= 0:
        raise ValueError('tau scale must be positive')
    if not bool(delta.abs().max()):
        return delta.clone(), dict(unchanged=True,rank_kept=0,energy_kept=0.)
    u,s,vh=torch.linalg.svd(delta.float(),full_matrices=False)
    beta=min(delta.shape)/max(delta.shape)
    threshold=omega(beta)*float(s.median())*tau_scale
    keep=s>threshold
    repaired=(u*(s*keep).unsqueeze(0))@vh
    return repaired, dict(unchanged=False,rank_kept=int(keep.sum()),
        rank_considered=len(s),threshold=threshold,beta=beta,
        energy_kept=float((s[keep]**2).sum()/(s**2).sum()),
        median_over='full spectrum',sigma_max=float(s.max()),sigma_median=float(s.median()))


def tensor_index(root):
    from safetensors import safe_open
    result={}
    for file in sorted(root.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                if key in result:
                    raise ValueError(f'duplicate tensor {key}')
                result[key]=file
    if not result:
        raise ValueError(f'no safetensors checkpoint in {root}')
    return result


def repair_checkpoint(base, tuned, dest, tau_scale=1., device='cuda'):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base,tuned,dest=map(Path,(base,tuned,dest))
    if dest.exists():
        raise FileExistsError(f'refusing to overwrite repair {dest}')
    bi,ti=tensor_index(base),tensor_index(tuned)
    if bi.keys()!=ti.keys():
        raise ValueError('base/tuned checkpoint tensor keys differ')
    dest.mkdir(parents=True)
    reports=[]
    for shard in sorted(set(ti.values())):
        output={}
        with safe_open(str(shard),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                value=reader.get_tensor(key)
                if key.endswith(PROJECTIONS):
                    with safe_open(str(bi[key]),framework='pt',device='cpu') as original:
                        before=original.get_tensor(key)
                    if before.shape!=value.shape:
                        raise ValueError(f'shape differs: {key}')
                    delta=value.to(device=device,dtype=torch.float32)-before.to(device=device,dtype=torch.float32)
                    repaired,report=filter_delta(delta,tau_scale)
                    output[key]=(before.to(device=device,dtype=torch.float32)+repaired).to(device='cpu',dtype=value.dtype)
                    reports.append(dict(tensor=key,**report))
                    print(f"[fullft repair] {key}: kept {report['rank_kept']}",flush=True)
                    del delta,repaired,before
                else:
                    output[key]=value
        save_file(output,str(dest/shard.name),metadata={'format':'pt'})
        del output
    changed=[r for r in reports if not r['unchanged']]
    if not changed or not any(r['rank_kept'] for r in changed):
        raise ValueError('repair removed every projection update; output remains incomplete')
    for file in tuned.iterdir():
        if file.is_file() and file.suffix in ('.json','.jinja') and file.name not in ('run_manifest.json',):
            shutil.copyfile(file,dest/file.name)
    from datetime import datetime,timezone
    report=dict(base=str(base.resolve()),tuned=str(tuned.resolve()),tau_scale=tau_scale,
        delta_is_lowrank_by_construction=False,scope='attention and MLP projection matrices',
        other_tensors='tuned values unchanged',noise_assumption_validated=False,
        finished_utc=datetime.now(timezone.utc).isoformat(),modules=reports)
    import hashlib
    report['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['torch_version']=torch.__version__
    report['source_files']={str(p.resolve()):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                            for p in set(bi.values()) | set(ti.values())}
    atomic_json(dest/'repair_manifest.json',report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',required=True)
    ap.add_argument('--tuned',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--tau-scale',type=float,default=1.)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    repair_checkpoint(a.base,a.tuned,a.out,a.tau_scale,a.device)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
