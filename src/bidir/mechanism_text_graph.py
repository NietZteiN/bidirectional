"""Amendment53: isolate text-only Gemma gradient repair without invalidating layer proofs."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from bidir import mechanism_suite as mx
from bidir.pipeline_state import atomic_json

PROTOCOL='mechanism_text_graph_v2'
OUT=mx.OUT.parent/PROTOCOL
NON_TEXT={'vision_tower','multi_modal_projector'}


def applies(job):return job['mode']=='hf' and job['model'] in ['gemma3-4b','gemma3-12b']
def name(job):return job['name'].replace('ev_mx_','ev_mx2_',1)
def output(job):return OUT/job['domain']/job['model']/f"s{job['seed']}"/('smoke' if job['smoke'] else 'production')


def code_hash():
    return hashlib.sha256(bytes.fromhex(mx.code_hash())+Path(__file__).read_bytes()+
        (mx.ROOT/'scripts/116_mechanism_text_graph_worker.py').read_bytes()).hexdigest()


def excluded(name):
    parts=name.split('.')
    return bool(NON_TEXT.intersection(parts) and any(p in {'lora_A','lora_B'} for p in parts))


def project_text_graph(model):
    """Exclude only known vision LoRA factors, preserving strict checks for text parameters.

    Legacy checkpoints contain zero-B vision adapters despite exclusion strings in their
    configs. Enumerating those unused factors causes autograd.grad to fail on text inputs.
    Other unused parameters still fail; allow_unused is never used to hide a disconnected
    text adapter. No state_dict or saved weight is altered.
    """
    original_parameters=model.named_parameters
    original_set_adapter=model.set_adapter
    def freeze():
        for label,param in original_parameters():
            if excluded(label):param.requires_grad_(False)
    def parameters(*args,**kwargs):
        for label,param in original_parameters(*args,**kwargs):
            if not excluded(label):yield label,param
    def set_adapter(*args,**kwargs):
        result=original_set_adapter(*args,**kwargs);freeze();return result
    model.named_parameters=parameters
    model.set_adapter=set_adapter
    freeze()
    return original_parameters


@contextmanager
def text_only_peft():
    """Apply the projection to this isolated worker's loaded PEFT model only."""
    from peft import PeftModel
    original=PeftModel.__dict__['from_pretrained']
    captures=[]
    def from_pretrained(cls,*args,**kwargs):
        model=original.__get__(None,cls)(*args,**kwargs)
        parameters=project_text_graph(model);captures.append(parameters)
        return model
    PeftModel.from_pretrained=classmethod(from_pretrained)
    try:yield captures
    finally:PeftModel.from_pretrained=original


def completed(job):
    if not applies(job):return mx.completed(job)
    out=output(job)
    try:
        summary=json.loads((out/'summary.json').read_text())
        if not summary['valid'] or summary['protocol']!=PROTOCOL or summary['worker_sha256']!=code_hash():return False
        if summary['panel_sha256']!=mx.sha(mx.DATA/'manifest.json') or summary['job']!=job:return False
        mx.selected(job['domain'])
        if summary['adapters']!=mx.adapter_inventory(job,freeze=False):return False
        if any(mx.sha(out/f)!=digest for f,digest in summary['file_sha256'].items()):return False
        for path in (mx.ROOT/'runs/status').glob(name(job)+'.*.json'):
            status=json.loads(path.read_text())
            if (status.get('exit_code')==0 and str(status.get('job_id'))==str(summary['slurm_job_id'])
                and status.get('finished_utc','')>=summary['finished_utc']):return True
    except (OSError,ValueError,KeyError,TypeError):pass
    return False


def public_status(job):
    return dict(implementation=PROTOCOL,execution_name=name(job),result_dir=str(output(job)))
