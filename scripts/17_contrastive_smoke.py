#!/usr/bin/env python
"""Real-model engineering smoke only; no experimental adapter is saved."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from bidir.config import resolve_model, DATA_DIR, RESULTS_DIR
from bidir.schema import read_pairs
from bidir.contrastive import masked_mean, symmetric_nce, negative_schedule
from bidir.domains.fmt import _parse


def main():
    torch.manual_seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires CUDA')
    mcfg = resolve_model('llama32-3b')
    tok = AutoTokenizer.from_pretrained(mcfg['hf_id'])
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mcfg['hf_id'], dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=8,lora_alpha=16,
                           target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).cuda()
    model.train()
    model.config.use_cache = False
    pairs = read_pairs(DATA_DIR / 'fmt' / 'train.jsonl')[:20]
    keys = [json.dumps(_parse(p.side_a,p.subtask.split('-')[0]),
                       sort_keys=True,separators=(',',':')) for p in pairs]
    schedule = negative_schedule([p.pair_id for p in pairs],keys,
                                [p.side_a for p in pairs],[p.side_b for p in pairs])
    selected = [0] + schedule[0]
    # No prompts, paired concatenation, generation targets, or special tokens enter encodings.
    def encode(side):
        batch = tok([getattr(pairs[i],side) for i in selected],padding=True,
                    add_special_tokens=False,return_tensors='pt').to('cuda')
        states = model.get_base_model().model(**batch,use_cache=False).last_hidden_state
        emb = masked_mean(states,batch['attention_mask']); emb.retain_grad()
        return emb, int(batch['attention_mask'].sum())
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    a, ta = encode('side_a'); b, tb = encode('side_b')
    loss = symmetric_nce(a[:1],b[:1],b.unsqueeze(0),a.unsqueeze(0))
    loss.backward(); torch.cuda.synchronize()
    ag = float(a.grad.abs().sum()); bg = float(b.grad.abs().sum())
    lora_grad = sum(float(p.grad.float().abs().sum()) for n,p in model.named_parameters()
                    if 'lora_' in n and p.grad is not None)
    assert ag > 0 and bg > 0 and lora_grad > 0 and torch.isfinite(loss)
    report = dict(engineering_smoke=True,experimental_run=False,model='llama32-3b',
        domain='fmt',seed=17,negative_count=4,temperature=0.1,loss=float(loss.detach()),
        side_a_gradient=ag,side_b_gradient=bg,lora_gradient=lora_grad,
        side_a_tokens=ta,side_b_tokens=tb,seconds=time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        independent_side_encoding=True,adapter_saved=False)
    out = RESULTS_DIR / 'audit' / 'contrastive_smoke_llama32-3b.json'
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
