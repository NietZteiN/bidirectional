"""Meaningful checks for matched interventions, loss masks and admission proofs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from bidir import mechanism_suite as mx


def worker():
    spec=importlib.util.spec_from_file_location('mechanism_worker',mx.ROOT/'scripts/113_mechanism_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_actual_lora_energy_matches_dense_weight_delta():
    tensors={}
    rng=torch.Generator().manual_seed(7)
    for layer in range(4):
        stem=f'base_model.model.layers.{layer}.q_proj.'
        tensors[stem+'lora_A.weight']=torch.randn(2,7,generator=rng)
        tensors[stem+'lora_B.weight']=torch.randn(5,2,generator=rng)
    cfg=dict(r=2,lora_alpha=8)
    energy=mx.layer_energies(tensors,cfg)
    for l,value in energy.items():
        stem=f'base_model.model.layers.{l}.q_proj.'
        delta=4*tensors[stem+'lora_B.weight'].double()@tensors[stem+'lora_A.weight'].double()
        assert value==pytest.approx(float((delta**2).sum()),rel=1e-12)
    variants=mx.control_coefficients(energy,4)
    for b in range(4):
        expected=sum(energy.values())-energy[b]
        for label,coeff in variants.items():
            if label.startswith(f'band{b}_'):
                assert sum(energy[l]*c*c for l,c in coeff.items())==pytest.approx(expected,rel=1e-12)
        assert variants[f'band{b}_remove'][b]==0
    assert variants==mx.control_coefficients(energy,4)


def test_saved_bfloat_controls_really_match_norm_and_preserve_source(tmp_path):
    from safetensors.torch import save_file,load_file
    tensors={}
    for l in range(4):
        stem=f'base_model.model.layers.{l}.q_proj.'
        tensors[stem+'lora_A.weight']=torch.ones(2,3,dtype=torch.bfloat16)*(l+1)
        tensors[stem+'lora_B.weight']=torch.ones(3,2,dtype=torch.bfloat16)/16
    src=tmp_path/'original';src.mkdir()
    save_file(tensors,str(src/'adapter_model.safetensors'))
    (src/'adapter_config.json').write_text(json.dumps(dict(r=2,lora_alpha=4)))
    original=mx.sha(src/'adapter_model.safetensors')
    w=worker();old=w.resolve_model;w.resolve_model=lambda _:dict(n_layers=4)
    try:systems=w.make_variants(str(src),tmp_path/'result','toy')
    finally:w.resolve_model=old
    assert len(systems)==16
    report=json.loads((tmp_path/'result/variant_manifest.json').read_text())
    for row in report['variants'].values():assert row['norm_relative_error']<=mx.plan()['norm_rtol']
    assert mx.sha(src/'adapter_model.safetensors')==original
    for name,t in load_file(str(Path(systems['band0_remove'])/'adapter_model.safetensors')).items():
        if '.layers.0.' in name and 'lora_B' in name:assert not t.any()


class ToyTokenizer:
    chat_template='test'
    def apply_chat_template(self,msgs,tokenize=False,add_generation_prompt=False):
        return ''.join(m['content'] for m in msgs if m['role']=='system')+'P:'+(msgs[-1]['content']+'!' if msgs[-1]['role']=='assistant' else '')
    def __call__(self,text,add_special_tokens=False):return dict(input_ids=[ord(c)%20 for c in text])


def test_native_completion_mask_rejects_truncation_and_masks_prompt():
    messages=[dict(role='user',content='x')]
    ids,labels=mx.completion_ids(ToyTokenizer(),messages,'yes')
    assert labels[:2]==[-100,-100]
    assert labels[2:]==ids[2:]
    with pytest.raises(ValueError,match='no silent truncation'):
        mx.completion_ids(ToyTokenizer(),messages,'yes',max_len=3)


def test_completion_loss_and_gradient_match_manual_calculation():
    import torch.nn.functional as F
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weights=torch.nn.Parameter(torch.tensor([[.2,-.1,.4],[.5,.3,-.2],[.1,.2,.3]]))
            self.device=torch.device('cpu')
        def forward(self,input_ids,use_cache=False):return SimpleNamespace(logits=self.weights[input_ids])
    model=Model();w=worker();encoded=([0,1,2],[-100,-100,2])
    mean,total,n=w.token_loss(model,encoded,True)
    manual=F.cross_entropy(model.weights[1:2],torch.tensor([2]))
    assert n==1
    assert torch.equal(mean,manual) and torch.equal(total,manual)
    grad=torch.autograd.grad(mean,model.weights)[0]
    assert not grad[0].any() and grad[1].any() and not grad[2].any()
    # First-order reverse interference prediction agrees with a real finite-difference step.
    g=torch.autograd.grad(F.cross_entropy(model.weights[1:2],torch.tensor([0])),model.weights)[0]
    dot=float((g*grad).sum());original=model.weights.detach().clone()
    with torch.no_grad():model.weights.copy_(original-1e-4*g)
    changed=float(w.token_loss(model,encoded)[0])-float(manual.detach())
    assert changed==pytest.approx(-1e-4*dot,abs=2e-7)


def test_registered_panel_contains_null_three_seeds_and_full_smoke_dependencies():
    jobs=mx.items()
    assert len(jobs)==30 and len({j['name'] for j in jobs})==30
    assert sum(j['smoke'] for j in jobs)==6
    for job in jobs:
        if not job['smoke']:assert mx.smoke_for(job) in jobs
    assert any(j.get('role')=='replicated reverse-null control' for j in jobs)


def test_frozen_panel_disjoint_and_outside_historical_slice():
    from bidir.schema import read_pairs
    from bidir.config import DATA_DIR
    if not (mx.DATA/'manifest.json').exists():pytest.skip('requires prepared project data')
    for d in {c['domain'] for c in mx.plan()['cells']}:
        ids={i['pair_id'] for i in mx.selected(d)}
        historical={p.pair_id for p in read_pairs(DATA_DIR/d/'test.jsonl')[:200]}
        assert len(ids)==512 and not ids&historical
        assert not ids & {i['pair_id'] for i in mx.selected(d,'val')}


def test_completion_requires_exact_current_receipt_and_exit_status(tmp_path,monkeypatch):
    job=mx.items()[0];out=tmp_path/'result';out.mkdir();status=tmp_path/'runs/status';status.mkdir(parents=True)
    monkeypatch.setattr(mx,'ROOT',tmp_path);monkeypatch.setattr(mx,'DATA',tmp_path)
    monkeypatch.setattr(mx,'output',lambda _:out);monkeypatch.setattr(mx,'code_hash',lambda:'code')
    monkeypatch.setattr(mx,'selected',lambda _:[]);monkeypatch.setattr(mx,'adapter_inventory',lambda *a,**kw:{})
    (tmp_path/'manifest.json').write_text('{}');(out/'evidence.jsonl').write_text('{}\n')
    doc=dict(valid=True,worker_sha256='code',panel_sha256=mx.sha(tmp_path/'manifest.json'),job=job,
        file_sha256={'evidence.jsonl':mx.sha(out/'evidence.jsonl')},adapters={},finished_utc='2026-10-09T00:00:00Z',slurm_job_id='7')
    (out/'summary.json').write_text(json.dumps(doc))
    assert not mx.completed(job)
    path=status/(job['name']+'.7.json');path.write_text(json.dumps(dict(exit_code=1,finished_utc='2026-10-09T00:01:00Z',job_id='7')))
    assert not mx.completed(job)
    path.write_text(json.dumps(dict(exit_code=0,finished_utc='2026-10-09T00:01:00Z',job_id='7')))
    assert mx.completed(job) # No requirement for a scientifically favorable result.
    (out/'evidence.jsonl').write_text('tampered\n');assert not mx.completed(job)


def test_production_refuses_missing_or_stale_smoke(monkeypatch,tmp_path):
    w=worker();job=next(j for j in mx.items() if not j['smoke'])
    feeder=tmp_path/'runs/feeder';feeder.mkdir(parents=True)
    (feeder/'mechanism.enabled').write_text(json.dumps(dict(worker_sha256='code',panel_sha256='panel')))
    monkeypatch.setattr(w,'ROOT',tmp_path);monkeypatch.setattr(mx,'code_hash',lambda:'code')
    monkeypatch.setattr(mx,'sha',lambda _:'panel');monkeypatch.setattr(mx,'completed',lambda _:False)
    args=SimpleNamespace(domain=job['domain'],model=job['model'],seed=job['seed'],mode=job['mode'],smoke=False)
    with pytest.raises(ValueError,match='matching full smoke'):w.frozen_job(args)


@pytest.mark.parametrize('with_vision',[False,True])
def test_entire_hf_smoke_exercises_real_peft_gradients_and_restores_checkpoints(tmp_path,monkeypatch,with_vision):
    from transformers import GPT2Config,GPT2LMHeadModel,AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    w=worker();cfg=mx.plan()
    tiny_config=GPT2Config(n_layer=1,n_head=2,n_embd=8,n_positions=32,vocab_size=20,
                           resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
    def plain():
        model=GPT2LMHeadModel(tiny_config)
        if with_vision:
            model.vision_tower=torch.nn.Module()
            model.vision_tower.q_proj=torch.nn.Linear(8,8,bias=False)
        return model
    torch.manual_seed(9);initial=plain()
    initial_state={k:v.detach().clone() for k,v in initial.state_dict().items()}
    def fresh(*args,**kwargs):
        model=plain();model.load_state_dict(initial_state);return model
    inventory={}
    for index,arm in enumerate(cfg['systems'][1:]):
        model=get_peft_model(fresh(),LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn','q_proj'] if with_vision else ['c_attn'],task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B' in name:p.normal_(std=.03*(index+1))
        path=tmp_path/arm;model.save_pretrained(path);inventory[arm]=dict(path=str(path))
    source_hashes={a:mx.sha(Path(v['path'])/'adapter_model.safetensors') for a,v in inventory.items()}
    instances=[dict(pair_id=str(i),subtask='same',side_a=str(i+1),side_b=str(i+5)) for i in range(8)]
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',fresh)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:ToyTokenizer())
    monkeypatch.setattr(w,'resolve_model',lambda _:dict(hf_id='tiny'))
    monkeypatch.setattr(w.domains,'get',lambda _:object())
    monkeypatch.setattr(w.prompts,'build_messages',lambda *a,**kw:[dict(role='user',content='x')])
    monkeypatch.setattr(mx,'selected',lambda domain,split='test':instances if split=='test' else instances[4:])
    job=mx.items()[0];out=tmp_path/'output';out.mkdir()
    if with_vision:
        from bidir import mechanism_text_graph as fix
        with pytest.raises(RuntimeError,match='not have been used in the graph'):
            w.hf(job,inventory,out)
        with fix.text_only_peft() as captures:
            result=w.hf(job,inventory,out)
            excluded=[(n,p) for params in captures for n,p in params() if fix.excluded(n)]
            assert excluded and all(not p.requires_grad for n,p in excluded)
    else:result=w.hf(job,inventory,out)
    assert result['candidate_rows']==8*2*4*3
    assert result['gradient_states']==3
    assert result['step_rows']==3*4*2*2*4
    assert all(mx.sha(Path(v['path'])/'adapter_model.safetensors')==source_hashes[a] for a,v in inventory.items())
    states=json.loads((out/'gradients.json').read_text())['states']
    assert all(-1.00001<=s['cosine']<=1.00001 for s in states)


def test_layer_smoke_checks_paired_coverage_and_accepts_negative_controls(tmp_path,monkeypatch):
    w=worker();cfg=dict(mx.plan(),bootstrap_reps=20)
    monkeypatch.setattr(mx,'plan',lambda:cfg)
    insts=[dict(pair_id=str(i),subtask='toy',side_a='a'+str(i),side_b='b'+str(i)) for i in range(16)]
    monkeypatch.setattr(mx,'selected',lambda _:insts)
    class Domain:
        def score_batch(self,direction,outputs,instances,config):
            return [dict(strict=int(o==w.prompts.completion_for(i,direction)),
                         echo=int(o==w.prompts.input_for(i,direction))) for o,i in zip(outputs,instances)]
    monkeypatch.setattr(w.domains,'get',lambda _:Domain())
    monkeypatch.setattr(w,'load_config',lambda _:dict(sampling={}))
    monkeypatch.setattr(w,'resolve_thresholds',lambda cfg,model:cfg)
    monkeypatch.setattr(w.prompts,'build_messages',lambda inst,direction,*a:[dict(role='user',content=w.prompts.completion_for(inst,direction))])
    from bidir import engine
    monkeypatch.setattr(engine,'get_engine',lambda *a,**kw:object())
    def generate(resident,messages,adapters,config):
        # All interventions deliberately equal SFT, demonstrating that null findings survive.
        return [m[0]['content'] if a else 'wrong' for m,a in zip(messages,adapters)],[1]*len(messages)
    monkeypatch.setattr(engine,'generate',generate)
    def variants(*args):
        (tmp_path/'variant_manifest.json').write_text('{}')
        return {f'band{b}_{kind}':f'/variant/{b}/{kind}' for b in range(4)
                for kind in ['remove','scale','random101','random202']}
    monkeypatch.setattr(w,'make_variants',variants)
    job=next(j for j in mx.items() if j['smoke'] and j['mode']=='layers')
    result=w.layers(job,{a:dict(path='/'+a) for a in ['sft','replay','mix5']},tmp_path)
    assert result['generation_rows']==20*2*16
    rows=[json.loads(x) for x in (tmp_path/'trials.jsonl').read_text().splitlines()]
    assert len({(r['system'],r['direction'],r['pair_id']) for r in rows})==len(rows)
    contrasts=json.loads((tmp_path/'contrasts.json').read_text())['comparisons']
    assert all(r['mean']==0 for r in contrasts if r['system'].startswith('band'))
    # A fresh source adapter identical to base is a technical failure, with raw evidence saved.
    monkeypatch.setattr(engine,'generate',lambda e,m,a,c:(['wrong']*len(m),[1]*len(m)))
    with pytest.raises(ValueError,match='effectiveness guard'):
        w.layers(job,{a:dict(path='/'+a) for a in ['sft','replay','mix5']},tmp_path)
    assert (tmp_path/'trials.jsonl').exists()


def test_text_projection_does_not_hide_unconnected_language_parameters():
    from bidir import mechanism_text_graph as fix
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text=torch.nn.Module();self.text.lora_A=torch.nn.Parameter(torch.ones(2))
            self.dead_language=torch.nn.Module();self.dead_language.lora_A=torch.nn.Parameter(torch.ones(2))
            self.vision_tower=torch.nn.Module();self.vision_tower.lora_A=torch.nn.Parameter(torch.ones(2))
        def set_adapter(self,name):
            for p in self.parameters():p.requires_grad_(True)
    model=Toy();fix.project_text_graph(model);model.set_adapter('sft')
    assert not model.vision_tower.lora_A.requires_grad
    active=list(model.named_parameters())
    assert [n for n,p in active]==['text.lora_A','dead_language.lora_A']
    with pytest.raises(RuntimeError,match='not have been used in the graph'):
        torch.autograd.grad(model.text.lora_A.sum(),[p for n,p in active])
    assert not fix.excluded('base_model.model.language_model.layers.0.q_proj.lora_A.sft.weight')
    assert not fix.excluded('base_model.model.vision_tower_like.q_proj.lora_A.sft.weight')


def test_revised_gemma_routing_keeps_original_layer_and_llama_jobs():
    from bidir import mechanism_text_graph as fix
    jobs=mx.items();repaired=[j for j in jobs if fix.applies(j)]
    assert len(repaired)==8 and sum(j['smoke'] for j in repaired)==2
    assert all(fix.name(j).startswith('ev_mx2_hf_') for j in repaired)
    assert all('mechanism_text_graph_v2' in str(fix.output(j)) for j in repaired)
    assert all(not fix.applies(j) for j in jobs if j['mode']=='layers' or j['model']=='llama32-3b')


def test_real_tiny_gemma_text_forward_excludes_only_disconnected_vision_adapters():
    from transformers import Gemma3Config,Gemma3TextConfig,Gemma3ForConditionalGeneration,SiglipVisionConfig
    from peft import LoraConfig,get_peft_model
    from bidir import mechanism_text_graph as fix
    text=Gemma3TextConfig(vocab_size=20,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
        sliding_window=8,layer_types=['sliding_attention','full_attention'])
    vision=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,image_size=4,patch_size=2)
    base=Gemma3ForConditionalGeneration(Gemma3Config(text_config=text.to_dict(),vision_config=vision.to_dict(),
        image_token_index=19,mm_tokens_per_image=4))
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.eval();w=worker();encoded=([2,4,5,6],[-100,-100,5,6])
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    loss,_,_=w.token_loss(model,encoded,True)
    with pytest.raises(RuntimeError,match='not have been used in the graph'):
        torch.autograd.grad(loss,parameters)
    original=fix.project_text_graph(model);model.set_adapter('default')
    assert any(fix.excluded(n) for n,p in original())
    assert all(not p.requires_grad for n,p in original() if fix.excluded(n))
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    loss,_,_=w.token_loss(model,encoded,True)
    gradients=torch.autograd.grad(loss,parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum()>0 for g in gradients)


def test_analysis_routes_revised_gemma_and_keeps_completion_counter_distinct(tmp_path,monkeypatch):
    from bidir import mechanism_text_graph as fix
    spec=importlib.util.spec_from_file_location('repaired_mechanism_analysis',mx.ROOT/'scripts/114_mechanism_analysis.py')
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    monkeypatch.setattr(analysis,'ROOT',tmp_path);(tmp_path/'paper').mkdir()
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}')
    monkeypatch.setattr(mx,'DATA',data);monkeypatch.setattr(fix,'OUT',tmp_path/'results')
    job=next(j for j in mx.items() if fix.applies(j) and not j['smoke'])
    out=fix.output(job);out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(dict(file_sha256={})))
    monkeypatch.setattr(analysis,'is_complete',lambda candidate:candidate==job)
    monkeypatch.setattr(analysis,'likelihood_report',lambda candidate:dict(revised=fix.applies(candidate)))
    monkeypatch.setattr(mx,'completed',lambda candidate:False)
    assert analysis.main()==0
    document=json.loads((tmp_path/'paper/MECHANISM_DIAGNOSTICS.json').read_text())
    assert document['production_complete']==1 and document['production_total']==24
    result=next(i for i in document['items'] if i['name']==job['name'])
    assert result['implementation']==fix.PROTOCOL
    assert json.loads(Path(result['analysis']).read_text())['result']['revised']
