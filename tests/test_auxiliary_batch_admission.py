import json
from types import SimpleNamespace
from bidir import config, train


def test_auxiliary_batch_admission_requires_valid_faster_profiles(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(config,'RESULTS_DIR',tmp_path/'results')
    monkeypatch.setattr(train,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(total_memory=80*1024**3))
    assert train.validated_auxiliary_microbatch('llama32-3b',64)==1
    statuses=tmp_path/'runs'/'status';statuses.mkdir(parents=True)
    for job,name,arm in [(440706,'cl_batch4_llama','cl_fwd'),(440707,'ce_batch4_llama','sft_extra_ce')]:
        (statuses/f'{name}.{job}.json').write_text(json.dumps({'exit_code':0}))
    for job,arm,micro,runtime in [(440645,'cl_fwd',1,10),(440646,'sft_extra_ce',1,12),
                                 (440706,'cl_fwd',4,5),(440707,'sft_extra_ce',4,6)]:
        out=config.RESULTS_DIR/'engineering_smokes'/f'llama32-3b_{arm}_{job}'
        out.mkdir(parents=True)
        (out/'profile.json').write_text(json.dumps({'exit_code':0,'peak_cuda_bytes':20*1024**3}))
        (out/'run_manifest.json').write_text(json.dumps({'config_resolved':{'train':{'per_device_batch':micro}}}))
        summary={'steps':2,'effective_batch':4,'train_runtime_s':runtime,
                 'direction_exposure':{'contrastive_gradient_gate_passed':True,'extra_ce_passes':6}}
        (out/'training_summary.json').write_text(json.dumps(summary))
    assert train.validated_auxiliary_microbatch('llama32-3b',64)==4
    for job,name,arm,micro,runtime in [
        (440966,'cl_m4_e8_llama','cl_fwd',4,8),
        (440967,'cl_m8_e8_llama','cl_fwd',8,4),
        (440968,'ce_m4_e8_llama','sft_extra_ce',4,10),
        (440969,'ce_m8_e8_llama','sft_extra_ce',8,5)]:
        (statuses/f'{name}.{job}.json').write_text(json.dumps({'exit_code':0}))
        out=config.RESULTS_DIR/'engineering_smokes'/f'llama32-3b_{arm}_{job}'
        out.mkdir(parents=True)
        (out/'profile.json').write_text(json.dumps({'exit_code':0,'peak_cuda_bytes':20*1024**3}))
        (out/'run_manifest.json').write_text(json.dumps({'config_resolved':{'train':{'per_device_batch':micro}}}))
        summary={'steps':2,'effective_batch':8,'train_runtime_s':runtime,
                 'direction_exposure':{'contrastive_gradient_gate_passed':True,'extra_ce_passes':6}}
        (out/'training_summary.json').write_text(json.dumps(summary))
    assert train.validated_auxiliary_microbatch('llama32-3b',64)==8
    slow=config.RESULTS_DIR/'engineering_smokes'/'llama32-3b_sft_extra_ce_440969'/'training_summary.json'
    data=json.loads(slow.read_text());data['train_runtime_s']=11;slow.write_text(json.dumps(data))
    assert train.validated_auxiliary_microbatch('llama32-3b',64)==4
    path=config.RESULTS_DIR/'engineering_smokes'/'llama32-3b_cl_fwd_440706'/'training_summary.json'
    data=json.loads(path.read_text());data['train_runtime_s']=11;path.write_text(json.dumps(data))
    assert train.validated_auxiliary_microbatch('llama32-3b',64)==1
    assert train.validated_auxiliary_microbatch('llama32-3b',63)==1


def test_larger_model_batching_needs_its_own_valid_faster_profiles(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(config,'RESULTS_DIR',tmp_path/'results')
    monkeypatch.setattr(train,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(total_memory=140*1024**3))
    model='gemma3-12b'
    assert train.validated_auxiliary_microbatch(model,64)==1
    configs=tmp_path/'configs';configs.mkdir()
    originals=[dict(job=1,name='cl_base',arm='cl_fwd'),dict(job=2,name='ce_base',arm='sft_extra_ce')]
    profiles=[dict(job=3,name='cl_four',arm='cl_fwd'),dict(job=4,name='ce_four',arm='sft_extra_ce')]
    (configs/'contrastive_scale_smokes.json').write_text(json.dumps({model:originals}))
    (configs/'contrastive_scale_batch4.json').write_text(json.dumps({model:profiles}))
    statuses=tmp_path/'runs/status';statuses.mkdir(parents=True)
    for entry in originals+profiles:
        (statuses/f"{entry['name']}.{entry['job']}.json").write_text(json.dumps({'exit_code':0}))
        out=config.RESULTS_DIR/'engineering_smokes'/f"{model}_{entry['arm']}_{entry['job']}"
        out.mkdir(parents=True)
        micro=1 if entry in originals else 4
        (out/'profile.json').write_text(json.dumps({'exit_code':0,'peak_cuda_bytes':40*1024**3}))
        (out/'run_manifest.json').write_text(json.dumps({'config_resolved':{'train':{'per_device_batch':micro}}}))
        (out/'training_summary.json').write_text(json.dumps(dict(steps=2,effective_batch=4,
            train_runtime_s=10 if micro==1 else 5,
            direction_exposure={'contrastive_gradient_gate_passed':True,'extra_ce_passes':6})))
    assert train.validated_auxiliary_microbatch(model,64)==4
    path=config.RESULTS_DIR/'engineering_smokes'/f'{model}_sft_extra_ce_4/profile.json'
    path.write_text(json.dumps({'exit_code':0,'peak_cuda_bytes':110*1024**3}))
    assert train.validated_auxiliary_microbatch(model,64)==1
    path.write_text(json.dumps({'exit_code':0,'peak_cuda_bytes':40*1024**3}))
    summary_path=path.with_name('training_summary.json')
    summary=json.loads(summary_path.read_text());summary['train_runtime_s']=12
    summary_path.write_text(json.dumps(summary))
    assert train.validated_auxiliary_microbatch(model,64)==1
