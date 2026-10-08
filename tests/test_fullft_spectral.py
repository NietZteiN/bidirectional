import json
import pytest
import torch
from safetensors.torch import save_file,load_file
from bidir.mech.fullft_spectral import repair_checkpoint, filter_delta


def test_full_checkpoint_repair_uses_weight_difference(tmp_path):
    base,tuned,out=[tmp_path/x for x in ['base','tuned','out']]
    base.mkdir();tuned.mkdir()
    key='model.layers.0.self_attn.q_proj.weight'
    before=torch.eye(4)*3
    delta=torch.diag(torch.tensor([20.,1.,.5,.1]))
    save_file({key:before,'norm.weight':torch.ones(4)},str(base/'model.safetensors'))
    save_file({key:before+delta,'norm.weight':torch.ones(4)*2},str(tuned/'model.safetensors'))
    (tuned/'config.json').write_text('{}')
    report=repair_checkpoint(base,tuned,out,1.,'cpu')
    weights=load_file(str(out/'model.safetensors'))
    assert torch.allclose(weights[key],before+torch.diag(torch.tensor([20.,0.,0.,0.])))
    assert torch.equal(weights['norm.weight'],torch.ones(4)*2)
    assert report['modules'][0]['rank_kept']==1
    assert json.loads((out/'repair_manifest.json').read_text())['delta_is_lowrank_by_construction'] is False
    with pytest.raises(FileExistsError): repair_checkpoint(base,tuned,out,1.,'cpu')


def test_zero_update_and_nonfinite():
    result,report=filter_delta(torch.zeros(3,3),1)
    assert report['unchanged'] and result.count_nonzero()==0
    with pytest.raises(ValueError):filter_delta(torch.full((3,3),float('nan')),1)
