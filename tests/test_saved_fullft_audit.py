import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from bidir.mech.fullft_spectral import repair_checkpoint

spec = importlib.util.spec_from_file_location('saved_fullft_audit',
    Path(__file__).resolve().parents[1] / 'scripts/23_audit_fullft_repair.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def repaired(tmp_path):
    base, tuned, dest = [tmp_path / name for name in ('base', 'tuned', 'repair')]
    base.mkdir(); tuned.mkdir()
    key = 'model.layers.0.self_attn.q_proj.weight'
    before = torch.eye(4) * 3
    save_file({key: before, 'norm.weight': torch.ones(4)}, str(base / 'model.safetensors'))
    save_file({key: before + torch.diag(torch.tensor([20., 1., .5, .1])),
               'norm.weight': torch.ones(4) * 2}, str(tuned / 'model.safetensors'))
    (tuned / 'config.json').write_text('{}')
    repair_checkpoint(base, tuned, dest, 1., 'cpu')
    return base, dest, key


def test_saved_repair_audit_checks_actual_weights(tmp_path):
    _, dest, _ = repaired(tmp_path)
    report = module.audit(dest)
    assert report['passed']
    assert report['saved_projections_different_from_base'] == 1
    assert report['saved_projections_different_from_tuned'] == 1
    assert report['non_projection_tensors_exactly_unchanged'] == 1


def test_changed_non_projection_tensor_is_rejected(tmp_path):
    _, dest, _ = repaired(tmp_path)
    path = dest / 'model.safetensors'
    values = load_file(str(path))
    values['norm.weight'] = torch.zeros(4)
    save_file(values, str(path))
    with pytest.raises(ValueError, match='non-projection tensor changed'):
        module.audit(dest)


def test_manifest_nonzero_rank_cannot_hide_saved_base_weights(tmp_path):
    base, dest, key = repaired(tmp_path)
    path = dest / 'model.safetensors'
    values = load_file(str(path))
    values[key] = load_file(str(base / 'model.safetensors'))[key]
    save_file(values, str(path))
    with pytest.raises(ValueError, match='identical to base or tuned'):
        module.audit(dest)
