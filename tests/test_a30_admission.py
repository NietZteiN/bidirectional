import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('a30_admission',
    Path(__file__).resolve().parents[1] / 'scripts/103_a30_admission.py')
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def test_only_expected_pending_workload_validated_jobs_can_move():
    good = ('447359', 'fullft_projection_repair_mt_de-en_s17', 'PENDING', 'h100,h200')
    assert admission.pending_target(*good, {'svd': True})
    assert not admission.pending_target(*good, {'evaluation': True})
    assert not admission.pending_target(*good, {'svd': False})
    assert not admission.pending_target(good[0], good[1], 'RUNNING', good[3], {'svd': True})
    assert not admission.pending_target(good[0], 'different_job', good[2], good[3], {'svd': True})
    assert not admission.pending_target(good[0], good[1], good[2], 'a30,h100', {'svd': True})


def test_svd_requires_large_projection_and_memory_headroom():
    profile = dict(engineering_smoke=True, exit_code=0, node='g-04-01', gpu='NVIDIA A30',
        synthetic_relative_residual=1e-6, peak_cuda_bytes=10, total_cuda_bytes=24,
        spectrum={'rank_considered': 3072})
    assert admission.valid_svd(profile)
    assert not admission.valid_svd({**profile, 'peak_cuda_bytes': 20})
    assert not admission.valid_svd({**profile, 'spectrum': {'rank_considered': 128}})


def test_partial_or_ineffective_multi_adapter_campaign_does_not_admit():
    summary = dict(domain='fmt', model='gemma3-4b', n_instances=20,
        systems={a: None for a in admission.ARMS},
        adapter_effectiveness={a: dict(n_compared=40, identical_rate=.1)
                               for a in admission.ARMS - {'base'}},
        cells=[dict(system=a, direction=d, n=20, subtask='ALL')
               for a in admission.ARMS for d in ('forward', 'reverse')])
    assert admission.valid_campaign(summary)
    assert not admission.valid_campaign({**summary, 'cells': summary['cells'][:-1]})
    assert not admission.valid_campaign({**summary, 'adapter_effectiveness': {}})
