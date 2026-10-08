#!/usr/bin/env python
"""Validate SVD and peak memory on the largest real projection without saving repair weights."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from bidir.config import RESULTS_DIR
from bidir.mech.fullft_spectral import PROJECTIONS, filter_delta, tensor_index
from bidir.pipeline_state import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tuned', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    left = torch.randn(256, 8, device='cuda')
    right = torch.randn(8, 128, device='cuda')
    exact = left @ right
    filtered, _ = filter_delta(exact, 1.)
    residual = float(torch.linalg.norm(filtered - exact) / torch.linalg.norm(exact))
    if residual > 1e-4:
        raise RuntimeError(f'SVD reconstruction failed: {residual}')
    del left, right, exact, filtered
    bi, ti = tensor_index(args.base), tensor_index(args.tuned)
    candidates = []
    for key, file in ti.items():
        if key.endswith(PROJECTIONS):
            with safe_open(str(file), framework='pt', device='cpu') as reader:
                shape = reader.get_slice(key).get_shape()
            candidates.append((shape[0] * shape[1], key))
    _, key = max(candidates)
    with safe_open(str(bi[key]), framework='pt', device='cpu') as reader:
        before = reader.get_tensor(key).to(device='cuda', dtype=torch.float32)
    with safe_open(str(ti[key]), framework='pt', device='cpu') as reader:
        after = reader.get_tensor(key).to(device='cuda', dtype=torch.float32)
    delta = after - before
    del before, after
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    filtered, spectrum = filter_delta(delta, 1.)
    torch.cuda.synchronize()
    if not torch.isfinite(filtered).all():
        raise RuntimeError('nonfinite filtered matrix')
    report = dict(engineering_smoke=True, exit_code=0,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
        gpu=torch.cuda.get_device_name(), base=str(args.base), tuned=str(args.tuned),
        projection=key, shape=list(delta.shape), synthetic_relative_residual=residual,
        elapsed_s=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        total_cuda_bytes=torch.cuda.get_device_properties(0).total_memory, spectrum=spectrum)
    out = RESULTS_DIR / 'engineering_smokes/fullft_svd' / f'{os.environ.get("SLURM_JOB_ID", "local")}.json'
    atomic_json(out, report)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
