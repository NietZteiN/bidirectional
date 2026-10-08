#!/usr/bin/env python
"""Check saved repair coverage, finite weights and unchanged non-projection tensors on CPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from bidir.config import RESULTS_DIR
from bidir.mech.fullft_spectral import PROJECTIONS, tensor_index
from bidir.pipeline_state import atomic_json


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(checkpoint):
    manifest_path = checkpoint / 'repair_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, info in manifest['source_files'].items():
        stat = Path(name).stat()
        if stat.st_size != info['size'] or stat.st_mtime_ns != info['mtime_ns']:
            raise ValueError(f'source checkpoint changed: {name}')
    base, tuned = Path(manifest['base']), Path(manifest['tuned'])
    bi, ti, ri = tensor_index(base), tensor_index(tuned), tensor_index(checkpoint)
    if bi.keys() != ti.keys() or ti.keys() != ri.keys():
        raise ValueError('base/tuned/repaired tensor coverage differs')
    projections = {key for key in ti if key.endswith(PROJECTIONS)}
    module_names = [m['tensor'] for m in manifest['modules']]
    if len(module_names) != len(set(module_names)) or set(module_names) != projections:
        raise ValueError('manifest projection coverage differs')
    unchanged = 0
    different_from_base = 0
    different_from_tuned = 0
    for key in sorted(ti):
        with safe_open(str(ti[key]), framework='pt', device='cpu') as reader:
            original = reader.get_tensor(key)
        with safe_open(str(ri[key]), framework='pt', device='cpu') as reader:
            repaired = reader.get_tensor(key)
        if original.shape != repaired.shape or original.dtype != repaired.dtype:
            raise ValueError(f'shape/dtype differs: {key}')
        if not torch.isfinite(repaired).all():
            raise ValueError(f'nonfinite saved weights: {key}')
        if key not in projections:
            if not torch.equal(original, repaired):
                raise ValueError(f'non-projection tensor changed: {key}')
            unchanged += 1
        else:
            different_from_tuned += int(not torch.equal(original, repaired))
            with safe_open(str(bi[key]), framework='pt', device='cpu') as reader:
                baseline = reader.get_tensor(key)
            different_from_base += int(not torch.equal(baseline, repaired))
            del baseline
        del original, repaired
    if not different_from_base or not different_from_tuned:
        raise ValueError('saved projection repair is identical to base or tuned checkpoint')
    return dict(passed=True, checkpoint=str(checkpoint), base=str(base), tuned=str(tuned),
        tensor_count=len(ti), projection_count=len(projections),
        non_projection_tensors_exactly_unchanged=unchanged,
        saved_projections_different_from_base=different_from_base,
        saved_projections_different_from_tuned=different_from_tuned,
        finite_weights=True, dtype_shape_coverage_match=True,
        manifest_sha256=sha(manifest_path),
        checkpoint_sha256={str(p): sha(p) for p in sorted(set(ri.values()))},
        implementation_sha256=sha(Path(__file__)),
        finished_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(4, len(__import__('os').sched_getaffinity(0))))
    report = audit(args.checkpoint)
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
