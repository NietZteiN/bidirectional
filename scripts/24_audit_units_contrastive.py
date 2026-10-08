#!/usr/bin/env python
"""Audit every units training anchor's exact-quantity negative exclusions."""
import hashlib
import json
from pathlib import Path

from bidir.config import DATA_DIR, RESULTS_DIR
from bidir.contrastive import attach_candidates
from bidir.domains import units
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs


def main():
    paths = {split: DATA_DIR / 'units' / f'{split}.jsonl' for split in ['train', 'val', 'test']}
    splits = {split: read_pairs(path) for split, path in paths.items()}
    all_keys = [units.content_key(p) for rows in splits.values() for p in rows]
    assert len(all_keys) == len(set(all_keys)), 'physical quantities overlap across splits'
    conversion = {dimension: (scale, offset) for dimension, _, _, scale, offset in units.CONVERSIONS}
    for rows in splits.values():
        for row in rows:
            a, _ = units.parse(row.side_a)
            b, _ = units.parse(row.side_b)
            scale, offset = conversion[row.subtask]
            assert a == (b - offset) / scale
            assert units.content_key(row) == f'{row.subtask}:{a}'
    pool = splits['train']
    by_a = {p.side_a: p for p in pool}
    by_b = {p.side_b: p for p in pool}
    for seed in [17, 42, 1234]:
        candidates = attach_candidates([{}] * len(pool), pool, 'units', seed)
        for anchor, candidate in zip(pool, candidates):
            aa, bb = candidate['contrastive_a'], candidate['contrastive_b']
            assert aa[0] == anchor.side_a and bb[0] == anchor.side_b
            assert len(set(aa)) == len(set(bb)) == 5
            for a, b in zip(aa[1:], bb[1:]):
                assert by_a[a].pair_id == by_b[b].pair_id
                assert units.content_key(by_a[a]) != units.content_key(anchor)
    report = dict(passed=True, n_train=len(pool), seeds=[17,42,1234], negatives=4,
                  split_counts={s: len(rows) for s, rows in splits.items()},
                  source_sha256={s: hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()},
                  implementation_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/bidir/contrastive.py').read_bytes()).hexdigest())
    atomic_json(RESULTS_DIR / 'audit/contrastive_units_negatives.json', report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
