"""Versioned translation scorer requiring not-echo in BOTH directions."""
from bidir.domains import mt as original
from bidir.domains.mt import instruction, augmented_hint


def score_batch(direction, outputs, insts, cfg):
    rows = original.score_batch(direction, outputs, insts, cfg)
    for row in rows:
        row['strict'] = int(row['strict'] and not row['echo'])
        row['criterion'] += ' & not-echo-both-directions-v2'
    return rows
