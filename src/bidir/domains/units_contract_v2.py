"""Versioned conversion-rule contract; exact rational scoring remains unchanged."""
from bidir.domains import units as original
from bidir.domains.units import score_batch, content_key, augmented_hint
NAME = CELL = 'units_contract_v2'
CELL_OPTS = {}


def build_pairs(cfg):
    splits = original.build_pairs(cfg)
    for rows in splits.values():
        for pair in rows:
            pair.domain = NAME
            pair.pair_id = NAME+'::'+pair.pair_id
    return splits


def instruction(direction, inst):
    return ('Exact rules: 1 m = 100 cm; 1 kg = 1000 g; 1 h = 60 min. '
            'F = C * 9/5 + 32; C = (F - 32) * 5/9. '
            'For a non-terminating result, output an exact integer fraction, not a rounded decimal. '
            'OUTPUT FORMAT: one numeric value, one space, then the requested unit. '
            'No equation, comma grouping, degree symbol or explanation.\n'
            +original.instruction(direction, inst))
