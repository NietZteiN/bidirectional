"""Versioned explicit serialization contract; unchanged structural criterion."""
from bidir.domains import fmt as original
from bidir.domains.fmt import score_batch, augmented_hint
NAME = CELL = 'fmt_contract_v2'
CELL_OPTS = {}


def build_pairs(cfg):
    splits = original.build_pairs(cfg)
    for rows in splits.values():
        for pair in rows:
            pair.domain = NAME
            pair.pair_id = NAME+'::'+pair.pair_id
    return splits


def instruction(direction, inst):
    return (original.instruction(direction, inst)+'\n\n'
            'Serialization contract: XML must be a single complete <root>...</root> document. '
            'Represent array entries by <item> elements. Preserve nested structure, '
            'numeric values and Boolean true/false values. JSON must use actual numbers '
            'and booleans, not quoted numeric or Boolean strings. CSV must include its header. '
            'Output only the requested document, with no surrounding explanation.')
