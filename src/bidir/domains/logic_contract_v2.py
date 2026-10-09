"""Versioned assignment-by-assignment contract; same four-variable semantics."""
from bidir.domains import logic as original
from bidir.domains.logic import score_batch, content_key, augmented_hint
NAME = CELL = 'logic_contract_v2'
CELL_OPTS = {}


def build_pairs(cfg):
    splits = original.build_pairs(cfg)
    for rows in splits.values():
        for pair in rows:
            pair.domain = NAME
            pair.pair_id = NAME+'::'+pair.pair_id
    return splits


def instruction(direction, inst):
    assignments = ', '.join(''.join(str(int(value)) for value in row) for row in original.ASSIGNMENTS)
    contract = ('Evaluate the input independently on every assignment below. '
                'The output is a 16-character string of 0/1 bits, not a formula. '
                if direction == 'forward' else
                'For each output bit equal to 1, make one parenthesized minterm: use each '
                'variable positively if its assignment bit is 1 and use not VARIABLE otherwise. '
                'Join the minterms with or. Output only that Python Boolean expression. ')
    return ('Assignment order (a,b,c,d): '+assignments+'. '+contract
            +'The leftmost assignment bit belongs to a; the rightmost belongs to d.\n'
            +original.instruction(direction, inst))
