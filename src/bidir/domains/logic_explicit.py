"""Post-diagnostic output-contract variant; same four-variable function split."""
from bidir.domains.logic import *
from bidir.domains import logic as original
NAME = CELL = 'logic_explicit'

def build_pairs(cfg):
    splits=original.build_pairs(cfg)
    for rows in splits.values():
        for p in rows:
            p.domain=NAME; p.pair_id=p.pair_id.replace('logic::',NAME+'::',1)
    return splits

def instruction(direction,inst):
    if direction=='forward':
        contract=('OUTPUT FORMAT: exactly 16 characters, each 0 or 1. '
                  'Evaluate the formula once per assignment in the stated row order. '
                  'Your answer must be the resulting bits, not a rewritten formula. ')
    else:
        contract=('OUTPUT FORMAT: one Boolean expression using Python not/and/or. '
                  'Construct a formula true exactly on the assignments marked 1. '
                  'Do not use &, |, ~, a table, or explanations. ')
    return contract+original.instruction(direction,inst)
