"""Post-diagnostic complete-function contract; same bounded program semantics."""
from bidir.domains.py_cpp import *
from bidir.domains import py_cpp as original
NAME = CELL = 'py_cpp_explicit'

def build_pairs(cfg):
    splits=original.build_pairs(cfg)
    for rows in splits.values():
        for p in rows:
            p.domain=NAME; p.pair_id=NAME+'::'+p.pair_id
    return splits

def instruction(direction,inst):
    contract=('OUTPUT FORMAT: a complete C++ function long long f(long long x) '
              '{ return EXPRESSION; }. Include the signature, braces, and semicolon. '
              if direction=='forward' else
              'OUTPUT FORMAT: a complete Python function def f(x): followed by an '
              'indented return EXPRESSION. Include the signature and balanced parentheses. ')
    return contract+original.instruction(direction,inst)
