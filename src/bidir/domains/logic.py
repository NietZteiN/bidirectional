"""Four-variable Boolean formulas paired with full truth tables.

A bounded AST interpreter accepts only names, bool constants, not/and/or.
Semantic keys group all equivalent formulas into one split.
"""
import ast
import itertools
import random
import re

from bidir.schema import PairInstance
from bidir.domains._common import base_row

NAME = CELL = 'logic'
CELL_OPTS = {}
VARIABLES = ('a', 'b', 'c', 'd')
ASSIGNMENTS = tuple(itertools.product((False, True), repeat=4))


def truth_bits(expression):
    if len(expression) > 12000:
        raise ValueError('formula too long')
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except MemoryError as exc:
        # CPython reports an exhausted parser stack as MemoryError for some
        # deeply nested model outputs. Reject that syntax while propagating
        # actual allocation failures as measurement failures.
        if 'Parser stack overflowed' in str(exc):
            raise ValueError('formula too complex') from exc
        raise
    nodes = list(ast.walk(tree))
    if len(nodes) > 3000:
        raise ValueError('formula too complex')
    allowed = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.UnaryOp,
               ast.Not, ast.BoolOp, ast.And, ast.Or)
    for node in nodes:
        if not isinstance(node, allowed):
            raise ValueError('unsupported formula syntax')
        if isinstance(node, ast.Name) and node.id not in VARIABLES:
            raise ValueError('unknown variable')
        if isinstance(node, ast.Constant) and type(node.value) is not bool:
            raise ValueError('only Boolean constants allowed')

    def interpret(node, env):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return not interpret(node.operand, env)
        if isinstance(node, ast.BoolOp):
            values = [interpret(v, env) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError('invalid node')

    return ''.join('1' if interpret(tree.body, dict(zip(VARIABLES, row))) else '0'
                   for row in ASSIGNMENTS)


def formula(bits):
    terms = ['(' + ' and '.join(v if x else f'not {v}'
             for v, x in zip(VARIABLES, row)) + ')'
             for row, bit in zip(ASSIGNMENTS, bits) if bit == '1']
    return ' or '.join(terms) if terms else 'False'


def content_key(pair):
    meta = pair.meta if hasattr(pair, 'meta') else pair['meta']
    return meta['truth_bits']


def build_pairs(cfg):
    rng = random.Random(int(cfg.get('seed', 17)))
    sizes = [(s, int(cfg[f'n_{s}'])) for s in ('train', 'val', 'test')]
    # Exclude constants: all remaining functions are sampled uniformly without replacement.
    if sum(n for _, n in sizes) > 65534:
        raise ValueError('four-variable Boolean function space exhausted')
    masks = rng.sample(range(1, 65535), sum(n for _, n in sizes))
    out = {}; pos = 0
    for split, size in sizes:
        rows = []
        for i in range(size):
            bits = format(masks[pos], '016b'); pos += 1
            rows.append(PairInstance(pair_id=f'logic::{split}::{i}', domain=NAME,
                subtask='4var', side_a=formula(bits), side_b=bits, split=split,
                meta={'truth_bits': bits, 'determinable': True}))
        out[split] = rows
    return out


def instruction(direction, inst):
    shared = ('Variables are a,b,c,d. Truth-table row order is 0000,0001,0010,0011,'
              '0100,0101,0110,0111,1000,1001,1010,1011,1100,1101,1110,1111; '
              '0 means False and 1 means True. Formula syntax is Python: not, and, or, '
              'parentheses, a,b,c,d, True, False. Any equivalent formula is accepted. '
              'Return only the requested representation. ')
    if direction == 'forward':
        return shared + 'Convert this formula to 16 output bits: ' + inst['side_a']
    return shared + 'Convert these 16 output bits to a formula: ' + inst['side_b']


def augmented_hint(direction):
    return 'Check all 16 assignments in the specified order. Return only the requested answer.'


def score_batch(direction, outputs, insts, cfg):
    rows = []
    for output, inst in zip(outputs, insts):
        source = inst['side_a' if direction == 'forward' else 'side_b']
        target = inst['side_b' if direction == 'forward' else 'side_a']
        row = base_row(output, source, target)
        try:
            if direction == 'forward':
                got = output.strip()
                if not re.fullmatch('[01]{16}', got):
                    raise ValueError('invalid truth table')
            else:
                got = truth_bits(output)
            row['off_target'] = 0
        except (ValueError, SyntaxError, RecursionError, TypeError):
            got = None; row['off_target'] = 1
        row['semantic_match'] = int(got == inst['meta']['truth_bits'])
        row['strict'] = int(row['semantic_match'] and not row['echo']
                            and not row['empty_output'] and not row['off_target'])
        row['criterion'] = 'all 16 truth values equal; no echo'
        rows.append(row)
    return rows
