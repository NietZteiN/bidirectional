"""Versioned bounded C++ grammar accepting signed L/LL integer literals."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
import re
import subprocess
import tempfile

from bidir.domains import py_cpp as original
from bidir.domains._common import base_row
from bidir.domains.py_cpp import content_key, instruction, augmented_hint, INPUTS

NAME = CELL = 'py_cpp_typed_v2'
CELL_OPTS = {}


def build_pairs(cfg):
    splits = original.build_pairs(cfg)
    for rows in splits.values():
        for pair in rows:
            pair.domain = NAME
            pair.pair_id = NAME+'::'+pair.pair_id
    return splits


def cpp_node(text):
    source = original.clean(text)
    match = re.fullmatch(r'(?:int|long\s+long)\s+f\s*\(\s*(?:int|long\s+long)\s+x\s*\)\s*\{\s*return\s+([^;]+);\s*\}', source, re.S)
    if not match:
        raise ValueError('one bounded C++ arithmetic return function required')
    # Only signed decimal L/LL suffixes are accepted. The original bounded
    # interpreter still rejects overflow, arbitrary identifiers and calls.
    expression = re.sub(r'(?<![\w.])(\d+)(?:LL|ll|L|l)(?![\w.])', r'\1', match[1])
    parser = original.CppParser(expression)
    node = parser.expression()
    if parser.peek() is not None:
        raise ValueError('trailing expression tokens')
    original.behavior(node)
    return node, source


@lru_cache(maxsize=4096)
def compiled_behavior(source):
    _, source = cpp_node(source)
    with tempfile.TemporaryDirectory(prefix='bidir_cpp_v2_') as directory:
        path = Path(directory)
        code, binary = path/'main.cpp', path/'program'
        code.write_text('#include <iostream>\n'+source+'\nint main(){for(int x=-16;x<=16;++x) std::cout<<f(x)<<"\\n";}\n')
        try:
            result = subprocess.run(['g++','-std=c++17','-O0',str(code),'-o',str(binary)], capture_output=True, text=True, timeout=15)
            if result.returncode:
                raise RuntimeError('validated C++ failed compilation: '+result.stderr[:300])
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            if result.returncode:
                raise RuntimeError('validated C++ failed execution')
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('compiler/executor timeout is a measurement failure') from exc
        values = tuple(int(value) for value in result.stdout.splitlines())
        if len(values) != len(INPUTS):
            raise RuntimeError('incomplete execution coverage')
        return values


def score_batch(direction, outputs, insts, cfg):
    def one(item):
        output, inst = item
        row = base_row(output, inst['side_a' if direction=='forward' else 'side_b'],
                       inst['side_b' if direction=='forward' else 'side_a'])
        try:
            if direction == 'forward':
                node, source = cpp_node(output)
                got = compiled_behavior(source)
                if got != original.behavior(node):
                    raise RuntimeError('compiler disagrees with bounded interpreter')
            else:
                got = original.behavior(original.python_node(output))
            row['off_target'] = 0
        except (ValueError, SyntaxError, RecursionError, TypeError):
            got = None
            row['off_target'] = 1
        row['behavior_match'] = int(got == tuple(inst['meta']['behavior']))
        row['strict'] = int(row['behavior_match'] and not row['echo'] and not row['empty_output'])
        row['execution_coverage'] = len(INPUTS) if got is not None else 0
        row['criterion'] = 'typed-v2 bounded finite-input equivalence; no echo'
        return row
    allocated = len(os.sched_getaffinity(0))
    workers = int(cfg.get('exec_workers', min(4, allocated)))
    if not 1 <= workers <= allocated:
        raise ValueError('executor workers exceed allocation')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(one, zip(outputs, insts)))
