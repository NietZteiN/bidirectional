#!/usr/bin/env python
"""Compare supported single-input arithmetic functions against existing code/exec.
Unsupported source is counted explicitly; this is not unrestricted program equivalence.
"""
import ast
import json
import subprocess
from collections import Counter
from bidir.config import DATA_DIR, RESULTS_DIR
from bidir.schema import read_pairs
from bidir.domains.py_cpp import behavior, content_key, python_node


def canonical_node(source):
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('not one function')
    f=tree.body[0]
    if len(f.args.args)!=1: raise ValueError('not one argument')
    argument=f.args.args[0]
    if argument.annotation is not None and not (isinstance(argument.annotation,ast.Name) and argument.annotation.id=='int'):
        raise ValueError('non-integer contract')
    old=argument.arg
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.copy_location(ast.Name(id='x',ctx=node.ctx),node) if node.id==old else node
    f=Rename().visit(f); f.name='f'; f.args.args[0].arg='x'
    f.args.args[0].annotation=None; f.returns=None
    if f.body and isinstance(f.body[0],ast.Expr) and isinstance(f.body[0].value,ast.Constant) and isinstance(f.body[0].value.value,str):
        f.body=f.body[1:]
    return python_node(ast.unparse(ast.fix_missing_locations(tree)))


def main():
    new_keys={s:{content_key(p) for p in read_pairs(DATA_DIR/'py_cpp'/f'{s}.jsonl')}
              for s in ('train','val','test')}
    report={'scope':'bounded single-input integer return expressions; unsupported not proved disjoint',
            'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
            'existing':{}}
    for cell in ('code','exec'):
        for split in ('train','val','test'):
            path=DATA_DIR/cell/f'{split}.jsonl'
            if not path.exists(): continue
            keys=set(); counts=Counter()
            for pair in read_pairs(path):
                sources=[pair.side_a,pair.side_b] if cell=='code' else [pair.meta['code']]
                for source in sources:
                    counts['source_programs']+=1
                    try:
                        values=behavior(canonical_node(source)); keys.add(values);counts['supported']+=1
                    except (ValueError,SyntaxError,RecursionError,TypeError): counts['unsupported']+=1
            overlaps={s:len(keys&new_keys[s]) for s in new_keys}
            report['existing'][f'{cell}/{split}']={**counts,'unique_supported_behaviors':len(keys),'overlap':overlaps}
    report['passes_supported_overlap']=not any(any(r['overlap'].values()) for r in report['existing'].values())
    out=RESULTS_DIR/'audit'/'py_cpp_existing_overlap.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['passes_supported_overlap'] else 1


if __name__=='__main__':
    raise SystemExit(main())
