"""Exploratory Python↔C++ translation of bounded integer arithmetic functions.

Declared input space is x=-16..16. Programs contain a single return expression;
no generated code is executed until an allowlisted AST/grammar has validated it.
Compilation and execution failures caused by harness timeouts raise measurement errors.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile

from bidir.domains._common import base_row
from bidir.schema import PairInstance

NAME = CELL = 'py_cpp'
CELL_OPTS = {}
INPUTS = tuple(range(-16,17))
OPS = {ast.Add:lambda a,b:a+b, ast.Sub:lambda a,b:a-b, ast.Mult:lambda a,b:a*b,
       ast.Lt:lambda a,b:a<b, ast.LtE:lambda a,b:a<=b,
       ast.Gt:lambda a,b:a>b, ast.GtE:lambda a,b:a>=b,
       ast.Eq:lambda a,b:a==b, ast.NotEq:lambda a,b:a!=b}
SYMBOLS = {ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Lt:'<',ast.LtE:'<=',
           ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!='}


def clean(text):
    text=text.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:python|cpp|c\+\+)?\s*\n(.*?)\n```',text,re.S)
        if not match: raise ValueError('invalid code fence')
        text=match[1].strip()
    if len(text)>6000: raise ValueError('program too long')
    return text


def evaluate(node,x,depth=0):
    if depth>25: raise ValueError('expression too deep')
    if isinstance(node,ast.Name) and node.id=='x': value=x
    elif isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<=1000:
        value=node.value
    elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        value=evaluate(node.operand,x,depth+1)*(1 if isinstance(node.op,ast.UAdd) else -1)
    elif isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult):
        value=OPS[type(node.op)](evaluate(node.left,x,depth+1),evaluate(node.right,x,depth+1))
    elif isinstance(node,ast.Compare) and len(node.ops)==1 and type(node.ops[0]) in OPS:
        value=OPS[type(node.ops[0])](evaluate(node.left,x,depth+1),evaluate(node.comparators[0],x,depth+1))
    elif isinstance(node,ast.IfExp):
        condition=evaluate(node.test,x,depth+1)
        # Validate even unreachable branches and bound every intermediate operation.
        yes=evaluate(node.body,x,depth+1); no=evaluate(node.orelse,x,depth+1)
        value=yes if condition else no
    else: raise ValueError('unsupported arithmetic syntax')
    if abs(value)>2**31-1: raise ValueError('32-bit intermediate overflow')
    return value


def behavior(node):
    if len(list(ast.walk(node)))>200: raise ValueError('too many AST nodes')
    return tuple(int(evaluate(node,x)) for x in INPUTS)


def python_node(text):
    tree=ast.parse(clean(text))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('one function required')
    f=tree.body[0]
    if (f.name!='f' or f.decorator_list or len(f.args.args)!=1 or f.args.args[0].arg!='x'
        or f.args.defaults or f.args.kw_defaults or f.args.kwonlyargs or f.args.posonlyargs
        or f.args.vararg or f.args.kwarg or len(f.body)!=1 or not isinstance(f.body[0],ast.Return)):
        raise ValueError('expected def f(x) with one return expression')
    for annotation in (f.returns,f.args.args[0].annotation):
        if annotation is not None and not (isinstance(annotation,ast.Name) and annotation.id=='int'):
            raise ValueError('unsupported annotation')
    behavior(f.body[0].value)
    return f.body[0].value


class CppParser:
    precedence={'==':1,'!=':1,'<':2,'<=':2,'>':2,'>=':2,'+':3,'-':3,'*':4}
    operators={symbol:kind for kind,symbol in SYMBOLS.items()}
    def __init__(self,text):
        self.tokens=[]; pos=0
        while pos<len(text):
            match=re.match(r'\s*(<=|>=|==|!=|[()+*?:<>-]|x|\d+)',text[pos:])
            if not match:
                if text[pos:].strip(): raise ValueError('unsupported C++ token')
                break
            self.tokens.append(match[1]); pos+=match.end()
        if len(self.tokens)>200: raise ValueError('too many tokens')
        self.i=0
    def peek(self): return self.tokens[self.i] if self.i<len(self.tokens) else None
    def take(self):
        token=self.peek()
        if token is None: raise ValueError('unexpected end')
        self.i+=1; return token
    def expression(self,minimum=0,depth=0):
        if depth>25: raise ValueError('expression too deep')
        token=self.take()
        if token=='(':
            left=self.expression(depth=depth+1)
            if self.take()!=')': raise ValueError('missing parenthesis')
        elif token in ('+','-'):
            left=ast.UnaryOp(op=ast.UAdd() if token=='+' else ast.USub(),
                             operand=self.expression(5,depth+1))
        elif token=='x': left=ast.Name(id='x',ctx=ast.Load())
        elif token.isdigit(): left=ast.Constant(value=int(token))
        else: raise ValueError('invalid operand')
        while self.peek() in self.precedence and self.precedence[self.peek()]>=minimum:
            op=self.take(); right=self.expression(self.precedence[op]+1,depth+1)
            kind=self.operators[op]
            left=(ast.BinOp(left=left,op=kind(),right=right) if kind in (ast.Add,ast.Sub,ast.Mult)
                  else ast.Compare(left=left,ops=[kind()],comparators=[right]))
        if minimum==0 and self.peek()=='?':
            self.take(); yes=self.expression(depth=depth+1)
            if self.take()!=':': raise ValueError('missing ternary colon')
            left=ast.IfExp(test=left,body=yes,orelse=self.expression(depth=depth+1))
        return left


def cpp_node(text):
    source=clean(text)
    match=re.fullmatch(r'(?:int|long\s+long)\s+f\s*\(\s*(?:int|long\s+long)\s+x\s*\)\s*\{\s*return\s+([^;]+);\s*\}',source,re.S)
    if not match: raise ValueError('one C++ arithmetic return function required')
    parser=CppParser(match[1]); node=parser.expression()
    if parser.peek() is not None: raise ValueError('trailing expression tokens')
    behavior(node)
    return node,source


def cpp_expression(node):
    if isinstance(node,ast.Constant): return str(node.value)
    if isinstance(node,ast.Name): return 'x'
    if isinstance(node,ast.UnaryOp):
        return '('+('+' if isinstance(node.op,ast.UAdd) else '-')+cpp_expression(node.operand)+')'
    if isinstance(node,ast.BinOp):
        return '('+cpp_expression(node.left)+SYMBOLS[type(node.op)]+cpp_expression(node.right)+')'
    if isinstance(node,ast.Compare):
        return '('+cpp_expression(node.left)+SYMBOLS[type(node.ops[0])]+cpp_expression(node.comparators[0])+')'
    if isinstance(node,ast.IfExp):
        return '('+cpp_expression(node.test)+'?'+cpp_expression(node.body)+':'+cpp_expression(node.orelse)+')'
    raise ValueError('unsupported node')


@lru_cache(maxsize=4096)
def compiled_behavior(source):
    # The caller has already validated the entire source against the bounded grammar.
    _,validated=cpp_node(source)
    with tempfile.TemporaryDirectory(prefix='bidir_cpp_') as directory:
        path=Path(directory); src=path/'main.cpp'; binary=path/'program'
        src.write_text('#include <iostream>\n'+validated+'\nint main(){for(int x=-16;x<=16;++x) std::cout<<f(x)<<"\\n";}\n')
        try:
            result=subprocess.run(['g++','-std=c++17','-O0',str(src),'-o',str(binary)],
                capture_output=True,text=True,timeout=15)
            if result.returncode: raise RuntimeError('validated C++ failed compilation: '+result.stderr[:300])
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=5)
            if result.returncode: raise RuntimeError('validated C++ failed execution')
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('compiler/executor timeout is a measurement failure') from exc
        values=tuple(int(x) for x in result.stdout.splitlines())
        if len(values)!=len(INPUTS): raise RuntimeError('incomplete execution coverage')
        return values


def content_key(pair):
    meta=pair.meta if hasattr(pair,'meta') else pair['meta']
    return tuple(meta['behavior'])


def build_pairs(cfg):
    rng=random.Random(int(cfg.get('seed',17))); seen=set(); out={}
    # Template families are disjoint. This pilot measures template-held-out behavior.
    families={'train':'quadratic','val':'piecewise_linear','test':'cubic'}
    for split in ('train','val','test'):
        rows=[]; attempts=0
        while len(rows)<int(cfg['n_'+split]):
            attempts+=1
            if attempts>int(cfg['n_'+split])*100: raise ValueError('unique function space exhausted')
            a=rng.choice([i for i in range(-12,13) if i]); b=rng.randint(-30,30); c=rng.randint(-50,50)
            family=families[split]
            if family=='quadratic': expr=f'{a} * x * x + {b} * x + {c}'
            elif family=='cubic': expr=f'{a} * x * x * x + {b} * x + {c}'
            else:
                d=rng.choice([i for i in range(-12,13) if i!=a]); e=rng.randint(-50,50); cut=rng.randint(-10,10)
                expr=f'({a} * x + {c}) if x < {cut} else ({d} * x + {e})'
            node=ast.parse(expr,mode='eval').body; values=behavior(node)
            if values in seen: continue
            seen.add(values); i=len(rows)
            rows.append(PairInstance(pair_id=f'py_cpp::{split}::{i}',domain=NAME,subtask=family,
                side_a='def f(x):\n    return '+expr,
                side_b='long long f(long long x) { return '+cpp_expression(node)+'; }',split=split,
                meta={'behavior':list(values),'template_family':family,'determinable':True}))
        out[split]=rows
    return out


def instruction(direction,inst):
    source=inst['side_a' if direction=='forward' else 'side_b']
    target='C++17' if direction=='forward' else 'Python'
    return (f'Translate this function to {target}, preserving behavior for integer x from -16 to 16. '
            'Return only the function f with argument x and one return expression. '
            'Use integers, x, parentheses, +, -, *, comparisons, and conditional expressions only. '
            'No imports, includes, calls, loops, assignments, or explanations. '
            'C++ types may be int or long long; Python may use int annotations.\n'+source)


def augmented_hint(direction):
    return 'Preserve arithmetic and conditional precedence for every allowed input.'


def score_batch(direction,outputs,insts,cfg):
    def one(item):
        output,inst=item; source=inst['side_a' if direction=='forward' else 'side_b']
        target=inst['side_b' if direction=='forward' else 'side_a']
        row=base_row(output,source,target)
        try:
            if direction=='forward':
                node,validated=cpp_node(output); expected=behavior(node)
                got=compiled_behavior(validated)
                if got!=expected: raise RuntimeError('compiler disagrees with bounded interpreter')
            else: got=behavior(python_node(output))
            row['off_target']=0
        except (ValueError,SyntaxError,RecursionError,TypeError):
            got=None; row['off_target']=1
        row['behavior_match']=int(got==tuple(inst['meta']['behavior']))
        row['strict']=int(row['behavior_match'] and not row['echo'] and not row['empty_output'])
        row['execution_coverage']=len(INPUTS) if got is not None else 0
        row['criterion']='equivalence on the entire declared finite input space; no echo'
        return row
    allocated=len(os.sched_getaffinity(0))
    workers=int(cfg.get('exec_workers',min(4,allocated)))
    if workers<1 or workers>allocated: raise ValueError('executor workers exceed allocation')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(one,zip(outputs,insts)))
