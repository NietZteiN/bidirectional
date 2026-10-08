"""Frozen paper-completion data, workers and paired reporting utilities.

GPU entrypoints live in scripts/105_paper_worker.py. Importing this module is CPU-only.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import random
import re
from pathlib import Path

from bidir.config import DATA_DIR, PROJECT_ROOT as ROOT, RESULTS_DIR, RUNS_DIR
from bidir.pipeline_state import atomic_json

PLAN = ROOT / 'configs/paper_finish.json'
DATA = DATA_DIR / 'paper_finish'
OUT = RESULTS_DIR / 'paper_finish_v1'
NEUTRAL_SYSTEM = 'You are a helpful assistant. Follow the user instructions carefully.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    return json.loads(PLAN.read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temporary.replace(path)


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


class OverlapIndex:
    """Exact/contained lexical overlap. Never claim this establishes semantic independence."""
    def __init__(self, sources):
        self.texts = []
        self.exact = defaultdict(set)
        self.shingles = defaultdict(set)
        for label, text in sources:
            value = normalized(text)
            if not value:
                continue
            idx = len(self.texts); self.texts.append((label, value))
            self.exact[value].add(label)
            words = value.split()
            for i in range(max(0, len(words) - 7)):
                self.shingles[tuple(words[i:i+8])].add(idx)

    def hits(self, text):
        value = normalized(text); labels = set(self.exact.get(value, set()))
        words = value.split()
        if len(value) < 40 or len(words) < 8:
            return sorted(labels)
        candidates = set()
        for i in range(len(words) - 7):
            candidates.update(self.shingles.get(tuple(words[i:i+8]), set()))
        for idx in candidates:
            label, other = self.texts[idx]
            if value in other or (len(other) >= 40 and other in value):
                labels.add(label)
        return sorted(labels)


def local_sources(splits=('train', 'val')):
    paths = [p for split in splits for p in sorted(DATA_DIR.glob(f'*/{split}.jsonl'))
             if p.parent != DATA]
    sources = []
    for p in paths:
        for r in read_rows(p):
            for side in ('side_a', 'side_b'):
                sources.append((str(p.relative_to(ROOT)), r[side]))
    return sources, {str(p.relative_to(ROOT)): sha(p) for p in paths}


def filter_rows(rows, index, fields):
    accepted, removed, seen = [], [], set()
    for row in rows:
        values = tuple(normalized(row[f]) for f in fields)
        hits = sorted({label for f in fields for label in index.hits(row[f])})
        if not all(values) or values in seen or hits:
            removed.append({'id': row['pair_id'], 'sources': hits,
                            'reason': 'overlap' if hits else 'empty/duplicate'})
            continue
        seen.add(values); accepted.append(row)
    if not accepted:
        raise ValueError('overlap audit retained no evaluation instances')
    return accepted, removed


def prepare_data():
    """Pin external revisions, audit all input corpora, and freeze rows before generation."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    cfg = plan()
    DATA.mkdir(parents=True, exist_ok=True)
    mf = DATA / 'manifest.json'
    if mf.exists():
        manifest = frozen_data()
        print(f"Paper data already frozen: {manifest['n_instances']}", flush=True)
        return manifest
    sources, source_hashes = local_sources()
    index = OverlapIndex(sources)
    api = HfApi(); versions = {}; exclusions = {}; sizes = {}
    for task, spec in cfg['probes'].items():
        revision = api.dataset_info(spec['repo']).sha
        versions[task] = dict(repo=spec['repo'], revision=revision,
                              config=spec.get('config'), split=spec['split'])
        ds = load_dataset(spec['repo'], spec.get('config'), split=spec['split'], revision=revision)
        field = 'prompt' if task == 'ifeval' else 'question'
        rows = [dict(r, pair_id=f'{task}::{i}') for i, r in enumerate(ds)]
        rows, exclusions[task] = filter_rows(rows, index, [field])
        if task == 'gsm8k':
            train = load_dataset(spec['repo'], spec.get('config'), split='train', revision=revision)
            # First five nonoverlapping train questions, fixed before any model outputs.
            test_index = OverlapIndex([('retained_gsm8k_test', r['question']) for r in rows])
            demos = [dict(r) for r in train if not test_index.hits(r['question'])][:5]
            if len(demos) != 5:
                raise ValueError('insufficient disjoint GSM8K demonstrations')
            write_rows(DATA / 'gsm8k_demos.jsonl', demos)
        write_rows(DATA / f'{task}.jsonl', rows); sizes[task] = len(rows)
    transfer = cfg['transfer']; revision = api.dataset_info(transfer['repo']).sha
    versions['opus'] = dict(repo=transfer['repo'], revision=revision,
                            config=transfer['config'], split=transfer['split'])
    ds = load_dataset(transfer['repo'], transfer['config'], split=transfer['split'], revision=revision)
    external = []
    for i, item in enumerate(ds):
        a, b = item['translation']['de'].strip(), item['translation']['en'].strip()
        if all(transfer['min_chars'] <= len(t) <= transfer['max_chars'] for t in (a, b)):
            external.append(dict(pair_id=f'opus::de-en::{i}', domain='mt', subtask='de-en',
                side_a=a, side_b=b, split='test', meta={'src_lang': 'de', 'tgt_lang': 'en'}))
    # Stronger exclusion for independent transfer: include the original held-out sets.
    all_sources, all_hashes = local_sources(('train', 'val', 'test'))
    external, exclusions['opus'] = filter_rows(external, OverlapIndex(all_sources), ['side_a', 'side_b'])
    source_hashes.update(all_hashes)
    write_rows(DATA / 'opus.jsonl', external); sizes['opus'] = len(external)
    for name, rows in synthetic_data(cfg).items():
        write_rows(DATA / f'{name}.jsonl', rows); sizes[name] = len(rows)
    files = {p.name: sha(p) for p in DATA.glob('*.jsonl')}
    manifest = dict(campaign=cfg['campaign'], frozen_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(PLAN), dataset_versions=versions, files=files, n_instances=sizes,
        exclusions=exclusions, training_source_sha256=source_hashes,
        overlap_scope='normalized exact/contained strings; not semantic or pretraining independence',
        builder_sha256=sha(__file__))
    atomic_json(mf, manifest)
    print(json.dumps(dict(n_instances=sizes, removed={k: len(v) for k,v in exclusions.items()})), flush=True)
    return manifest


def synthetic_data(cfg):
    from bidir.domains import units, py_cpp
    import ast
    rng = random.Random(5017); out = {}
    quantity_keys = {r['meta']['quantity_key'] for s in ('train','val','test')
                     for r in read_rows(DATA_DIR/'units'/f'{s}.jsonl')}
    rows = []
    for i in range(cfg['synthetic_transfer']['units_magnitude']):
        dimension, ua, ub, scale, offset = units.CONVERSIONS[i % 4]
        while True:
            # Original generator is limited to |value| <= 10000. Exact cents outside support.
            value = Fraction(rng.choice((-1, 1)) * rng.randint(2000001, 20000000), 100)
            key = f'{dimension}:{value}'
            if key not in quantity_keys:
                quantity_keys.add(key); break
        rows.append(dict(pair_id=f'unit_ood::{i}', domain='units', subtask=dimension, split='test',
            side_a=f'{units.render(value)} {ua}', side_b=f'{units.render(scale*value+offset)} {ub}',
            meta=dict(quantity_key=key,unit_a=ua,unit_b=ub)))
    out['units_magnitude'] = rows
    seen = {tuple(r['meta']['behavior']) for s in ('train','val','test')
            for r in read_rows(DATA_DIR/'py_cpp'/f'{s}.jsonl')}
    rows = []
    for i in range(cfg['synthetic_transfer']['py_cpp_piecewise_quadratic']):
        while True:
            a,b,c,d,e,f = [rng.randint(-15,15) for _ in range(6)]
            cut = rng.randint(-8,8)
            expression = f'({a}*x*x + {b}*x + {c}) if x < {cut} else ({d}*x*x + {e}*x + {f})'
            node = ast.parse(expression, mode='eval').body; values = py_cpp.behavior(node)
            if a and d and values not in seen:
                seen.add(values); break
        cpp = py_cpp.cpp_expression(node)
        rows.append(dict(pair_id=f'py_cpp_ood::{i}', domain='py_cpp', subtask='piecewise_quadratic',
            split='test', side_a=f'def f(x):\n    return {expression}',
            side_b=f'int f(int x) {{ return {cpp}; }}', meta={'behavior': list(values)}))
    out['py_cpp_piecewise_quadratic'] = rows
    return out


def frozen_data():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if manifest['plan_sha256'] != sha(PLAN):
        raise ValueError('paper plan changed after corpus freeze; new campaign required')
    if any(sha(DATA / name) != digest for name,digest in manifest['files'].items()):
        raise ValueError('frozen paper corpus changed')
    if any(sha(ROOT / name) != digest for name,digest in manifest['training_source_sha256'].items()):
        raise ValueError('local corpus changed after overlap audit')
    return manifest


def cell_key(cell):
    return f"{cell['domain']}__{cell['model']}"


def result_dir(mode, cell, seed, smoke=False):
    return OUT / ('smokes' if smoke else mode) / cell_key(cell) / f's{seed}' / mode


def recipe_dir(cell, seed, variant, arm, smoke=False):
    namespace=cell_key(cell)+('__smoke' if smoke else '')
    return RUNS_DIR / 'paper_finish_v1' / namespace / f's{seed}' / f'{variant}__{arm}'


def complete_adapter(path):
    try:
        d = json.loads((path / 'run_manifest.json').read_text())
        return bool(d.get('adapter', {}).get('sha256') and
                    (path / 'final/adapter_model.safetensors').is_file() and
                    not (path / 'WITHDRAWN.txt').exists())
    except (OSError, ValueError):
        return False


def systems_for(mode, cell, seed, smoke=False):
    from bidir.train import adapter_dir
    arms = ['sft', 'replay', cell['mix']]
    if mode in ('probes', 'transfer') or cell['mix'] == 'mix5':
        arms += ['cl_fwd','cl_mix5','sft_extra_ce','mix5_extra_ce']
    if mode == 'recipe':
        arms = ['sft',cell['mix']]
    systems = {'base': None}
    for arm in dict.fromkeys(arms):
        path = adapter_dir(cell['domain'], cell['model'], arm, 32, seed)
        if not complete_adapter(path):
            raise FileNotFoundError(f'incomplete requested adapter: {path}')
        systems[arm] = str(path / 'final')
    if mode == 'recipe':
        variants = ['rank_high'] if smoke else plan()['recipe_variants']
        for variant in variants:
            variant_arms = ['sft'] if smoke else ['sft',cell['mix']]
            for arm in variant_arms:
                path = recipe_dir(cell,seed,variant,arm,smoke)
                if not complete_adapter(path):
                    raise FileNotFoundError(f'incomplete recipe adapter: {path}')
                systems[f'{variant}__{arm}'] = str(path/'final')
    return systems


def validate_coverage(rows, systems):
    if not rows or 'base' not in systems:
        raise ValueError('incomplete paired coverage: empty campaign or missing base')
    groups = defaultdict(set); clusters = {}
    for row in rows:
        key = (row['endpoint'],row['system']); pid = row['pair_id']
        if pid in groups[key]:
            raise ValueError('duplicate paper-suite trial')
        groups[key].add(pid)
        if row['strict'] not in (0,1):
            raise ValueError('nonbinary strict success')
        ck=(row['endpoint'],pid)
        if ck in clusters and clusters[ck]!=row['cluster_id']:
            raise ValueError('cluster mismatch')
        clusters[ck]=row['cluster_id']
    for endpoint in {r['endpoint'] for r in rows}:
        expected=groups[(endpoint,'base')]
        if not expected or any(groups[(endpoint,s)]!=expected for s in systems):
            raise ValueError('incomplete paired coverage')
    return {endpoint:len(groups[(endpoint,'base')]) for endpoint in {r['endpoint'] for r in rows}}


def paired_intervals(rows, systems, n_boot=2000):
    import numpy as np
    reports=[]
    for endpoint in sorted({r['endpoint'] for r in rows}):
        values={s:{r['pair_id']:r for r in rows if r['endpoint']==endpoint and r['system']==s} for s in systems}
        pairs=[(s,'base') for s in systems if s!='base']
        mixes=[s for s in systems if re.fullmatch(r'mix\d+',s)]
        pairs += [(s,'sft') for s in mixes]
        if 'replay' in systems: pairs.append(('replay','sft'))
        if 'cl_mix5' in systems: pairs += [('cl_mix5','mix5'),('cl_mix5','mix5_extra_ce')]
        for variant in plan()['recipe_variants']:
            for arm in ['sft',*mixes]:
                if f'{variant}__{arm}' in systems: pairs.append((f'{variant}__{arm}',arm))
        ids=sorted(values['base']); cluster=defaultdict(list)
        for i,pid in enumerate(ids): cluster[values['base'][pid]['cluster_id']].append(i)
        units=list(cluster.values()); rng=np.random.default_rng(17)
        draws=rng.integers(0,len(units),size=(n_boot,len(units)))
        sizes=np.array([len(u) for u in units]); denominators=sizes[draws].sum(axis=1)
        for a,b in dict.fromkeys(pairs):
            diff=np.array([values[a][pid]['strict']-values[b][pid]['strict'] for pid in ids],dtype=float)
            sums=np.array([diff[u].sum() for u in units]); boots=sums[draws].sum(axis=1)/denominators
            lo,hi=np.quantile(boots,[.025,.975])
            reports.append(dict(endpoint=endpoint,a=a,b=b,delta_pp=100*float(diff.mean()),
                ci_lo=100*float(lo),ci_hi=100*float(hi),n_pairs=len(ids),n_clusters=len(units),
                confidence=.95,draws=n_boot,multiplicity='unadjusted exploratory; no equivalence claim'))
    return reports
