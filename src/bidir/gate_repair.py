"""Amendment51 data/proof bookkeeping for isolated gate repairs."""
import hashlib
import json
from pathlib import Path

from bidir import domains
from bidir.config import PROJECT_ROOT as ROOT, DATA_DIR, RESULTS_DIR, load_config
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs, write_jsonl

PLAN = ROOT/'configs/gate_repair.json'
OUT = RESULTS_DIR/'gate_repair_v1'
_CODE_HASH = None


def plan():
    return json.loads(PLAN.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    global _CODE_HASH
    if _CODE_HASH is None:
        files = [Path(__file__), PLAN, ROOT/'scripts/109_gate_repair_worker.py',
                 ROOT/'scripts/110_gate_repair_train.py', ROOT/'src/bidir/train.py',
                 ROOT/'src/bidir/contrastive.py', ROOT/'src/bidir/arms.py',
                 ROOT/'configs/models.yaml', ROOT/'configs/train/_base_lora.yaml',
                 ROOT/'scripts/105_paper_worker.py', ROOT/'scripts/16_audit_criteria.py',
                 ROOT/'src/bidir/prompts.py', ROOT/'src/bidir/engine.py',
                 ROOT/'src/bidir/paper_suite.py', ROOT/'src/bidir/domains/_common.py']
        files += list((ROOT/'src/bidir/domains').glob('*.py'))
        files += list((ROOT/'configs/train').glob('contrastive*.yaml'))
        configs = {cell:load_config(f'domains/{cell}.yaml')
                   for cell in [*plan()['cells'], 'd2t_schema_v2', 'mt_de-en', *plan()['legacy_cells']]}
        _CODE_HASH = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(files))
                                   +json.dumps(configs,sort_keys=True).encode()).hexdigest()
    return _CODE_HASH


def data_hash(cell):
    files = [DATA_DIR/cell/(split+'.jsonl') for split in ['train','val','test']]
    if cell == 'd2t_schema_v2':
        files.append(DATA_DIR/cell/'extractor_schema.json')
    return hashlib.sha256(b''.join(bytes.fromhex(sha(p)) for p in files)).hexdigest()


def prepare():
    for cell in plan()['cells']:
        directory = DATA_DIR/cell
        if (directory/'build_report.json').exists():
            validate_splits(cell)
            continue
        module = domains.get(cell)
        cfg = load_config(f'domains/{cell}.yaml')
        splits = module.build_pairs(cfg)
        from bidir.domains._common import content_key as default_content_key
        key = getattr(module,'content_key',default_content_key)
        keys = {split:{key(p) for p in rows} for split, rows in splits.items()}
        if keys['test'] & (keys['train'] | keys['val']):
            raise ValueError('repair dataset leaks across test: '+cell)
        directory.mkdir(parents=True,exist_ok=True)
        counts = {split:write_jsonl(directory/(split+'.jsonl'),rows) for split, rows in splits.items()}
        atomic_json(directory/'build_report.json',dict(domain=cell,counts=counts,
            protocol=plan()['protocol'],config=cfg,train_test_overlap=0,val_test_overlap=0))
        validate_splits(cell)
    directory = DATA_DIR/'d2t_schema_v2'
    directory.mkdir(parents=True,exist_ok=True)
    from bidir.domains import d2t
    relations = set()
    for split in ['train','val','test']:
        destination = directory/(split+'.jsonl')
        pairs = read_pairs(DATA_DIR/'d2t'/(split+'.jsonl'))
        if split == 'train':
            relations = {predicate for pair in pairs for _,predicate,_ in d2t.parse_triples(pair.side_a)}
        if not destination.exists():
            for pair in pairs:
                pair.domain = 'd2t_schema_v2'
                pair.pair_id = 'd2t_schema_v2::'+pair.pair_id
            write_jsonl(destination,pairs)
    schema_path = directory/'extractor_schema.json'
    if not schema_path.exists():
        atomic_json(schema_path,dict(relations=sorted(relations),source='d2t/train.jsonl',
            source_sha256=sha(DATA_DIR/'d2t/train.jsonl'),uses_test_targets=False))
    if not (directory/'build_report.json').exists():
        atomic_json(directory/'build_report.json',dict(domain='d2t_schema_v2',protocol=plan()['protocol'],
            source_sha256={split:sha(DATA_DIR/'d2t'/(split+'.jsonl')) for split in ['train','val','test']}))
    validate_splits('d2t_schema_v2')


def validate_splits(cell):
    from bidir.domains._common import content_key as default_content_key
    module=domains.get(cell)
    key=getattr(module,'content_key',default_content_key)
    rows={split:read_pairs(DATA_DIR/cell/(split+'.jsonl')) for split in ['train','val','test']}
    keys={split:{key(pair) for pair in pairs} for split,pairs in rows.items()}
    ids={split:{pair.pair_id for pair in pairs} for split,pairs in rows.items()}
    overlaps={f'{a}_x_{b}':dict(content=len(keys[a]&keys[b]),pair_id=len(ids[a]&ids[b]))
              for a,b in [('train','test'),('val','test'),('train','val')]}
    if any(v['pair_id'] for v in overlaps.values()) or any(
        v['content'] for k,v in overlaps.items() if k.endswith('_x_test')):
        raise ValueError('repair split leakage: '+cell+': '+str(overlaps))
    path=DATA_DIR/cell/'build_report.json'
    report=json.loads(path.read_text())
    updated=dict(report,counts={split:len(pairs) for split,pairs in rows.items()},overlaps=overlaps)
    if report!=updated:atomic_json(path,updated)


def status_success(name, summary):
    for path in (ROOT/'runs/status').glob(name+'.*.json'):
        try:
            state = json.loads(path.read_text())
            if state.get('exit_code') == 0 and state.get('finished_utc','') >= summary['finished_utc']:
                return True
        except (ValueError,OSError):
            continue
    return False


def oracle_problems(proof):
    problems = []
    for direction in ['forward', 'reverse']:
        for probe, expected in [('gold',1),('echo',0),('empty',0),('garbage',0)]:
            value = proof.get('directions',{}).get(direction,{}).get(probe,{})
            if value.get('strict') != expected:
                problems.append(f'{direction}/{probe}: expected strict={expected}, got {value}')
    return problems


def gate_passed(cell, model):
    path = OUT/'gates'/model/'summary.json'
    if cell in plan()['legacy_cells']:
        path = OUT/'legacy'/model/'summary.json'
    if cell == 'd2t_schema_v2':
        path = OUT/'d2t_gates'/model/'summary.json'
    try:
        summary = json.loads(path.read_text())
        name = ('gr_legacy_' if cell in plan()['legacy_cells'] else 'gr_gate_')+model
        if cell == 'd2t_schema_v2':name = 'gr_d2t_gate_'+model
        if (summary['worker_sha256'] != code_hash() or not status_success(name,summary)
            or summary['data_sha256'].get(cell) != data_hash(cell)
            or summary.get('trial_sha256') != sha(path.parent/'trials.jsonl')
            or summary.get('oracle_sha256') != sha(path.parent/'oracle.json')):
            return None
        return summary['gates'][cell]['passes_gate']
    except (OSError,ValueError,KeyError):
        return None


def contrastive_smoke_passed(model, arm):
    out = OUT/'contrastive_smokes'/model/arm
    try:
        summary = json.loads((out/'summary.json').read_text())
        training = json.loads((out/'training_summary.json').read_text())
        exposure = training['direction_exposure']
        return (summary['worker_sha256']==code_hash()
                and summary['data_sha256']==data_hash(plan()['contrastive_cell'])
                and summary['negative_pool_sha256']==sha(DATA_DIR/'fmt/train.jsonl')
                and summary['training_sha256']==sha(out/'training_summary.json')
                and summary['steps']==2 and training['steps']==2
                and summary['peak_cuda_bytes']>0 and summary['exit_code']==0
                and status_success(f'gr_cl_smoke_{model}_{arm}',summary)
                and (exposure.get('contrastive_gradient_gate_passed')
                     if arm.startswith('cl_') else exposure.get('extra_ce_passes')==6))
    except (OSError,ValueError,KeyError):
        return False


def contrastive_smokes_ready(cell, model):
    cfg = plan()
    return cell==cfg['contrastive_cell'] and model==cfg['contrastive_model'] and all(
        contrastive_smoke_passed(model,arm) for arm in cfg['contrastive_smoke_arms'])
