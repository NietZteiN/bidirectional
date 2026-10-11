#!/usr/bin/env python
"""Read-only numerical manuscript audit and preflight of the active table pipeline.

Replays stored binary scores independently of the table renderer. It does not regenerate
answers, strengthen correctness predicates, submit jobs, or change experiment signatures.
"""
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bidir.config import RUNS_DIR, load_config, resolve_model
from bidir.pipeline_state import atomic_json
from bidir.train import adapter_dir, _effective_train_knobs

PAPER = ROOT / 'paper'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


LOADED_SOURCE_SHA256 = digest(__file__)


def load_script(name):
    spec = importlib.util.spec_from_file_location('audit_' + name, ROOT / 'scripts' / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def replay(path):
    """Reject duplicate scores and replay simple-strategy means by system and endpoint."""
    groups = defaultdict(dict)
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('strategy', 'simple') != 'simple':
                continue
            endpoint = row.get('endpoint', row.get('direction'))
            key = (row['system'], endpoint)
            pid = row['pair_id']
            if pid in groups[key] or row['strict'] not in (0, 1):
                raise ValueError('duplicate/nonbinary score: ' + str(path))
            groups[key][pid] = row['strict']
    means = defaultdict(dict)
    for (arm, endpoint), scores in groups.items():
        means[arm][endpoint] = sum(scores.values()) / len(scores)
    return dict(means), groups


def require_means(actual, expected, label):
    count = 0
    for arm, endpoints in expected.items():
        for endpoint, value in endpoints.items():
            if abs(actual[arm][endpoint] - value) > 1e-12:
                raise ValueError('incorrect mean: ' + label + '/' + arm + '/' + endpoint)
            count += 1
    return count


def verify_grid(path, cells, arms, raw, labels, with_model=False, model_labels=None):
    rows = []
    direction_column = 2 if with_model else 1
    for line in path.read_text().splitlines():
        if not line.strip().endswith(r'\\'):
            continue
        fields = [f.strip() for f in line.strip()[:-2].split('&')]
        if len(fields) > direction_column and fields[direction_column] in ('F', 'B'):
            rows.append(fields)
    if len(rows) != 2 * len(cells):
        raise ValueError('incorrect table row count: ' + str(path))
    checked = 0
    for i, cell in enumerate(cells):
        for j, (direction, letter) in enumerate([('forward', 'F'), ('reverse', 'B')]):
            fields = rows[2 * i + j]
            if fields[0] != (labels[cell['domain']] if j == 0 else '') or fields[direction_column] != letter:
                raise ValueError('incorrect task/direction label: ' + str(path))
            if with_model and fields[1] != (model_labels[cell['model']] if j == 0 else ''):
                raise ValueError('incorrect model label: ' + str(path))
            if len(fields) != direction_column + 1 + len(arms):
                raise ValueError('incorrect column count: ' + str(path))
            for arm, printed in zip(arms, fields[direction_column + 1:]):
                means = raw[cell['run']]
                if arm not in means:
                    expected = r'\textit{n/a}' if arm == 'cft' and cell['domain'] != 'code' else '--'
                    if printed != expected:
                        raise ValueError('incorrect missing/not-applicable entry: ' + str(path))
                else:
                    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', printed)
                    if not numbers or numbers[-1] != f'{100 * means[arm][direction]:.1f}':
                        raise ValueError('incorrect printed score: ' + str(path) + '/' + arm)
                    checked += 1
    return checked


def paper_audit():
    reports = {name: read(PAPER / (name + '.json')) for name in [
        'MAIN_RESULTS', 'EVIDENCE_SNAPSHOT', 'TASK_METHODS', 'MANUSCRIPT_COMPLETION',
        'MECHANISM_DIAGNOSTICS', 'PUBLICATION_ANALYSIS']}
    expected_hashes = {}
    for name, report in reports.items():
        sources = report.get('source_sha256', report.get('sources', {}))
        for path, sha in sources.items():
            if path in expected_hashes and expected_hashes[path] != sha:
                raise ValueError('reports disagree on source hash: ' + path)
            expected_hashes[path] = sha
    for path, sha in expected_hashes.items():
        if digest(path) != sha:
            raise ValueError('source hash mismatch: ' + path)

    split_count = 0
    from bidir.config import DATA_DIR, resolve_thresholds
    for domain, item in reports['TASK_METHODS']['tasks'].items():
        for split, expected in item['build']['counts'].items():
            path=DATA_DIR/domain/(split+'.jsonl')
            if path.exists():
                with path.open() as stream: actual=sum(1 for line in stream if line.strip())
                if actual != expected: raise ValueError('incorrect corpus count: '+str(path))
                split_count += 1
    for item in reports['TASK_METHODS']['translation_thresholds']:
        actual=resolve_thresholds(load_config('domains/'+item['domain']+'.yaml'),item['model'])['thresholds']
        if actual != item['thresholds']: raise ValueError('incorrect translation threshold')

    raw, groups, mean_count = {}, {}, 0
    def campaign(cell):
        nonlocal mean_count
        run = cell['run']
        if run not in raw:
            raw[run], groups[run] = replay(Path(run) / 'trials.jsonl')
        mean_count += require_means(raw[run], cell['means'], run)
        baseline = set(groups[run][('base', 'forward')])
        for arm, endpoints in cell['means'].items():
            for direction in endpoints:
                if set(groups[run][(arm, direction)]) != baseline:
                    raise ValueError('unequal paired coverage: ' + run)

    main, snapshot = reports['MAIN_RESULTS'], reports['EVIDENCE_SNAPSHOT']
    for cells in [main['core'], main['contrastive'], main['auxiliary'], snapshot['core_cells'],
                  reports['MANUSCRIPT_COMPLETION']['revised_domain_campaigns']]:
        for cell in cells:
            campaign(cell)
    ids = [(c['domain'], c['model'], c['seed']) for c in snapshot['core_cells']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate model/task/training-run inventory')
    severe = sum(raw[c['run']]['sft']['reverse'] <= .5 * raw[c['run']]['base']['reverse']
                 and raw[c['run']]['base']['reverse'] > 0 for c in snapshot['core_cells'])
    if severe != snapshot['resolved_numbers']['core-collapse-cells']:
        raise ValueError('incorrect severe-loss count')

    from bidir import domains
    renderer = load_script('117_main_results')
    for domain, verb in [('algebra', 'Expand'), ('algebra_rev', 'Factor')]:
        if not domains.get(domain).instruction('forward', dict(side_a='x', side_b='y')).startswith(verb):
            raise ValueError('incorrect algebra orientation')
    table_count = verify_grid(PAPER / 'tables/main_adaptations.tex', main['core'],
        ['base', 'sft', 'mix1', 'mix5'], raw, renderer.TASKS)
    table_count += verify_grid(PAPER / 'tables/main_adaptations_full.tex', main['core'],
        renderer.CORE_ARMS, raw, renderer.TASKS)
    # The two objective grids share a file; check each block independently.
    text = (PAPER / 'tables/main_loss_baselines.tex').read_text()
    blocks = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', text, re.S)
    if len(blocks) != 2: raise ValueError('missing objective table block')
    for block, cells, arms in zip(blocks, [main['contrastive'], main['auxiliary']],
                                 [renderer.CL_ARMS, ['base','sft','mix5','cft','unlikelihood','roundtrip']]):
        table_count += verify_grid_text(block, cells, arms, raw, renderer.TASKS, renderer.MODELS)

    expected_numbers = {}
    matched = 0
    for cell in main['core']:
        domain = cell['domain'].replace('_', '-')
        for arm, endpoints in raw[cell['run']].items():
            for direction, value in endpoints.items():
                expected_numbers[f'main-{domain}-{arm}-' + ('backward' if direction == 'reverse' else 'forward')] = f'{100*value:.1f}'
        base_summary = read(adapter_dir(cell['domain'], cell['model'], 'sft', 32, cell['seed']) / 'training_summary.json')
        for arm in ['sft','mix1','mix5','mix10','mix25','mix50']:
            path = adapter_dir(cell['domain'], cell['model'], arm, 32, cell['seed']) / 'training_summary.json'
            if not path.exists(): continue
            training = read(path)
            if training['steps'] != base_summary['steps'] or training['lengths']['n_kept'] != base_summary['lengths']['n_kept']:
                raise ValueError('example/step matching claim fails: ' + str(path))
            matched += 1
            if arm in ['sft','mix1','mix5']:
                key = f'main-{domain}-training-pairs' if arm == 'sft' else f'main-{domain}-{arm}-pairs'
                n = training['lengths']['n_kept'] if arm == 'sft' else training['balance']['by_task'].get('rev',0) - training['lengths'].get('dropped_by_task',{}).get('rev',0)
                expected_numbers[key] = str(n)
    for cell in main['contrastive'] + main['auxiliary']:
        for arm, endpoints in raw[cell['run']].items():
            for direction, value in endpoints.items():
                key = f"main-objective-{cell['domain'].replace('_','-')}-{cell['model']}-{arm}-" + ('backward' if direction == 'reverse' else 'forward')
                expected_numbers[key] = f'{100*value:.1f}'
    for key, item in main['numbers'].items():
        if expected_numbers[key] != item['value']:
            raise ValueError('incorrect numerical macro: ' + key)
    for item in reports['TASK_METHODS']['training_run_variation']:
        cells = [c for c in snapshot['core_cells'] if c['run'] in item['runs']]
        if len(cells) != len(item['seeds']): raise ValueError('incorrect repeat-run count')
        values = [[100*(raw[c['run']][arm][direction]-raw[c['run']]['sft'][direction])
                   for arm in ['mix1','mix5'] for direction in ['forward','reverse']] for c in cells]
        for i, bounds in enumerate(item['ranges']):
            actual = [min(v[i] for v in values), max(v[i] for v in values)]
            if any(abs(a-b)>1e-10 for a,b in zip(actual,bounds)):
                raise ValueError('incorrect training-run range: ' + item['domain'])
        prefix = 'main-repeat-' + item['domain'].replace('_','-')
        expected_numbers[prefix+'-runs'] = str(len(cells))
        for i, name in enumerate(['mix1-forward','mix1-backward','mix5-forward','mix5-backward']):
            for value, suffix in zip([min(v[i] for v in values),max(v[i] for v in values)],['lo','hi']):
                expected_numbers[prefix+'-'+name+'-'+suffix] = f'{value:.1f}'
    for key, item in reports['TASK_METHODS']['numbers'].items():
        if expected_numbers[key] != item['value']:
            raise ValueError('incorrect repeat-run macro: ' + key)
    expected_numbers.update({k:str(v) for k,v in snapshot['resolved_numbers'].items()})
    definitions = re.findall(r'\\csname bidirnum@([^\\]+)\\endcsname\{([^}]+)\}',
                             (PAPER/'numbers.tex').read_text())
    emitted = dict(definitions)
    if len(emitted) != len(definitions): raise ValueError('duplicate numerical definition')
    used = set(re.findall(r'\\NUM\{([a-z0-9-]+)\}',
                         (PAPER/'main.tex').read_text()+(PAPER/'planned_evaluations.tex').read_text()))
    for key in used:
        if emitted.get(key) != expected_numbers.get(key) or key not in expected_numbers:
            raise ValueError('incorrect/missing emitted numerical macro: ' + key)
    for item in reports['TASK_METHODS']['auxiliary_intervals']:
        for contrast in item['contrasts']:
            d=contrast['direction']; means=raw[item['run']]
            delta=100*(means[item['method']][d]-means[item['reference']][d])
            if abs(delta-contrast['delta_pp'])>1e-8 or contrast['confidence']!=.95:
                raise ValueError('incorrect auxiliary contrast: ' + item['run'])
    for item in reports['PUBLICATION_ANALYSIS']['frontiers']:
        expected = {arm:{d:m[arm] for d,m in item['means'].items() if arm in m}
                    for arm in item['means']['forward']}
        campaign(dict(run=item['run'], means=expected))
    return dict(source_hashes=len(expected_hashes), replayed_campaigns=len(raw),
        report_sha256={name:digest(PAPER/(name+'.json')) for name in reports},
        presentation_sha256={name:digest(PAPER/name) for name in [
            'main.tex','planned_evaluations.tex','task_methods.tex','numbers.tex','main.pdf',
            'tables/main_adaptations.tex','tables/main_adaptations_full.tex','tables/main_loss_baselines.tex']},
        corpus_splits=split_count, translation_threshold_rows=len(reports['TASK_METHODS']['translation_thresholds']),
        checked_means=mean_count, main_table_scores=table_count,
        registered_numeric_values=len(main['numbers']), core_combinations=len(ids),
        published_numeric_macros=len(used),
        severe_losses=severe, matched_training_summaries=matched,
        repeat_task_orientations=len(reports['TASK_METHODS']['training_run_variation']),
        pending_campaign_arms=len(read(PAPER/'TABLE_COMPLETION_AUDIT.json')['pending_campaign_arms']))


def verify_grid_text(text, cells, arms, raw, labels, model_labels):
    """Use the same verifier on a table block without touching the source file."""
    class TextPath:
        def read_text(self): return text
        def __str__(self): return 'main_loss_baselines.tex'
    return verify_grid(TextPath(), cells, arms, raw, labels, with_model=True, model_labels=model_labels)


def run_preflight():
    from obtune.provenance import sha256_dir
    worker = load_script('120_table_worker')
    gate = load_script('95_runner')
    queue = subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%R'], text=True)
    jobs = {bits[1]:dict(job=bits[0],state=bits[2],reason=bits[3])
            for line in queue.splitlines() if len(bits:=line.split('|')) == 4}
    checked_adapters, rows = set(), []
    for cell in worker.plan()['cells']:
        if gate.gate_passed(cell['domain'], cell['model']) is not True:
            raise ValueError('missing original capability/scorer check: ' + worker.key(cell))
        knobs = _effective_train_knobs(load_config(worker.recipe(cell)), resolve_model(cell['model']))
        if knobs['per_device_batch'] * knobs['grad_accum'] != 64:
            raise ValueError('invalid effective batch')
        required = set(cell['arms'])-set(cell['new_arms'])-{'base'}
        for arm in set(cell['arms'])-{'base'}:
            valid = worker.adapter_valid(cell,arm)
            if arm in required and not valid:
                raise ValueError('invalid prerequisite: ' + worker.key(cell) + '/' + arm)
            if not valid: continue  # A missing new arm is the scheduled workload.
            directory = adapter_dir(cell['domain'],cell['model'],arm,32,worker.plan()['seed'])
            if directory not in checked_adapters:
                manifest = read(directory/'run_manifest.json')
                training = read(directory/'training_summary.json')
                if not math.isfinite(float(training['train_loss'])):
                    raise ValueError('nonfinite prerequisite training loss: ' + str(directory))
                if sha256_dir(directory/'final') != manifest['adapter']['sha256']:
                    raise ValueError('prerequisite adapter bytes changed: ' + str(directory))
                checked_adapters.add(directory)
        phases = {p:jobs[worker.job_name(cell,p)] for p in ['smoke','train','eval_smoke','eval']
                  if worker.job_name(cell,p) in jobs}
        if len(phases)>1: raise ValueError('multiple live phases for one campaign')
        ready = worker.ready_phase(cell)
        if ready.startswith('blocked'): raise ValueError(ready + ': ' + worker.key(cell))
        completed = [p for p in ['smoke','train','eval_smoke','eval'] if worker.completed(cell,p)]
        rows.append(dict(cell=worker.key(cell), ready=ready, completed=completed, live=phases,
                         microbatch=knobs['per_device_batch'], gradient_accumulation=knobs['grad_accum']))
    if int(os.environ.get('BIDIR_GPU_CAP','0')) != 0:
        raise ValueError('artificial GPU admission cap is enabled')
    modules = ['torch','transformers','trl','peft','vllm','sacrebleu','sympy','yaml']
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing: raise ValueError('missing runtime modules: '+str(missing))
    sql = load_config('domains/sql.yaml')
    sql_budget = sql['engine']['gpu_memory_utilization'] + sql['roundtrip_gpu_memory_utilization']
    if not 0 < sql_budget < 1: raise ValueError('SQL model/parser GPU budget does not fit')
    score_python = Path(os.environ['BIDIR_SCORE_ENV'])/'bin/python'
    subprocess.run([str(score_python),'-c','import importlib.util; assert importlib.util.find_spec("comet")'],check=True,timeout=30)
    return dict(campaigns=rows, verified_prerequisite_adapters=len(checked_adapters),
                runtime_modules=modules, comet_environment=True,
                sql_combined_gpu_fraction=sql_budget,
                scratch_free_gb=round(shutil.disk_usage(RUNS_DIR).free/1024**3,1))


def main():
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), errors={},
                  auditor_sha256=LOADED_SOURCE_SHA256,
                  scope='Replays stored scores and checks runtime prerequisites; does not regenerate model answers or prove future GPU success.')
    with ExitStack() as stack:
        for name in ['.paper_evidence.lock','.paper_synthesis.lock','.main_results.lock','.table_audit.lock']:
            lock=stack.enter_context((ROOT/'runs/feeder'/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        try: report['paper']=paper_audit()
        except Exception as exc: report['errors']['paper']=str(exc)
    try: report['pipeline']=run_preflight()
    except Exception as exc: report['errors']['pipeline']=str(exc)
    atomic_json(ROOT/'runs/feeder/accuracy_preflight_latest.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return int(bool(report['errors']))


if __name__=='__main__':
    raise SystemExit(main())
