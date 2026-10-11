#!/usr/bin/env python
"""Fill methods and versioned-domain appendices from saved, checked evidence only."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bidir.pipeline_state import atomic_json, evaluation_succeeded
from bidir.train import adapter_dir

spec = importlib.util.spec_from_file_location('main_results', ROOT/'scripts/117_main_results.py')
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)
PAPER = ROOT/'paper'
LABELS = {'fmt_contract_v2':'Formatting contract',
          'units_contract_v2':'Units contract', 'py_cpp_typed_v2':'Typed Python/C++'}
ARMS = ['base', 'sft', 'mix50', 'replay', 'rev']


def repaired_campaigns(snapshot, sources):
    cells = []
    seen = set()
    for c in snapshot['followup_reports']:
        run = Path(c['run'])
        if not run.name.startswith('repair_domain_pilot_s'):
            continue
        key = (c['domain'], c['model'], c['seed'])
        if key in seen:
            raise ValueError('duplicate revised-domain campaign; specify a completion-time selection rule')
        seen.add(key)
        name = f"ev_{c['domain']}_{c['model']}_s{c['seed']}_repair_domain_pilot"
        if not evaluation_succeeded(run, ROOT/'runs/status', name):
            raise ValueError('missing matching successful evaluation: '+str(run))
        for path in (ROOT/'runs/status').glob(name+'.*.json'):
            sources[str(path)] = results.sha(path)
        means = results.campaign_rates(run, snapshot['sources'][str(run/'trials.jsonl')], sources)
        if set(means) != set(ARMS):
            raise ValueError('incomplete registered revised-domain arm set')
        cells.append(dict(domain=c['domain'], model=c['model'], seed=c['seed'],
                          run=str(run), means=means))
    return sorted(cells, key=lambda c:(list(LABELS).index(c['domain']), c['model'], c['seed']))


def revised_table(cells):
    lines = [r'\begin{table*}[t]', r'\centering\footnotesize',
             r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{lll lrrrrr}',
             r'\toprule', r'Versioned task & Model & Seed & Dir. & Base & SFT & $50\%$ & Replay & Rev. \\',
             r'\midrule']
    for c in cells:
        for direction, label in [('forward','F'), ('reverse','B')]:
            row = ([LABELS[c['domain']], results.MODELS[c['model']], str(c['seed'])]
                   if direction == 'forward' else ['','',''])
            row += [label]+[results.value(c['means'], a, direction) for a in ARMS]
            lines.append(' & '.join(row)+r' \\')
        lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}',
        r'\caption{All completed Amendment51 versioned-domain pilot campaigns, strict success (\%). '
        r'Each F/B pair uses its own fresh same-pass base and all registered controls; seeds are '
        r'reported separately. These post-inspection contracts reuse the original semantic corpora '
        r'and are exploratory diagnostics, not independent task replication. Only the $50\%$ '
        r'reversal dose was tested here. Red uses the descriptive SFT backward-loss threshold '
        r'from the main table. Original failures remain reported separately; no cross-version '
        r'subtraction or best-seed selection is used.}',
        r'\label{tab:revised-domain-results}', r'\end{table*}']
    return '\n'.join(lines)+'\n'


def methods(snapshot, sources):
    cells = sorted((c for c in snapshot['core_cells'] if c['model']=='llama32-3b' and c['seed']==17),
                   key=lambda c:list(results.TASKS).index(c['domain']))
    rows = []
    recipes = []
    for c in cells:
        data = ROOT/'data'/c['domain']
        build = json.loads((data/'build_report.json').read_text())
        counts = {}
        for split in ['train', 'val', 'test']:
            path = data/(split+'.jsonl')
            counts[split] = sum(1 for line in path.open() if line.strip())
            if counts[split] != build['counts'][split]:
                raise ValueError('built corpus count changed: '+str(path))
            sources[str(path)] = results.sha(path)
        sources[str(data/'build_report.json')] = results.sha(data/'build_report.json')
        adapter = adapter_dir(c['domain'], c['model'], 'sft', 32, c['seed'])
        manifest = json.loads((adapter/'run_manifest.json').read_text())
        training = json.loads((adapter/'training_summary.json').read_text())
        summary = json.loads((Path(c['run'])/'summary.json').read_text())
        for path in [adapter/'run_manifest.json', adapter/'training_summary.json', Path(c['run'])/'summary.json']:
            sources[str(path)] = results.sha(path)
        recipe = manifest['config_resolved']
        recipes.append(dict(peft=recipe['peft'], train={k:recipe['train'][k] for k in
                            ['lr','epochs','dtype','lr_scheduler_type','warmup_ratio','weight_decay',
                             'max_grad_norm','max_seq_len','save_strategy']}, effective_batch=training['effective_batch']))
        rows.append(dict(domain=c['domain'], built_counts=counts,
                         recorded_training_rows=training['lengths']['n_kept'],
                         evaluated_pairs=summary['n_instances'], steps=training['steps'],
                         per_device_batch=recipe['train']['per_device_batch'],
                         gradient_accumulation=recipe['train']['grad_accum']))
    if any(r != recipes[0] for r in recipes[1:]):
        raise ValueError('reference training recipes differ; report them separately')
    r = recipes[0]; t = r['train']; p = r['peft']
    lines = [r'\paragraph{Recorded reference recipe.}',
        'The main-table Llama3B reference campaigns use the common recorded recipe: '
        f"LoRA rank ${p['r']}$, scaling ${p['alpha']}$, dropout ${p['dropout']:g}$, "
        f"learning rate ${t['lr']:g}$, ${t['epochs']}$ epochs, effective batch ${r['effective_batch']}$, "
        f"and maximum sequence length ${t['max_seq_len']}$ tokens. "
        f"The scheduler is {t['lr_scheduler_type']} with warm-up fraction ${t['warmup_ratio']:g}$, "
        f"weight decay ${t['weight_decay']:g}$ and gradient clipping ${t['max_grad_norm']:g}$. "
        f"Training uses {t['dtype']} and final checkpoints without intermediate checkpoint selection. "
        'LoRA targets query, key, value and output attention projections and the gate, up and down MLP projections. '
        'Allocation-specific microbatches and accumulation retain the recorded effective batch. '
        'Full-weight and robustness variants are separate recipes, not substitutions for this reference.',
        r'\begin{table*}[t]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lrrrrrrr}',r'\toprule',
        r'Task orientation & Train & Val. & Test & Retained & Eval. & Steps & Batch $\times$ accum. \\',r'\midrule']
    for row in rows:
        values = [results.TASKS[row['domain']], *[str(row['built_counts'][s]) for s in ['train','val','test']],
                  str(row['recorded_training_rows']), str(row['evaluated_pairs']), str(row['steps']),
                  f"${row['per_device_batch']}\\times{row['gradient_accumulation']}$"]
        lines.append(' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Built paired-corpus sizes and recorded forward-SFT training/evaluation counts '
        r'for the main-table reference campaigns. Train/validation/test counts are verified against '
        r'the current frozen JSONL files; retained rows, steps and batch shapes come from each '
        r'saved training manifest/summary. These are distinct provenance records: the code '
        r'campaign retains fewer training rows than the current built corpus. Historical '
        r'training data bytes are not inferred from a current file hash. Eval. counts paired '
        r'instances per arm and direction. Opposite task orientations reuse task families; '
        r'they are not independent datasets.}',r'\label{tab:reference-methods}',r'\end{table*}']
    return '\n'.join(lines)+'\n', rows, r


def main():
    snapshot = json.loads((PAPER/'EVIDENCE_SNAPSHOT.json').read_text())
    registration = ROOT/'configs/gate_repair.json'
    sources = {str(registration):results.sha(registration)}
    revised = repaired_campaigns(snapshot, sources)
    text, counts, recipe = methods(snapshot, sources)
    for name, content in [('revised_domain_results.tex', revised_table(revised)), ('methods_details.tex', text)]:
        (PAPER/'tables'/name).write_text('% GENERATED by scripts/118_manuscript_completion.py\n'+content)
    atomic_json(PAPER/'MANUSCRIPT_COMPLETION.json', dict(
        generator_sha256=results.sha(__file__), revised_domain_campaigns=revised,
        reference_methods=counts, reference_recipe=recipe, source_sha256=sources,
        selection='Every completed versioned-domain pilot; all arms and seeds, no effect-sign selection.',
        limitations='No new measurements; reused corpora are not independent new domains; historical data hashes are not reconstructed.'))
    print('Manuscript appendices:', len(revised), 'versioned-domain campaigns;', len(counts), 'reference method records')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
