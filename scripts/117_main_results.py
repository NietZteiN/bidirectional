#!/usr/bin/env python
"""Render task-by-adaptation rates from audited, separate evaluation passes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bidir.pipeline_state import atomic_json, evaluation_succeeded
from bidir.train import adapter_dir

PAPER = ROOT/'paper'
CORE_ARMS = ['base','sft','mix1','mix5','mix10','mix25','mix50','replay',
             'rev','fwd2x','flip','mixedtask']
CL_ARMS = ['base','sft','mix5','replay','rev','cl_fwd','cl_mix5','cl_shuffled',
           'sft_extra_ce','mix5_extra_ce']
TASKS = {'mt_de-en':'German $\\to$ English', 'mt_en-de':'English $\\to$ German',
         'mt_en-zh':'English $\\to$ Chinese', 'mt_zh-en':'Chinese $\\to$ English',
         'code':'Code transformation', 'sql':'Question $\\to$ SQL',
         'fmt':'Format conversion', 'relation':'Factual relations',
         'algebra':'Polynomial expansion', 'algebra_rev':'Polynomial factorization',
         'exec':'Execution prediction', 'units':'Unit conversion'}
MODELS = {'llama32-3b':'L3B','llama31-8b':'L8B','gemma3-4b':'G4B',
          'gemma3-12b':'G12B','olmo2-1b':'O1B'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_auxiliary(records):
    latest={};timestamps={}
    for c in records:
        if c['seed']!=17 or not set(c['means']) & {'cft','unlikelihood','roundtrip'}:continue
        k=(c['domain'],c['model'],c['seed'])
        timestamp=(json.loads((Path(c['run'])/'summary.json').read_text())['finished_utc'],c['run'])
        if k not in latest or timestamp>timestamps[k]:latest[k]=c;timestamps[k]=timestamp
    return list(latest.values())


def campaign_rates(run, expected_hash, sources):
    """Reject partial/duplicate coverage and ineffective arms, including non-core arms."""
    run = Path(run)
    trial = run/'trials.jsonl'
    if (run/'WITHDRAWN.txt').exists() or sha(trial) != expected_hash:
        raise ValueError('withdrawn or changed trials: '+str(run))
    summary = json.loads((run/'summary.json').read_text())
    arms = set(summary['arms'])
    groups = defaultdict(dict)
    for line in trial.open():
        row = json.loads(line)
        if row.get('strategy') != 'simple':
            continue
        key = (row['system'],row['direction'])
        if row['system'] not in arms or row['direction'] not in ['forward','reverse']:
            raise ValueError('unexpected arm/direction')
        if row['pair_id'] in groups[key] or row['strict'] not in [0,1]:
            raise ValueError('duplicate or nonbinary trial')
        groups[key][row['pair_id']] = row['strict']
    ids = set(groups[('base','forward')])
    if not ids or len(ids) != summary['n_instances']:
        raise ValueError('incomplete same-pass base')
    for arm in arms:
        for direction in ['forward','reverse']:
            if set(groups[(arm,direction)]) != ids:
                raise ValueError('incomplete arm/direction: '+arm)
        if arm != 'base' and summary.get('adapter_effectiveness',{}).get(arm,{}).get('identical_rate',1) >= .999:
            raise ValueError('ineffective/unverified arm: '+arm)
    sources[str(trial)] = expected_hash
    sources[str(run/'summary.json')] = sha(run/'summary.json')
    return {arm:{d:sum(groups[(arm,d)].values())/len(ids)
                for d in ['forward','reverse']} for arm in arms}


def value(means, arm, direction, not_applicable=False):
    if arm not in means:
        return r'\textit{n/a}' if not_applicable else '--'
    number = f"{100*means[arm][direction]:.1f}"
    if arm in ['mix1','mix5']:
        return r'\textcolor{blue!60!black}{\textbf{'+number+'}}'
    if (arm == 'sft' and direction == 'reverse' and means['base'][direction] > 0
            and means[arm][direction] <= .5*means['base'][direction]):
        return r'\textcolor{red!70!black}{\textbf{'+number+'}}'
    return number


def grid(campaigns, arms, labels, with_model=False, dose_groups=False):
    columns = 'll'+('l' if with_model else '')+'r'*len(arms)
    headers = ['Task']+(['Model'] if with_model else [])+['Dir.']+labels
    lines = [r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+columns+'}',
             r'\toprule']
    if dose_groups:
        lines += [r' & & \multicolumn{2}{c}{Reference} & \multicolumn{5}{c}{Reversed-pair share} & \multicolumn{5}{c}{Other adaptations} \\',
                  r'\cmidrule(lr){3-4}\cmidrule(lr){5-9}\cmidrule(lr){10-14}']
    lines += [' & '.join(headers)+r' \\',r'\midrule']
    for i, c in enumerate(campaigns):
        if i:
            lines.append(r'\addlinespace[2pt]')
        for direction,label in [('forward','F'),('reverse','B')]:
            row = [TASKS[c['domain']] if direction == 'forward' else '']
            if with_model:
                row.append(MODELS[c['model']] if direction == 'forward' else '')
            row += [label]+[value(c['means'],a,direction,
                                 a=='cft' and c['domain']!='code') for a in arms]
            lines.append(' & '.join(row)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])


def full_adaptations(core):
    return '\n'.join([r'\begin{table*}[t]',r'\centering\footnotesize',
        grid(core,CORE_ARMS,['Base','SFT',r'1\%',r'5\%',r'10\%',r'25\%',r'50\%',
                            'Replay','Rev.',r'Fwd2$\times$','Flip','Multi'],dose_groups=True),
        r'\caption{Full dose ladder and secondary controls, strict success (\%). '
        r'Llama 3.2-3B, training seed 17. F/B means forward/backward. '
        r'Replay replaces half of task examples with generic instructions; Rev. trains '
        r'backward only; Fwd2$\times$ doubles forward training; Flip trains both directions '
        r'at doubled budget; Multi mixes other tasks. Each task uses the base model '
        r'evaluated on the same examples as every adaptation. Dashes denote proportions '
        r'not yet evaluated. Low-dose relation/execution experiments have fewer than one '
        r'effective batch of distinct reversed pairs; these exploratory additions are '
        r'excluded from the original dose-knee analysis.}\label{tab:full-adaptations}',r'\end{table*}'])


def render(core, contrastive, auxiliary):
    intro = (r'Strict success (\%); F/B denotes forward/backward. '
             r'Blue highlights small reversed-pair doses; red highlights SFT backward '
             r'success at most half the base rate (a descriptive threshold). ')
    primary = '\n'.join([r'\begin{table*}[!t]',r'\centering\footnotesize',
        grid(core,['base','sft','mix1','mix5'],['Base','Forward SFT',r'1\% reversal',r'5\% reversal']),
        r'\caption{'+intro+r'Llama 3.2-3B, training seed 17, all eleven task orientations. '
        r'Dose columns replace that share of forward pairs with their reversals, keeping '
        r'instances and optimizer steps fixed. Preservation starts from the base checkpoint; '
        r'this is not repair after forward-only training. Full doses and controls appear '
        r'in Table~\ref{tab:full-adaptations}, and variation across training runs in '
        r'Table~\ref{tab:main-seed-variation}. Dashes denote proportions not yet evaluated. '
        r'Relation/execution low doses are exploratory small-corpus additions.}'
        r'\label{tab:main-adaptations}',r'\end{table*}'])
    pieces = [r'\begin{table*}[!t]',r'\centering\footnotesize',
              r'\textbf{(a) Contrastive objectives and exposure controls}\par\smallskip',
        grid(contrastive,CL_ARMS,['Base','SFT',r'5\%','Replay','Rev.','CL-F','CL-M',
                               'Shuffle','CE-F+','CE-M+'],True)]
    if auxiliary:
        pieces += [r'\par\medskip\textbf{(b) Other auxiliary objectives}\par\smallskip',
            grid(auxiliary,['base','sft','mix5','cft','unlikelihood','roundtrip'],
                 ['Base','SFT',r'5\%','CFT','Unlikelihood','Round-trip'],True)]
    pieces += [r'\caption{Strict success (\%); F/B means forward/backward. '
        r'Colors follow Table~\ref{tab:main-adaptations}. Contrastive and other auxiliary '
        r'objectives, training seed 17. Each task/model comparison evaluates its own '
        r'base and adaptations on the same examples. '
        r'L/G denotes Llama/Gemma; B denotes billions of parameters. The $5\%$ column is '
        r'ordinary reversed-pair SFT. CL-F/M adds alignment to forward/mixed-direction CE; '
        r'Shuffle randomizes positive pairs; CE-F+/M+ adds CE as an exposure proxy; '
        r'CFT is the reconstructed code-equivalence contrastive fine-tuning recipe. '
        r'Dashes denote objectives not yet evaluated; n/a marks code-equivalence CFT '
        r'outside code. Paired contrasts and variation across training runs appear '
        r'in Appendix~\ref{app:complete-main-comparisons}.}'
        r'\label{tab:main-loss-baselines}',r'\end{table*}']
    return primary, '\n'.join(pieces)


def main():
    snapshot = json.loads((PAPER/'EVIDENCE_SNAPSHOT.json').read_text())
    core_records = [c for c in snapshot['core_cells'] if c['model']=='llama32-3b' and c['seed']==17]
    cl_records = [c for c in snapshot['contrastive_reports'] if c['seed']==17
                  and c['domain'] in ['fmt','mt_de-en','units']]
    aux_records=latest_auxiliary(snapshot['followup_reports'])
    records = core_records+cl_records+aux_records
    fingerprints = {str(Path(c['run'])/'trials.jsonl'):sha(Path(c['run'])/'trials.jsonl')
                    for c in records}
    fingerprints.update({str(Path(c['run'])/'summary.json'):sha(Path(c['run'])/'summary.json') for c in records})
    # Admission and exposure changes must invalidate tables even if trials stay identical.
    admission = {}
    for c in records:
        run = Path(c['run'])
        admission[str(run)] = dict(withdrawn=(run/'WITHDRAWN.txt').exists(),
                                  exposure=c.get('auxiliary_exposure',{}))
        if c in core_records:
            for arm in ['sft','mix1','mix5']:
                path=adapter_dir(c['domain'],c['model'],arm,32,17)/'training_summary.json'
                fingerprints[str(path)]=sha(path) if path.exists() else None
        for method in set(c.get('means',{})) & {'cft','unlikelihood','roundtrip'}:
            adapter = adapter_dir(c['domain'],c['model'],method,32,17)
            admission[str(adapter)] = (adapter/'WITHDRAWN.txt').exists()
            for filename in ['training_summary.json','run_manifest.json']:
                file = adapter/filename
                fingerprints[str(file)] = sha(file) if file.exists() else None
    statuses = {str(p):sha(p) for p in (ROOT/'runs/status').glob('*s17*.json')}
    signature = hashlib.sha256(json.dumps([sha(__file__),fingerprints,admission,statuses],sort_keys=True).encode()).hexdigest()
    report_path = PAPER/'MAIN_RESULTS.json'
    if (report_path.exists() and json.loads(report_path.read_text()).get('signature') == signature
            and all((PAPER/'tables'/f).exists() for f in ['main_adaptations.tex','main_adaptations_full.tex','main_loss_baselines.tex'])):
        print('main result tables unchanged');return 0
    sources = {}; core=[]; contrastive=[]; auxiliary=[]; rejected=[]
    for c in records:
        run = Path(c['run'])
        tag = run.name.rsplit('_s',1)[0]
        method = next(iter(set(c.get('means',{})) & {'cft','unlikelihood','roundtrip'}),None)
        name = (f"at_ev_{method}_{c['domain']}_{c['model']}_s17" if method else
                f"ev_{c['domain']}_{c['model']}_s17"+('' if tag=='small' else '_'+tag))
        names = [name, f"ev_{c['domain']}_{c['model']}_s17_{tag}",
                 f"ev_{tag}_{c['domain']}_{c['model']}_s17"]
        if not any(evaluation_succeeded(run,ROOT/'runs/status',n) for n in names):
            rejected.append(dict(run=str(run),reason='missing successful matching status'));continue
        try:
            means = campaign_rates(run,snapshot['sources'][str(run/'trials.jsonl')],sources)
            for audited_method in set(c.get('means',{})) & {'cft','unlikelihood','roundtrip'}:
                if not c.get('auxiliary_exposure',{}).get(audited_method,{}).get('valid'):
                    raise ValueError('unverified auxiliary exposure')
                adapter = adapter_dir(c['domain'],c['model'],audited_method,32,17)
                if (adapter/'WITHDRAWN.txt').exists():
                    raise ValueError('withdrawn auxiliary adapter')
                training = json.loads((adapter/'training_summary.json').read_text())
                tokens = (training.get('direction_exposure') or {}).get('supervised_tokens_by_direction',{})
                counts = (training.get('lengths') or {}).get('supervised_tokens_by_task',{})
                valid = (tokens.get('conflict_unlikelihood',0)>0 if audited_method=='unlikelihood' else
                         tokens.get('roundtrip_reverse_ce',0)>0 if audited_method=='roundtrip' else
                         counts.get('pos',0)>0 and counts.get('neg',0)>0)
                if not valid:
                    raise ValueError('current training metadata lacks auxiliary exposure')
                for filename in ['training_summary.json','run_manifest.json']:
                    sources[str(adapter/filename)] = sha(adapter/filename)
        except (OSError,ValueError,KeyError) as exc:
            rejected.append(dict(run=str(run),reason=str(exc)));continue
        entry = dict(domain=c['domain'],model=c['model'],seed=17,run=str(run),means=means,
                     trial_sha256=fingerprints[str(run/'trials.jsonl')])
        (auxiliary if method else core if tag in ['small','table_core'] else contrastive).append(entry)
    core.sort(key=lambda c:list(TASKS).index(c['domain']))
    contrastive.sort(key=lambda c:(list(TASKS).index(c['domain']),c['model']))
    auxiliary.sort(key=lambda c:(list(TASKS).index(c['domain']),c['model']))
    if {c['domain'] for c in core} != set(TASKS)-{'units'}:
        raise ValueError('reference panel incomplete; do not select a favorable subset')
    primary,losses = render(core,contrastive,auxiliary)
    for name,content in [('main_adaptations.tex',primary),('main_adaptations_full.tex',full_adaptations(core)),('main_loss_baselines.tex',losses)]:
        (PAPER/'tables'/name).write_text('% GENERATED by scripts/117_main_results.py\n'+content+'\n')
    numbers={}
    for c in core:
        for arm in ['base','sft','mix1','mix5']:
            if arm not in c['means']:continue
            for direction,label in [('forward','forward'),('reverse','backward')]:
                numbers[f"main-{c['domain'].replace('_','-')}-{arm}-{label}"] = dict(
                    value=f"{100*c['means'][arm][direction]:.1f}",run=c['run'],trial_sha256=c['trial_sha256'])
        for arm in ['sft','mix1','mix5']:
            path=adapter_dir(c['domain'],c['model'],arm,32,17)/'training_summary.json'
            if not path.exists():continue
            t=json.loads(path.read_text());sources[str(path)]=sha(path)
            key=(f"main-{c['domain'].replace('_','-')}-training-pairs" if arm=='sft' else
                 f"main-{c['domain'].replace('_','-')}-{arm}-pairs")
            n=(t['lengths']['n_kept'] if arm=='sft' else
               t['balance']['by_task'].get('rev',0)-t['lengths'].get('dropped_by_task',{}).get('rev',0))
            numbers[key]=dict(value=str(n),run=c['run'],trial_sha256=c['trial_sha256'],sources=[str(path)])
    for c in auxiliary+contrastive:
        for arm in c['means']:
            for direction,label in [('forward','forward'),('reverse','backward')]:
                numbers[f"main-objective-{c['domain'].replace('_','-')}-{c['model']}-{arm}-{label}"]=dict(
                    value=f"{100*c['means'][arm][direction]:.1f}",run=c['run'],trial_sha256=c['trial_sha256'])
    atomic_json(report_path,dict(signature=signature,selection='Llama3B seed 17 all11 orientations; all original seed 17 CL and audited auxiliary cells',
        core=core,contrastive=contrastive,auxiliary=auxiliary,rejected=rejected,numbers=numbers,
        source_sha256=sources,generator_sha256=sha(__file__),
        interpretation='Descriptive per-pass rates; no pooled seeds, best-seed selection, cross-pass subtraction or significance from highlighting.'))
    print('Main tables:',len(core),'core;',len(contrastive),'contrastive;',len(auxiliary),'auxiliary;',len(rejected),'rejected')
    return 0


if __name__=='__main__':
    import fcntl
    with (ROOT/'runs/feeder/.main_results.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raise SystemExit(main())
