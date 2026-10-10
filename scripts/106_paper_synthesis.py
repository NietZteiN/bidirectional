#!/usr/bin/env python
"""Cache audited dose frontiers, measured costs and completed publication evaluations."""
from collections import defaultdict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import paper_suite as suite
from bidir.config import RESULTS_DIR,RUNS_DIR
from bidir.pipeline_state import atomic_json,evaluation_succeeded
from bidir.train import adapter_dir
PAPER=ROOT/'paper'
LOADED_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PANEL=[('mt_de-en','llama32-3b'),('fmt','gemma3-4b'),('units','gemma3-12b')]
ARMS=['base','sft','replay','mix1','mix5','mix10','mix25','mix50','rev','cl_fwd','cl_mix5','sft_extra_ce','mix5_extra_ce']


def source_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publication_cell(summary):
    mode=summary['mode']
    key=('probe' if mode=='probes' else mode)+'_panel'
    return next(c for c in suite.plan()[key]
                if c['domain']==summary['domain'] and c['model']==summary['model'])


def publication_seed_ranges(results, mode):
    """Summarize within-campaign differences; seed ranges are not confidence intervals."""
    groups = defaultdict(dict)
    for result in results:
        summary = result['summary']
        if summary['mode'] != mode or not summary.get('eligibility_passed'):
            continue
        mix = publication_cell(summary)['mix']
        if mode == 'recipe':
            settings = [('original', 'sft', mix)] + [
                (variant, variant+'__sft', variant+'__'+mix)
                for variant in suite.plan()['recipe_variants']]
        else:
            settings = [('original', 'sft', mix)]
        prefixes = sorted({endpoint.rsplit('/', 1)[0] for endpoint in summary['means']})
        for prefix in prefixes:
            forward = summary['means'][prefix+'/forward']
            reverse = summary['means'][prefix+'/reverse']
            for setting, sft, mixed in settings:
                key = (summary['domain'], summary['model'], prefix, setting)
                if summary['seed'] in groups[key]:
                    raise ValueError('duplicate publication campaign for '+str(key))
                groups[key][summary['seed']] = [
                    100*(forward[sft]-forward['base']),
                    100*(reverse[sft]-reverse['base']),
                    100*(forward[mixed]-forward[sft]),
                    100*(reverse[mixed]-reverse[sft])]
    return [dict(domain=key[0], model=key[1], endpoint=key[2], setting=key[3],
                 seeds=sorted(values), ranges=[(min(v[i] for v in values.values()),
                                               max(v[i] for v in values.values()))
                                              for i in range(4)])
            for key, values in sorted(groups.items())]


def publication_range_table(results, mode):
    rows = publication_seed_ranges(results, mode)
    labels = {'llama32-3b': 'L3B', 'llama31-8b': 'L8B',
              'gemma3-4b': 'G4B', 'gemma3-12b': 'G12B'}
    rendered = []
    for row in rows:
        label = row['setting'] if mode == 'recipe' else row['endpoint']
        rendered.append(' & '.join([
            row['domain'].replace('_', r'\_')+'/'+labels[row['model']],
            label.replace('_', r'\_'), str(len(row['seeds'])),
            *[f'[{lo:+.1f}, {hi:+.1f}]' for lo, hi in row['ranges']]])+r' \\')
    return ('\\begin{table*}[t]\n\\centering\\footnotesize\n'
            '\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllrrrr}\n\\toprule\n'
            'Cell & Setting & Seeds & $\\Delta F_{S-B}$ & $\\Delta R_{S-B}$ & '
            '$\\Delta F_{M-S}$ & $\\Delta R_{M-S}$ '+r'\\'+'\n\\midrule\n'
            +'\n'.join(rendered)+'\n\\bottomrule\n\\end{tabular}}\n'
            +'\\caption{Completed '+mode+' panel: ranges across training seeds of '
            'within-campaign strict-success changes (percentage points). '
            '$S$, $B$, and $M$ denote forward SFT, same-pass base, and the registered '
            'directional mixture. Recipe rows compare matched SFT/mixture variants. '
            'Mixture doses are 5\\% for translation/format and 50\\% for units/Python--C++; '
            'L/G identify Llama/Gemma and B denotes billions of parameters. '
            'Ranges are descriptive seed variation, not confidence intervals; '
            'per-campaign paired intervals and every control remain in the analysis artifact.}\n'
            +'\\label{tab:publication-'+mode+'-ranges}\n\\end{table*}\n')


def dataset(run):
    rows=[r for r in suite.read_rows(run/'trials.jsonl') if r.get('strategy')=='simple' and r['system'] in ARMS]
    for r in rows:r['endpoint']=r['direction']
    return rows


def publication_probe_table(results):
    """Keep strict endpoints separate and retain adverse training-seed outcomes."""
    groups=defaultdict(dict)
    comparisons=[('sft','base'),('mix5','sft'),('replay','sft')]
    labels={'llama32-3b':'Llama3B','gemma3-4b':'Gemma4B','gemma3-12b':'Gemma12B'}
    for result in results:
        summary=result['summary']
        if summary['mode']!='probes' or not summary.get('eligibility_passed'):continue
        for endpoint in ['gsm8k','ifeval']:
            contrasts={(c['a'],c['b']):c['delta_pp'] for c in result['contrasts']['contrasts']
                       if c['endpoint']==endpoint}
            key=(summary['domain'],summary['model'],endpoint)
            if summary['seed'] in groups[key]:raise ValueError('duplicate probe campaign')
            groups[key][summary['seed']]=[contrasts[pair] for pair in comparisons]
    rows=[]
    for (domain,model,endpoint),seeds in sorted(groups.items()):
        intervals=[f"[{min(v[i] for v in seeds.values()):+.1f}, {max(v[i] for v in seeds.values()):+.1f}]"
                   for i in range(len(comparisons))]
        rows.append(' & '.join([domain.replace('_',r'\_')+'/'+labels[model],endpoint,str(len(seeds)),*intervals])+r' \\')
    return ('\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llllll}\n\\toprule\n'
        +r'Cell & Endpoint & Seeds & SFT--base & mix5--SFT & Replay--SFT \\'+'\n\\midrule\n'
        +'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n'
        +'\\caption{Strict general-ability endpoint changes: descriptive minimum/maximum across '
        'all completed training seeds, in percentage points. Each difference uses its own fresh '
        'campaign baseline. These ranges are not confidence intervals. Per-seed paired intervals, '
        'all objective controls and overlap exclusions remain in the evidence artifact; endpoints '
        'are not pooled.}\\label{tab:publication-probe-ranges}\n\\end{table*}\n')


def frontier_run(run,summary):
    import numpy as np
    rows=dataset(run);systems={r['system']:None for r in rows}
    coverage=suite.validate_coverage(rows,systems)
    if not coverage:return None
    means={d:{s:sum(r['strict'] for r in rows if r['system']==s and r['direction']==d)/n for s in systems}
           for d,n in coverage.items()}
    contrasts=suite.paired_intervals(rows,systems)
    rate_intervals={}
    for direction in coverage:
        groups={s:{r['pair_id']:r for r in rows if r['system']==s and r['direction']==direction} for s in systems}
        ids=sorted(groups['base']);clusters=defaultdict(list)
        for i,pid in enumerate(ids):clusters[groups['base'][pid]['cluster_id']].append(i)
        units=list(clusters.values());rng=np.random.default_rng(17)
        draws=rng.integers(0,len(units),size=(2000,len(units)))
        sizes=np.array([len(u) for u in units]);den=sizes[draws].sum(axis=1)
        rate_intervals[direction]={}
        for s in systems:
            vals=np.array([groups[s][pid]['strict'] for pid in ids]);sums=np.array([vals[u].sum() for u in units])
            lo,hi=np.quantile(sums[draws].sum(axis=1)/den,[.025,.975])
            rate_intervals[direction][s]={'lo':float(lo),'hi':float(hi),'confidence':.95}
    costs={}
    for arm in systems:
        if arm=='base':continue
        path=adapter_dir(summary['domain'],summary['model'],arm,32,summary['seed'])
        training=path/'training_summary.json'
        if not training.exists():costs[arm]={'available':False};continue
        data=json.loads(training.read_text())
        costs[arm]=dict(available=True,training_summary_sha256=source_hash(training),
            train_runtime_s=data.get('train_runtime_s'),optimizer_steps=data.get('steps'),
            effective_batch=data.get('effective_batch'),lengths=data.get('lengths'),
            direction_exposure=data.get('direction_exposure'),manifest_sha256=source_hash(path/'run_manifest.json'))
    return dict(domain=summary['domain'],model=summary['model'],seed=summary['seed'],run=str(run),
        trial_sha256=source_hash(run/'trials.jsonl'),means=means,rate_intervals=rate_intervals,contrasts=contrasts,costs=costs,
        campaign=run.name,evaluation_layout=summary.get('evaluation_layout','single_engine_pass'))


def table(rows,caption,label):
    return ('\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llllll}\n\\toprule\n'
        +'Task & Model & Seed & Arm & Forward (\\%) & Reverse (\\%) '+r'\\'+'\n\\midrule\n'
        +'\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
        +'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{tab:'+label+'}\n\\end{table*}\n')


def plot(frontiers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours={'base':'black','sft':'#c43c39','replay':'#777777','mix1':'#d9e6ca','mix5':'#8abd75',
             'mix10':'#509c58','mix25':'#2f7c43','mix50':'#176735','rev':'#8e61aa',
             'cl_fwd':'#dd9221','cl_mix5':'#146db0','sft_extra_ce':'#deb770','mix5_extra_ce':'#6baed6'}
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained')
    names={'mt_de-en':'German to English','fmt':'Format conversion','units':'Unit conversion'}
    labels={'llama32-3b':'Llama 3B','gemma3-4b':'Gemma 4B','gemma3-12b':'Gemma 12B'}
    shown={}
    for ax,(domain,model) in zip(axes,PANEL):
        for f in frontiers:
            if (f['domain'],f['model'])!=(domain,model):continue
            marker='s' if f['campaign'].startswith('contrastive') else 'o'
            alpha=1 if f['seed']==17 else .4
            for arm in ARMS:
                if arm not in f['means']['forward']:continue
                artist=ax.scatter(100*f['means']['forward'][arm],100*f['means']['reverse'][arm],
                    color=colours[arm],marker=marker,alpha=alpha,s=25)
                shown.setdefault(arm,artist)
                if f['seed']==17:
                    x,y=100*f['means']['forward'][arm],100*f['means']['reverse'][arm]
                    ci=f['rate_intervals']
                    ax.errorbar(x,y,xerr=[[max(0,x-100*ci['forward'][arm]['lo'])],
                                         [max(0,100*ci['forward'][arm]['hi']-x)]],
                        yerr=[[max(0,y-100*ci['reverse'][arm]['lo'])],
                              [max(0,100*ci['reverse'][arm]['hi']-y)]],
                        color=colours[arm],alpha=.6,linewidth=.6,fmt='none')
        ax.set(xlim=(-3,103),ylim=(-3,103),title=f'{names[domain]}\n{labels[model]}',
               xlabel='Forward strict success (%)')
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9);ax.xaxis.label.set_fontsize(8)
    axes[0].set_ylabel('Reverse strict success (%)',fontsize=8)
    fig.legend([shown[a] for a in ARMS if a in shown],[a.replace('_',' ') for a in ARMS if a in shown],
               loc='outside lower center',ncol=5,fontsize=7,frameon=False)
    folder=PAPER/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'preservation_frontier.pdf');fig.savefig(folder/'preservation_frontier.png',dpi=200)
    plt.close(fig)


def main():
    mf=suite.DATA/'manifest.json'
    if not mf.exists():return 0
    summaries=[p for p in RESULTS_DIR.glob('*/*/*/*/summary.json')
               if (p.parent.parent.parent.name,p.parent.parent.name) in PANEL and
               (p.parent.name.startswith('small_s') or p.parent.name.startswith('contrastive_pilot_s')
                or p.parent.name.startswith('generality_scale_pilot_s'))]
    publication=list(suite.OUT.glob('*/*/*/*/summary.json'))
    # Only source artifacts, not generated outputs, enter the signature.
    inputs=[Path(__file__),ROOT/'src/bidir/paper_suite.py',suite.PLAN,mf]
    inputs += [q for p in summaries+publication for q in [p,p.parent/'trials.jsonl',p.parent/'contrasts.json'] if q.exists()]
    inputs += [p for domain,model in PANEL for p in (RUNS_DIR/'adapters'/domain/model).glob('*/training_summary.json')]
    signature=hashlib.sha256(json.dumps(dict(loaded_generator=LOADED_SOURCE_SHA256,
        files=sorted((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in inputs))).encode()).hexdigest()
    out=PAPER/'PUBLICATION_ANALYSIS.json'
    old=json.loads(out.read_text()) if out.exists() else {}
    required=['figures/preservation_frontier.pdf','tables/preservation_summary.tex',
              'tables/preservation_costs.tex','tables/paper_finish_protocol.tex',
              'tables/publication_recipe_ranges.tex','tables/publication_robust_ranges.tex',
              'tables/publication_probe_ranges.tex']
    if old.get('signature')==signature and all((PAPER/name).exists() for name in required):
        print('paper synthesis unchanged');return 0
    spec=importlib.util.spec_from_file_location('publication_planner',ROOT/'scripts/104_paper_finish.py')
    planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
    frontiers=[];sources={};results=[]
    latest={}
    for p in sorted(summaries,key=lambda p:(json.loads(p.read_text()).get('finished_utc',''),str(p))):
        run=p.parent;summary=json.loads(p.read_text())
        tag=run.name.rsplit('_s',1)[0]
        name=f"ev_{summary['domain']}_{summary['model']}_s{summary['seed']}"+('' if tag=='small' else '_'+tag)
        if not evaluation_succeeded(run,ROOT/'runs/status',name) or (run/'WITHDRAWN.txt').exists():continue
        effects=summary.get('adapter_effectiveness',{})
        if any(v.get('identical_rate',1)>=.999 for v in effects.values()):continue
        latest[(summary['domain'],summary['model'],summary['seed'],tag)]=(run,summary)
    for run,summary in latest.values():
        f=frontier_run(run,summary)
        if f:
            frontiers.append(f);sources[str(run/'summary.json')]=source_hash(run/'summary.json')
            sources[str(run/'trials.jsonl')]=f['trial_sha256']
            for arm,cost in f['costs'].items():
                if cost.get('available'):
                    path=adapter_dir(f['domain'],f['model'],arm,32,f['seed'])
                    sources[str(path/'training_summary.json')]=cost['training_summary_sha256']
                    sources[str(path/'run_manifest.json')]=cost['manifest_sha256']
    for p in publication:
        summary=json.loads(p.read_text())
        if summary.get('engineering_smoke') or summary.get('mode')=='train':continue
        mode=summary['mode']
        cell=publication_cell(summary)
        if planner.completed(mode,cell,summary['seed']):
            results.append(dict(summary=summary,contrasts=json.loads((p.parent/'contrasts.json').read_text())
                                if (p.parent/'contrasts.json').exists() else None))
            sources[str(p)]=source_hash(p);sources[str(p.parent/'trials.jsonl')]=source_hash(p.parent/'trials.jsonl')
            if (p.parent/'contrasts.json').exists():
                sources[str(p.parent/'contrasts.json')]=source_hash(p.parent/'contrasts.json')
    plot(frontiers)
    display=[]
    for f in frontiers:
        if f['seed']!=17 or f['campaign'].startswith('contrastive'):continue
        for arm in ['base','sft','replay','mix5','mix50']:
            if arm in f['means']['forward']:
                display.append([f['domain'].replace('_',r'\_'),f['model'].replace('_',r'\_'),f['seed'],arm,
                                f"{100*f['means']['forward'][arm]:.1f}",f"{100*f['means']['reverse'][arm]:.1f}"])
    (PAPER/'tables/preservation_summary.tex').write_text(table(display,
        'Representative seed-17 preservation rates. Each model/task row set comes from one complete campaign. '
        'Additional doses, seed variation, paired intervals and measured cost records are retained in the source-backed analysis. '
        'The units boundary case is included independently of effect sign.', 'preservation-summary'))
    cost_rows=[]
    for f in frontiers:
        if f['seed']!=17:continue
        for arm in ['sft','mix5','mix50','replay','cl_fwd','cl_mix5','sft_extra_ce','mix5_extra_ce']:
            cost=f['costs'].get(arm,{})
            if not cost.get('available'):continue
            lengths=cost.get('lengths') or {}
            exposure=cost.get('direction_exposure') or {}
            runtime=cost.get('train_runtime_s')
            cost_rows.append([f['domain'].replace('_',r'\_'),f['model'].replace('_',r'\_'),arm.replace('_',r'\_'),
                'unavailable' if runtime is None else f'{runtime/60:.1f}',str(cost.get('optimizer_steps','unavailable')),
                str(lengths.get('sequence_tokens_total','unavailable'))])
    cost_tex=table(cost_rows,
        'Recorded seed-17 trainer runtimes (minutes), optimizer steps, and sequence tokens in the tokenized CE corpus '
        '(before epoch repetition). Auxiliary encoded-side exposure remains separately recorded in the analysis; '
        'these token counts are not total objective FLOPs. Runtime varies with GPU/training layout and is not '
        'a claim of matched wall time. Missing measurements are unavailable, never inferred.', 'preservation-costs')
    cost_tex=cost_tex.replace('Seed & Arm & Forward (\\%) & Reverse (\\%)','Arm & Minutes & Steps & CE corpus tokens')
    (PAPER/'tables/preservation_costs.tex').write_text(cost_tex)
    cfg=suite.plan();manifest=suite.frozen_data()
    protocol_rows=[]
    for label,key in [('General probes','probe'),('Prompt/template tests','robust'),('Independent transfer','transfer'),('Recipe evaluations','recipe')]:
        cells=cfg[key+'_panel']
        panel='; '.join(c['domain'].replace('_',r'\_')+'/'+c['model'].replace('_',r'\_') for c in cells)
        protocol_rows.append(f"{label} & {len(cells)} & {','.join(map(str,cfg['seeds']))} & {len(cells)*len(cfg['seeds'])} "+r'\\')
    protocol=('\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llll}\n\\toprule\n'
        +'Registered evaluation & Model/task cells & Training seeds & Campaigns '+r'\\'+'\n\\midrule\n'
        +'\n'.join(protocol_rows)+'\n\\bottomrule\n\\end{tabular}\n'
        +'\\caption{Frozen paper-completion panel. Probe corpora retain '+str(manifest['n_instances']['ifeval'])
        +' IFEval prompts and '+str(manifest['n_instances']['gsm8k'])+' GSM8K questions; independent transfer retains '
        +str(manifest['n_instances']['opus'])+' OPUS translation pairs. Synthetic magnitude/template tests retain '
        +str(manifest['n_instances']['units_magnitude'])+' instances each. Recipe packs train '
        +str(len(cfg['recipe_panel'])*len(cfg['seeds'])*len(cfg['recipe_variants'])*2)+' new adapters in isolated directories. '
        +'Counts are planned workloads and audited corpus sizes, not completed results.}\n'
        +'\\label{tab:paper-finish-protocol}\n\\end{table*}\n')
    (PAPER/'tables/paper_finish_protocol.tex').write_text(protocol)
    boundary_rows=[]
    for result in results:
        summary=result['summary']
        if summary['mode']!='transfer':continue
        for endpoint,gate in sorted(summary.get('gate',{}).items()):
            boundary_rows.append(' & '.join([summary['model'].replace('_',r'\_'),str(summary['seed']),
                endpoint.rsplit('/',1)[-1],f"{100*gate['rate']:.1f}",
                f"{100*gate['format_fail']:.1f}",f"{100*gate['echo_probe_strict']:.1f}",
                'pass' if summary['eligibility_passed'] else 'blocked'])+r' \\')
    if boundary_rows:
        (PAPER/'tables/transfer_boundaries.tex').write_text(
            '\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lllllll}\n\\toprule\n'
            +'Model & Seed & Direction & Base (\\%) & Format failure (\\%) & Echo (\\%) & Campaign '+r'\\'+'\n\\midrule\n'
            +'\n'.join(boundary_rows)+'\n\\bottomrule\n\\end{tabular}\n'
            +'\\caption{Completed OPUS transfer eligibility measurements with unchanged thresholds. '
            'Each campaign requires both directions to satisfy the frozen rate, format-failure, and '
            'echo conjunction. Failed campaigns preserve base-only trials and do not generate tuned '
            'comparisons; an echo failure is a criterion-validity boundary, not evidence of adapter collapse.}\n'
            +'\\label{tab:transfer-boundaries}\n\\end{table*}\n')
    for mode in ('probes','transfer','robust','recipe'):
        selected=[r for r in results if r['summary']['mode']==mode and r['summary'].get('eligibility_passed')]
        if not selected:continue
        output=[]
        for result in selected:
            s=result['summary']
            for c in result['contrasts']['contrasts']:
                if c['b']!='sft' and (c['a'],c['b'])!=('sft','base'):continue
                output.append(' & '.join([s['domain'].replace('_',r'\_'),s['model'].replace('_',r'\_'),str(s['seed']),
                    c['endpoint'].replace('_',r'\_'),(c['a']+' vs '+c['b']).replace('_',r'\_'),
                    f"{c['delta_pp']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]"])+r' \\')
        # Long tables stay as standalone source-backed artifacts until final paper selection.
        (PAPER/f'tables/publication_{mode}.tex').write_text(
            '\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llllll}\n\\toprule\n'
            +'Task & Model & Seed & Endpoint & Contrast & Change [interval] '+r'\\'+'\n\\midrule\n'
            +'\n'.join(output)+'\n\\bottomrule\n\\end{tabular}\n\\caption{Completed '+mode
            +' exploratory strict-success changes in percentage points with paired cluster-bootstrap intervals. '
            +'No equivalence or across-cell pooling claim.}\n\\end{table*}\n')
    atomic_json(out,dict(signature=signature,updated_utc=datetime.now(timezone.utc).isoformat(),
        frontiers=frontiers,publication_results=results,source_sha256=sources,
        inference='paired exploratory within-campaign intervals; seed ranges descriptive; no pooled/equivalence verdict',
        cost_limit='runtime and exposure only where recorded; no invented FLOPs, queue or evaluation time'))
    for mode in ('recipe', 'robust'):
        (PAPER/f'tables/publication_{mode}_ranges.tex').write_text(publication_range_table(results, mode))
    (PAPER/'tables/publication_probe_ranges.tex').write_text(publication_probe_table(results))
    atomic_json(PAPER/'PUBLICATION_PROVENANCE.json',dict(source_sha256=sources,signature=signature,
        artifacts=['figures/preservation_frontier.pdf','figures/preservation_frontier.png',
                   'tables/preservation_summary.tex','tables/preservation_costs.tex',
                   'tables/paper_finish_protocol.tex',
                   *[str(p.relative_to(PAPER)) for p in sorted((PAPER/'tables').glob('publication_*.tex'))],
                   *(['tables/transfer_boundaries.tex'] if boundary_rows else [])],
        generator_sha256=LOADED_SOURCE_SHA256))
    print('Paper synthesis:',len(frontiers),'frontiers;',len(results),'validated publication campaigns')
    return 0


if __name__=='__main__':
    with (ROOT/'runs/feeder/.paper_synthesis.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('paper synthesis already active');raise SystemExit(0)
        raise SystemExit(main())
