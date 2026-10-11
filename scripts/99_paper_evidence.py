#!/usr/bin/env python
"""Audited evidence snapshot; partial campaigns stay explicitly partial."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.config import RESULTS_DIR
from bidir.pipeline_state import evaluation_succeeded,atomic_json

STATUS=ROOT/'runs/status'
PAPER=ROOT/'paper'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def escape(text):return text.replace('_',r'\_')
def latest_core_campaigns(campaigns):
    latest={}
    for cell in sorted(campaigns,key=lambda c:(c['finished_utc'],c['run'])):
        latest[(cell['domain'],cell['model'],cell['seed'])]=cell
    superseded=[c for c in campaigns if latest[(c['domain'],c['model'],c['seed'])]['run']!=c['run']]
    return list(latest.values()),superseded
def table(headers,rows,caption,label):
    return ('\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule\n'
        +' & '.join(headers)+r' \\'+'\n\\midrule\n'
        +'\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
        +'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{tab:'+label+'}\n\\end{table*}\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--if-changed',action='store_true')
    ap.add_argument('--build',action='store_true')
    args=ap.parse_args()
    tracked=set()
    for pattern in ['*/*/*/*/trials.jsonl','*/*/*/*/summary.json',
                    '*/*/*/*/contrastive_contrasts.json','*/*/*/*/followup_contrasts.json','base_gates/*/*.json',
                    'audit/fullft_saved_repair_*.json']:
        tracked.update(RESULTS_DIR.glob(pattern))
    tracked.update(STATUS.glob('*.json'))
    tracked.update([Path(__file__),PAPER/'main.tex',ROOT/'scripts/93_numbers.py',ROOT/'scripts/117_main_results.py',
                    ROOT/'scripts/118_manuscript_completion.py'])
    tracked.update((ROOT/'data').glob('*/build_report.json'))
    tracked.update((ROOT/'data').glob('*/*.jsonl'))
    from bidir.config import RUNS_DIR
    for filename in ['training_summary.json','run_manifest.json']:
        tracked.update(RUNS_DIR.glob(f'adapters/*/llama32-3b/sft_r32_s17/{filename}'))
    for path in [PAPER/'MECHANISM_DIAGNOSTICS.json',ROOT/'configs/paper_sprint.json',
                 PAPER/'tables/mechanism_sprint.tex']:
        if path.exists():tracked.add(path)
    if (PAPER/'planned_evaluations.tex').exists():
        tracked.add(PAPER/'planned_evaluations.tex')
    if (PAPER/'PUBLICATION_ANALYSIS.json').exists():
        tracked.add(PAPER/'PUBLICATION_ANALYSIS.json')
    signature=hashlib.sha256(json.dumps(sorted((str(p),p.stat().st_size,p.stat().st_mtime_ns)
                                             for p in tracked)).encode()).hexdigest()
    snapshot=PAPER/'EVIDENCE_SNAPSHOT.json'
    unchanged=False
    if args.if_changed and snapshot.exists():
        unchanged=json.loads(snapshot.read_text()).get('input_signature')==signature
        main_outputs = [PAPER/'MAIN_RESULTS.json', PAPER/'tables/main_adaptations.tex',
                        PAPER/'tables/main_loss_baselines.tex',PAPER/'MANUSCRIPT_COMPLETION.json',
                        PAPER/'tables/methods_details.tex',PAPER/'tables/revised_domain_results.tex']
        if unchanged and (not args.build or (all(p.exists() for p in main_outputs) and (PAPER/'main.pdf').exists()
                                            and (PAPER/'main.pdf').stat().st_mtime>=snapshot.stat().st_mtime)):
            print('paper evidence unchanged');return 0
    if unchanged:
        build_paper();return 0
    reports=[]; sources={}; cells=[]; rejected=[]
    for p in sorted(RESULTS_DIR.glob('*/*/*/contrastive_pilot_s*/contrastive_contrasts.json')):
        d=json.loads(p.read_text());run=p.parent
        name=f"ev_{d['domain']}_{d['model']}_s{d['seed']}_contrastive_pilot"
        if not evaluation_succeeded(run,STATUS,name) or d['trial_sha256']!=sha(run/'trials.jsonl'):
            rejected.append(str(run));continue
        if not d.get('same_pass_only') or len([c for c in d['contrasts'] if c.get('primary')])!=5:
            rejected.append(str(run));continue
        reports.append(d);sources[str(p)]=sha(p);sources[str(run/'trials.jsonl')]=d['trial_sha256']
    reports.sort(key=lambda d:(d['domain'],d['model'],d['seed']))
    rows=[]; control_rows=[]
    for d in reports:
        contrasts={(c['a'],c['b']):c for c in d['contrasts'] if c.get('primary')}
        vals=[]
        for key in [('cl_fwd','sft'),('cl_mix5','mix5')]:
            c=contrasts[key];vals.append(f"{c['delta_pp']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
        rows.append([escape(d['domain']),escape(d['model']),d['seed'],*vals])
        control_values=[]
        for key in [('cl_fwd','sft_extra_ce'),('cl_fwd','cl_shuffled'),('cl_mix5','mix5_extra_ce')]:
            c=contrasts[key]
            control_values.append(f"{c['delta_pp']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
        display_model={'llama32-3b':'Llama 3B','gemma3-4b':'Gemma 4B'}.get(d['model'],escape(d['model']))
        control_rows.append([escape(d['domain']),display_model,d['seed'],*control_values])
    (PAPER/'tables').mkdir(exist_ok=True)
    (PAPER/'tables/contrastive_snapshot.tex').write_text(table(
        ['Task','Model','Seed',r'$\Delta$ CL-fwd versus SFT',r'$\Delta$ CL-mix5 versus mix5'],rows,
        'Completed exploratory reverse-generation contrasts (percentage points), with 99\\% paired cluster-bootstrap intervals. '
        'Five primary contrasts are adjusted within each cell; no adjustment across cells. '
        'All proof-valid original and scale/domain-extension cells are included; failed-gate cells '
        'remain eligibility boundaries rather than tuned comparisons.',
        'contrastive-snapshot'))
    (PAPER/'tables/contrastive_controls.tex').write_text(table(
        ['Task','Model','Seed','CL-fwd vs extra CE','CL-fwd vs shuffled','CL-mix5 vs extra CE'],control_rows,
        'The remaining three registered primary reverse-generation contrasts, in percentage points with 99\\% '
        'paired cluster-bootstrap intervals. Together with Table~\\ref{tab:contrastive-snapshot}, these display '
        'all five primary comparisons for every completed cell. Extra CE is an exposure/compute proxy, not '
        'matched measured FLOPs. Degenerate zero intervals at an observed generation floor do not establish '
        'equivalence. Results remain exploratory; new scale/domain extensions are pending.', 'contrastive-controls'))
    gates=[]
    for domain in ['units','logic','py_cpp']:
        for model in ['llama32-3b','gemma3-4b']:
            paths=sorted((RESULTS_DIR/'base_gates'/domain).glob(model+'_*.json'))
            if not paths:continue
            p=paths[-1];d=json.loads(p.read_text());sources[str(p)]=sha(p)
            gates.append([escape(domain),escape(model),f"{100*d['directions']['forward']['rate']:.1f}",
                          f"{100*d['directions']['reverse']['rate']:.1f}", 'Pass' if d['passes_gate'] else 'Fail'])
    (PAPER/'tables/domain_gates_snapshot.tex').write_text(table(
        ['Task','Model','Forward','Reverse','Eligibility'],gates,
        'Original new-domain base gates: strict success in percent. Failed cells are retained and receive no tuned pilot. '
        'Explicit-output diagnostics use the same semantic splits and are separate exploratory variants.', 'domain-gates-snapshot'))
    diagnostic_gates=[]
    for domain in ['units_explicit','logic_explicit','py_cpp_explicit']:
        for model in ['llama32-3b','gemma3-4b']:
            paths=sorted((RESULTS_DIR/'base_gates'/domain).glob(model+'_*.json'))
            if not paths:continue
            p=paths[-1];d=json.loads(p.read_text())
            if 'passes_gate' not in d:continue
            diagnostic_gates.append(dict(source=str(p),**d));sources[str(p)]=sha(p)
    repair_audits={}
    for p in sorted((RESULTS_DIR/'audit').glob('fullft_saved_repair_*.json')):
        d=json.loads(p.read_text())
        if d.get('passed') is True:
            repair_audits[d['checkpoint']]=d;sources[str(p)]=sha(p)
    for trial in sorted(RESULTS_DIR.glob('*/*/*/small_s*/trials.jsonl')):
        run=trial.parent
        if not re.fullmatch(r'small_s\d+',run.name):continue
        try:
            summary=json.loads((run/'summary.json').read_text())
            if summary['domain']=='fmt_novel':continue  # never-had control, not an eligible main task
            name=f"ev_{summary['domain']}_{summary['model']}_s{summary['seed']}"
            if not evaluation_succeeded(run,STATUS,name):
                rejected.append(str(run));continue
            desired={'base','sft','mix5','mix50','replay','rev'} & set(summary['arms'])
            groups={};seen=set()
            for line in trial.open():
                r=json.loads(line)
                if r['system'] not in desired or r.get('strategy')!='simple':continue
                key=(r['system'],r['direction'],r['pair_id'])
                if key in seen:raise ValueError('duplicate trial')
                seen.add(key)
                if r['strict'] not in [0,1]:raise ValueError('nonbinary strict')
                groups.setdefault(key[:2],{})[r['pair_id']]=r['strict']
            expected=groups.get(('base','reverse'),{})
            if len(expected)!=summary['n_instances'] or not expected:raise ValueError('incomplete base')
            if any(set(groups.get((s,d),{}))!=set(expected) for s in desired for d in ['forward','reverse']):
                raise ValueError('incomplete arm coverage')
            means={s:{d:sum(groups[(s,d)].values())/len(expected) for d in ['forward','reverse']} for s in desired}
            effect=summary.get('adapter_effectiveness',{})
            if any(s not in effect or effect[s]['identical_rate']>=.999 for s in desired-{'base'}):
                raise ValueError('effectiveness guard failed/missing')
            cells.append(dict(domain=summary['domain'],model=summary['model'],seed=summary['seed'],
                n_instances=len(expected),run=str(run),means=means,
                finished_utc=summary['finished_utc'],
                collapse=means['sft']['reverse']<=.5*means['base']['reverse'] if 'sft' in means else None))
            sources[str(trial)]=sha(trial)
        except (KeyError,ValueError,OSError) as exc:rejected.append(str(run)+': '+str(exc))
    audited_campaigns=len(cells)
    cells,superseded=latest_core_campaigns(cells)
    followups=[];full_rows=[];generality_rows=[];full_cells=0
    for p in sorted(RESULTS_DIR.glob('*/*/*/*/followup_contrasts.json')):
        d=json.loads(p.read_text());trial=p.parent/'trials.jsonl'
        if (p.parent/'WITHDRAWN.txt').exists() or d['trial_sha256']!=sha(trial):continue
        followups.append(d);sources[str(p)]=sha(p);sources[str(trial)]=d['trial_sha256']
        if p.parent.name.startswith('fullft_campaign_s'):
            m=d['means']
            full_cells += 1
            for direction in ['forward','reverse']:
                full_rows.append([escape(d['domain']),d['seed'],direction,
                    f"{100*m['base'][direction]:.1f}",f"{100*m['fullft_sft'][direction]:.1f}",
                    f"{100*m['fullft_mix5'][direction]:.1f}"])
        if p.parent.name.startswith('generality_scale_pilot_s'):
            for arm in ['base','sft','mix50','replay','rev']:
                m=d['means'][arm]
                generality_rows.append([escape(d['domain']),escape(d['model']),d['seed'],arm,
                                       f"{100*m['forward']:.1f}",f"{100*m['reverse']:.1f}"])
    (PAPER/'tables/fullft_snapshot.tex').write_text(table(
        ['Task','Seed','Direction','Base','Full SFT','Full mix5'],full_rows,
        'Completed Llama3B full-weight campaign rates in percent, with forward and reverse retention. '
        'Checkpoints use isolated engines and identical held-out instances; '
        'these comparisons span engine restarts. Paired intervals are recorded in followup contrasts. '
        'An empty body means no validated full campaign has completed.', 'fullft-snapshot'))
    (PAPER/'tables/generality_snapshot.tex').write_text(table(
        ['Task','Model','Seed','Arm','Forward','Reverse'],generality_rows,
        'Eligible larger-model generality campaigns: strict success in percent, with all arms and the base '
        'evaluated in one fresh pass per model/task/seed. The completed unit-conversion replications '
        'improve both directions under SFT and are retained null-collapse boundary cases. '
        'Additional eligible campaigns enter this table after validated analysis.', 'generality-snapshot'))
    # Keep every model/seed separate. These summaries are inventories, not pooled tests.
    macros={'contrastive-completed-cells':len(reports),'contrastive-planned-cells':27,
            'audited-core-cells':len(cells),'failed-new-domain-gates':sum(r[-1]=='Fail' for r in gates),
            'audited-core-campaigns':audited_campaigns,
            'failed-explicit-domain-gates':sum(d['passes_gate'] is False for d in diagnostic_gates),
            'audited-fullft-repairs':len(repair_audits),
            'core-collapse-cells':sum(c['collapse'] is True for c in cells),
            'core-model-count':len({c['model'] for c in cells}),'core-domain-count':len({c['domain'] for c in cells}),
            'fullft-completed-cells':full_cells,'fullft-planned-cells':6}
    model_rows=[]
    for model in sorted({c['model'] for c in cells}):
        group=[c for c in cells if c['model']==model]
        model_rows.append([escape(model),len({c['domain'] for c in group}),len(group),sum(c['collapse'] is True for c in group)])
    (PAPER/'tables/core_inventory.tex').write_text(table(
        ['Model','Task orientations','Cells','Threshold met'],model_rows,
        'Audited model--task--seed cells. Threshold met means SFT reverse strict success is at most half the same-pass base rate. '
        'Counts are descriptive and are not independent task counts or pooled significance tests.', 'core-inventory'))
    plot_core(cells)
    atomic_json(PAPER/'EVIDENCE_SNAPSHOT.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        input_signature=signature,
        resolved_numbers=macros,core_cells=cells,contrastive_reports=reports,followup_reports=followups,
        explicit_domain_gates=diagnostic_gates,fullft_repair_audits=list(repair_audits.values()),
        superseded_core_campaigns=superseded,
        core_selection_rule='latest validated completion timestamp per domain/model/training seed; no effect-sign selection',
        rejected_runs=rejected,sources=sources,
        mechanism_claims='five audited sprint diagnostic campaigns; additional registered diagnostics unmeasured; no cross-model/seed pooling',
        legacy_auxiliary_arms='requires current withdrawal/exposure audit in scripts/117_main_results.py before main-table inclusion'))
    atomic_json(PAPER/'PROVENANCE.json',dict(
        generated_by='scripts/99_paper_evidence.py',updated_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={name:dict(evidence='EVIDENCE_SNAPSHOT.json',source_hashes=sources)
                   for name in ['tables/core_inventory.tex','tables/contrastive_snapshot.tex',
                                'tables/contrastive_controls.tex',
                                'tables/domain_gates_snapshot.tex','tables/fullft_snapshot.tex','tables/generality_snapshot.tex',
                                'figures/directional_loss_snapshot.pdf']},
        input_signature=signature))
    (PAPER/'evidence_snapshot.tex').write_text('\n'.join(
        r'\expandafter\def\csname bidirnum@'+k+r'\endcsname{'+str(v)+'}' for k,v in macros.items())+'\n')
    print(json.dumps(dict(core_cells=len(cells),contrastive_reports=len(reports),rejected=len(rejected)),indent=2))
    if args.build:build_paper()
    return 0

def plot_core(cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=PAPER/'figures';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(5.4,4.3),layout='constrained')
    for model in sorted({c['model'] for c in cells}):
        group=[c for c in cells if c['model']==model and 'sft' in c['means']]
        ax.scatter([100*c['means']['base']['reverse'] for c in group],
                   [100*c['means']['sft']['reverse'] for c in group],
                   label=model,s=25,alpha=.65,linewidths=.3,edgecolors='white')
    ax.plot([0,100],[0,100],color='.6',linewidth=1,label='Unchanged')
    ax.plot([0,100],[0,50],color='.3',linestyle='--',linewidth=1,label='Half of base')
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='Untouched reverse strict success (%)',
           ylabel='Reverse success after forward SFT (%)')
    ax.legend(fontsize=7,loc='upper left',framealpha=.9)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'directional_loss_snapshot.pdf')
    fig.savefig(folder/'directional_loss_snapshot.png',dpi=170)
    plt.close(fig)

def build_paper():
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/117_main_results.py')],cwd=ROOT,check=True,timeout=180)
    subprocess.run([sys.executable,str(ROOT/'scripts/118_manuscript_completion.py')],cwd=ROOT,check=True,timeout=180)
    subprocess.run([sys.executable,str(ROOT/'scripts/93_numbers.py')],cwd=ROOT,check=True,timeout=180)
    completed=subprocess.run(['make','paper'],cwd=ROOT,capture_output=True,text=True,timeout=180)
    (ROOT/'runs/feeder/paper_build_latest.log').write_text(completed.stdout+completed.stderr)
    print('\n'.join(completed.stdout.splitlines()[-5:]))
    completed.check_returncode()

if __name__=='__main__':
    import fcntl
    with (ROOT/'runs/feeder/.paper_evidence.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('paper evidence refresh already active');raise SystemExit(0)
        raise SystemExit(main())
