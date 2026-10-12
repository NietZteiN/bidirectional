#!/usr/bin/env python
"""Publish complete paired-judgment comparisons and paired uncertainty automatically."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main():
    worker=module('pj_report_worker','124_paired_judgment_worker.py')
    renderer=module('pj_report_rates','117_main_results.py')
    paired=module('pj_report_intervals','50_contrasts.py')
    paper=ROOT/'paper/bidir';records=[];pending=[];sources={
        str(Path(__file__)):worker.sha(__file__),str(Path(worker.__file__)):worker.sha(worker.__file__),
        str(worker.PLAN):worker.sha(worker.PLAN)}
    for cell in worker.plan()['cells']:
        if not worker.completed(cell,'eval'):
            pending.append(cell);continue
        receipt=worker.receipt_path(cell,'eval');r=worker.read(receipt);run=Path(r['run'])
        means=renderer.campaign_rates(run,r['trial_sha256'],sources)
        sources[str(receipt)]=worker.sha(receipt)
        signature=hashlib.sha256(json.dumps([r['signature'],r['trial_sha256'],worker.sha(__file__),worker.sha(ROOT/'scripts/50_contrasts.py')]).encode()).hexdigest()
        path=run/'paired_judgment_contrasts.json'
        if path.exists() and worker.read(path).get('signature')==signature:
            contrasts=worker.read(path)['contrasts']
        else:
            rows=[json.loads(line) for line in (run/'trials.jsonl').read_text().splitlines()]
            contrasts=[paired.paired_delta(rows,a,b,'strict',d,n_boot=2000)
                       for a in worker.plan()['new_arms'] for b in ['sft','mix5'] for d in ['forward','reverse']]
            contrasts += [paired.paired_delta(rows,'paired_judge','paired_judge_shuffle','strict',d,n_boot=2000)
                          for d in ['forward','reverse']]
            if any(c is None for c in contrasts):raise ValueError('missing paired contrast')
            atomic_json(path,dict(signature=signature,contrasts=contrasts,trial_sha256=r['trial_sha256']))
        sources[str(path)]=worker.sha(path)
        for arm in worker.plan()['new_arms']:
            out=worker.output_adapter(cell,arm)
            for name in ['training_summary.json','run_manifest.json','paired_judgment_protocol.json']:
                sources[str(out/name)]=worker.sha(out/name)
        records.append(dict(cell,seed=worker.plan()['seed'],run=str(run),means=means,contrasts=contrasts))
    content='';interval_content=''
    if records:
        grid=renderer.grid(records,worker.plan()['evaluation_arms'],
                           ['Base','Forward SFT',r'5\% reversal','PJ-5','Shuffled labels'],True)
        content='\n'.join([r'\begin{table*}[t]\centering\footnotesize',grid,
          r'\caption{Forward (F) and backward (B) strict success (\%) with paired judgments '
          r'(PJ-5). PJ-5 replaces the same nominal 5\% of pairs as reversed-pair SFT with '
          r'balanced matching/mismatching judgments; shuffled labels use identical inputs '
          r'with randomized Yes/No targets. These are separately designed cross-domain '
          r'baselines, not code-equivalence CFT. Each task/model comparison uses the same '
          r'test examples; training seed 17. Only complete evaluations are shown.}'
          r'\label{tab:paired-judgment}',r'\end{table*}'])
        lines=[r'\begin{table*}[t]\centering\footnotesize',r'\begin{tabular}{lllrr}\toprule',
               r'Task/model & Method & Compared with & Forward change & Backward change \\\midrule']
        for record in records:
            for a,b in [(a,b) for a in worker.plan()['new_arms'] for b in ['sft','mix5']]+[('paired_judge','paired_judge_shuffle')]:
                cs=[next(c for c in record['contrasts'] if (c['a'],c['b'],c['direction'])==(a,b,d)) for d in ['forward','reverse']]
                names={'paired_judge':'PJ-5','paired_judge_shuffle':'Shuffled labels','sft':'Forward SFT','mix5':r'5\% reversal'}
                fields=[renderer.TASKS[record['domain']]+'/'+renderer.MODELS[record['model']],names[a],names[b]]
                fields += [f"{c['delta_pp']:+.1f} [{c['ci_lo']:+.1f}, {c['ci_hi']:+.1f}]" for c in cs]
                lines.append(' & '.join(fields)+r' \\')
        lines += [r'\bottomrule\end{tabular}',r'\caption{Paired-judgment changes in percentage points with 95\% paired cluster-bootstrap intervals (2,000 draws). Positive values favor the first method. Intervals are exploratory and unadjusted for multiple comparisons.}\label{tab:paired-judgment-intervals}\end{table*}']
        interval_content='\n'.join(lines)
    for name,text in [('paired_judgment_results.tex',content),
                      ('paired_judgment_intervals.tex',interval_content)]:
        path=paper/'tables'/name
        if not path.exists() or path.read_text()!=text+'\n':path.write_text(text+'\n')
    report=dict(protocol=worker.plan()['protocol'],complete=len(records),total=len(worker.plan()['cells']),
                records=records,pending=pending,source_sha256=sources)
    destination=paper/'PAIRED_JUDGMENT_RESULTS.json'
    if not destination.exists() or worker.read(destination)!=report:atomic_json(destination,report)
    print('Paired judgments:',len(records),'/',len(worker.plan()['cells']),'complete')
    return 0


if __name__=='__main__':raise SystemExit(main())
