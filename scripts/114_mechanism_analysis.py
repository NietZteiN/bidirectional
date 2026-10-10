#!/usr/bin/env python
"""Collect proof-valid Amendment52 evidence without selecting favorable diagnostics."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx, mechanism_text_graph as text_graph
from bidir.pipeline_state import atomic_json
from bidir import paper_sprint


def result_path(job):return text_graph.output(job) if text_graph.applies(job) else mx.output(job)
def is_complete(job):return text_graph.completed(job) if text_graph.applies(job) else mx.completed(job)


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]


def likelihood_report(job):
    cfg=mx.plan();out=result_path(job);rows=read_rows(out/'likelihood.jsonl')
    ids=[i['pair_id'] for i in mx.selected(job['domain'])[:cfg['likelihood_n']]]
    lookup={(r['system'],r['direction'],r['pair_id'],r['candidate']):r for r in rows}
    if len(lookup)!=len(rows) or len(rows)!=len(cfg['systems'])*2*len(ids)*3:
        raise ValueError('likelihood result coverage invalid')
    systems=[];contrasts=[]
    for arm in cfg['systems']:
        for direction in ['forward','reverse']:
            r=dict(system=arm,direction=direction,n=len(ids))
            for metric in ['nll_total','nll_per_token']:
                for candidate in ['gold','echo','shuffled']:
                    values=[lookup[(arm,direction,i,candidate)][metric] for i in ids]
                    r[candidate+'_'+metric]=sum(values)/len(values)
                for candidate in ['echo','shuffled']:
                    margins=[lookup[(arm,direction,i,candidate)][metric]-lookup[(arm,direction,i,'gold')][metric] for i in ids]
                    r['gold_beats_'+candidate+'_'+metric]=sum(m>0 for m in margins)/len(margins)
                    contrasts.append(dict(system=arm,direction=direction,reference=candidate,
                        metric='candidate_minus_gold_'+metric,**mx.paired_interval(margins,reps=cfg['bootstrap_reps'])))
                if arm!='base':
                    delta=[lookup[(arm,direction,i,'gold')][metric]-lookup[('base',direction,i,'gold')][metric] for i in ids]
                    contrasts.append(dict(system=arm,direction=direction,reference='base',metric='gold_delta_'+metric,
                        **mx.paired_interval(delta,reps=cfg['bootstrap_reps'])))
                if arm=='mix5':
                    delta=[lookup[(arm,direction,i,'gold')][metric]-lookup[('sft',direction,i,'gold')][metric] for i in ids]
                    contrasts.append(dict(system=arm,direction=direction,reference='sft',metric='gold_delta_'+metric,
                        **mx.paired_interval(delta,reps=cfg['bootstrap_reps'])))
            systems.append(r)
    layerjob=mx.item({k:v for k,v in job.items() if k in ['domain','model','role']},job['seed'],'layers')
    joined=[]
    if mx.completed(layerjob):
        gen={(r['system'],r['direction'],r['pair_id']):r for r in read_rows(mx.output(layerjob)/'trials.jsonl')}
        for arm in cfg['generation_systems']:
            for direction in ['forward','reverse']:
                for i in ids:
                    gold=lookup[(arm,direction,i,'gold')];echo=lookup[(arm,direction,i,'echo')]
                    raw=gen[(arm,direction,i)]
                    joined.append(dict(system=arm,direction=direction,pair_id=i,
                        strict_noecho=raw['strict_noecho'],echo=raw.get('echo',0),
                        off_target=raw.get('off_target',0),gold_beats_copy_total=gold['nll_total']<echo['nll_total'],
                        gold_beats_copy_per_token=gold['nll_per_token']<echo['nll_per_token']))
        # Same pairs, separately recorded engines. Preserve individual failure cases.
        from bidir.paper_suite import write_rows
        write_rows(out/'generation_likelihood_join.jsonl',joined)
    return dict(systems=systems,contrasts=contrasts,
        gradient_states=json.loads((out/'gradients.json').read_text())['states'],
        update_contrasts=json.loads((out/'contrasts.json').read_text())['comparisons'],
        joined_generation_pairs=len(joined),generation_likelihood_engines_differ=True)


def sprint_tables(document):
    config=paper_sprint.plan()
    if config is None:return
    selected=[r for r in document['items'] if r.get('execution_name',r['name']) in config['production_jobs']]
    document['sprint']=dict(config,production_total=len(selected),
        production_complete=sum(r['state']=='complete' for r in selected))
    esc=lambda value:str(value).replace('_',r'\_')
    interval=lambda c:f"{c['mean']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
    text=[r'\section{Budget-limited explanatory diagnostics}',r'\label{app:mechanism-sprint}',
        'Amendment54 limits new compute to five fixed seed-17 diagnostics and required engineering smokes. '
        'The original 24-job panel remains registered; unrun jobs are deferred. Selection uses the time budget '
        'and distinct scientific questions, not new diagnostic outcomes. Completed earlier seeds are retained. '
        'These single-seed analyses are exploratory; paired intervals measure test-pair uncertainty, '
        'not training-seed stability. All candidates, controls, bands and local steps remain in the evidence bundle.',
        f"Proof-valid diagnostic jobs at this snapshot: {document['sprint']['production_complete']} of {len(selected)}."]
    hf_rows=[];gradient_rows=[];layer_rows=[]
    for record in selected:
        if record['state']!='complete':continue
        result=json.loads(Path(record['analysis']).read_text())['result']
        cell=[esc(record['domain']),esc(record['model'])]
        if record['mode']=='hf':
            for direction in ['forward','reverse']:
                lookup={(c['system'],c['reference']):c for c in result['contrasts']
                    if c['direction']==direction and c['metric']=='gold_delta_nll_per_token'}
                hf_rows.append(cell+[direction,interval(lookup[('sft','base')]),interval(lookup[('mix5','sft')])])
            states={r['system']:r for r in result['gradient_states']}
            gradient_rows.append(cell+[f"{states[a]['cosine']:+.3f}" for a in ['sft','replay','mix5']])
        else:
            for c in result['contrasts']:
                if c['system'].endswith('_remove') and c['reference'].startswith('band'):
                    layer_rows.append([esc(c['system']),esc(c['reference']),c['direction'],interval(c)])
    llama_hf={r['domain'] for r in selected if r['state']=='complete'
        and r['mode']=='hf' and r['model']=='llama32-3b'}
    if {'fmt','mt_de-en'}<=llama_hf:
        text.append('In the completed Llama formatting and translation diagnostics, forward SFT '
            'lowers gold completion loss in the forward direction and raises it in reverse; '
            'mixed-direction SFT lowers reverse loss relative to SFT with small mean forward changes '
            '(Table~\\ref{tab:sprint-likelihood}). The local SFT gradient cosines are near zero '
            '(Table~\\ref{tab:sprint-gradients}); these checkpoints provide little support for a '
            'simple persistent gradient-conflict account. Gradients at these final checkpoints '
            'do not reconstruct interference along the training trajectory. This descriptive '
            'evidence concerns answer likelihood and does not establish stored knowledge.')
    def table(headers,rows,caption,label):
        if not rows:return
        text.extend([r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{'+'l'*len(headers)+'}',
            r'\toprule',' & '.join(headers)+r' \\',r'\midrule',
            '\n'.join(' & '.join(row)+r' \\' for row in rows),r'\bottomrule',r'\end{tabular}',
            r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\end{table*}'])
    table(['Task','Model','Direction','SFT minus base','mix5 minus SFT'],hf_rows,
        r'Gold completion NLL per token, paired differences with descriptive 95\% bootstrap intervals '
        'on 128 fixed pairs. Negative values favor the first system. Candidate likelihood does not '
        'establish generation correctness or stored knowledge. All seeds here are 17.', 'sprint-likelihood')
    table(['Task','Model','SFT','Replay','mix5'],gradient_rows,
        'Forward/reverse validation-gradient cosine on 16 pairs in local LoRA A/B coordinates. '
        'Gemma excludes inactive vision factors. These coordinate-dependent quantities do not '
        'by themselves establish interference or causality.', 'sprint-gradients')
    table(['Removed band','Matched control','Direction','Success difference'],layer_rows,
        'All fixed-depth removals versus retained-norm-matched controls for formatting/Llama3B '
        r'(seed17), strict no-echo success differences on 512 pairs with descriptive 95\% paired '
        'intervals. Retained and edit norms are both recorded; only retained norms are matched. '
        'No best band or control is selected.', 'sprint-layer-controls')
    path=ROOT/'paper/tables/mechanism_sprint.tex';content='\n'.join(text)+'\n'
    if not path.exists() or path.read_text()!=content:path.write_text(content)
    document['sprint']['table_sha256']=mx.sha(path)


def main():
    if not (mx.DATA/'manifest.json').exists():print('mechanism analysis awaits prepared data');return 0
    records=[];source_sha={};complete=0
    for job in mx.items():
        if job['smoke']:continue
        valid=is_complete(job)
        job_name=text_graph.name(job) if text_graph.applies(job) else job['name']
        record=dict(job,state='complete' if valid else paper_sprint.deferred_reason(job_name) or 'awaiting proof-valid result',
                    **(text_graph.public_status(job) if text_graph.applies(job) else {}))
        if valid:
            complete+=1;out=result_path(job);summary=json.loads((out/'summary.json').read_text())
            for filename in ['summary.json',*summary['file_sha256']]:source_sha[str(out/filename)]=mx.sha(out/filename)
            cache=out/'explanatory_analysis.json'
            signature=dict(summary_sha256=mx.sha(out/'summary.json'),analyzer_sha256=mx.sha(__file__),
                paired_layer_summary_sha256=None)
            if job['mode']=='hf':
                layerjob=mx.item({k:v for k,v in job.items() if k in ['domain','model','role']},job['seed'],'layers')
                if mx.completed(layerjob):signature['paired_layer_summary_sha256']=mx.sha(mx.output(layerjob)/'summary.json')
            previous=json.loads(cache.read_text()) if cache.exists() else {}
            if previous.get('signature')!=signature:
                result=(likelihood_report(job) if job['mode']=='hf' else
                    dict(contrasts=json.loads((out/'contrasts.json').read_text())['comparisons'],
                         variant_manifest=json.loads((out/'variant_manifest.json').read_text())))
                atomic_json(cache,dict(signature=signature,result=result))
            record['analysis']=str(cache);source_sha[str(cache)]=mx.sha(cache)
        records.append(record)
    document=dict(protocol=mx.plan()['protocol'],production_complete=complete,production_total=24,
        items=records,source_sha256=source_sha,panel_sha256=mx.sha(mx.DATA/'manifest.json'),
        analyzer_sha256=mx.sha(__file__),interpretation=mx.plan()['interpretation'],
        multiplicity='Descriptive unadjusted paired intervals; no best-band/step/seed selection or pooled mechanism verdict.')
    sprint_tables(document)
    target=ROOT/'paper/MECHANISM_DIAGNOSTICS.json'
    if not target.exists() or json.loads(target.read_text())!=document:atomic_json(target,document)
    print(f'Mechanism analysis: {complete}/24 production jobs proof-valid')
    return 0


if __name__=='__main__':raise SystemExit(main())
