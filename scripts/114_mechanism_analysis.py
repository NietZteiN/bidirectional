#!/usr/bin/env python
"""Collect proof-valid Amendment52 evidence without selecting favorable diagnostics."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import mechanism_suite as mx
from bidir.pipeline_state import atomic_json


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]


def likelihood_report(job):
    cfg=mx.plan();out=mx.output(job);rows=read_rows(out/'likelihood.jsonl')
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


def main():
    if not (mx.DATA/'manifest.json').exists():print('mechanism analysis awaits prepared data');return 0
    records=[];source_sha={};complete=0
    for job in mx.items():
        if job['smoke']:continue
        valid=mx.completed(job)
        record=dict(job,state='complete' if valid else 'awaiting proof-valid result')
        if valid:
            complete+=1;out=mx.output(job);summary=json.loads((out/'summary.json').read_text())
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
    target=ROOT/'paper/MECHANISM_DIAGNOSTICS.json'
    if not target.exists() or json.loads(target.read_text())!=document:atomic_json(target,document)
    print(f'Mechanism analysis: {complete}/24 production jobs proof-valid')
    return 0


if __name__=='__main__':raise SystemExit(main())
