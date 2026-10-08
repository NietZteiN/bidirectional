#!/usr/bin/env python
"""Amendment37 same-pass contrastive contrasts; only complete successful evaluations."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.config import RESULTS_DIR
from bidir.schema import iter_jsonl

spec=importlib.util.spec_from_file_location('paired_contrasts',ROOT/'scripts'/'50_contrasts.py')
paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)
PRIMARY=[('cl_fwd','sft'),('cl_fwd','sft_extra_ce'),('cl_fwd','cl_shuffled'),
         ('cl_mix5','mix5'),('cl_mix5','mix5_extra_ce')]
REQUIRED={'base','sft','mix5','replay','rev','cl_fwd','cl_mix5','cl_shuffled','sft_extra_ce','mix5_extra_ce'}


def validate_rows(rows,expected):
    """Reject partial/duplicate coverage, rather than silently comparing intersections."""
    coverage=defaultdict(set); clusters={}
    for row in rows:
        if row['strategy']!='simple': continue
        system=row['system']; direction=row['direction']; pid=row['pair_id']
        if system not in REQUIRED: continue
        key=(system,direction)
        if pid in coverage[key]: raise ValueError('duplicate paired trial')
        coverage[key].add(pid)
        if row['strict'] not in (0,1): raise ValueError('strict must be binary')
        cluster=row.get('cluster_id') or pid
        if pid in clusters and clusters[pid]!=cluster: raise ValueError('inconsistent cluster identity')
        clusters[pid]=cluster
    common=None
    for direction in ('forward','reverse'):
        for system in REQUIRED:
            pairs=coverage[(system,direction)]
            if len(pairs)!=expected: raise ValueError(f'incomplete {system}/{direction} coverage')
            if common is None: common=pairs
            elif pairs!=common: raise ValueError('systems do not share the same evaluation pairs')


def analyse_run(run,n_boot=5000):
    summary=json.loads((run/'summary.json').read_text())
    trials=run/'trials.jsonl';rows=list(iter_jsonl(trials))
    validate_rows(rows,summary['n_instances'])
    contrasts=[]
    for direction in ('reverse','forward'):
        for a,b in PRIMARY:
            result=paired.paired_delta(rows,a,b,'strict',direction,n_boot=n_boot,confidence=0.99)
            result['primary']=direction=='reverse'
            result['interpretation']=('positive interval' if result['ci_lo']>0 else
                                      'negative interval' if result['ci_hi']<0 else 'interval spans zero')
            # The pilot registers point estimates/intervals, not a new equivalence test.
            result.pop('verdict');result.pop('margin_pp')
            contrasts.append(result)
    return {'run':str(run),'domain':summary['domain'],'model':summary['model'],'seed':summary['seed'],
            'trial_sha256':hashlib.sha256(trials.read_bytes()).hexdigest(),
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'bootstrap_script_sha256':hashlib.sha256((ROOT/'scripts'/'50_contrasts.py').read_bytes()).hexdigest(),
            'evaluation_finished_utc':summary['finished_utc'],'n_boot':n_boot,
            'confidence':0.99,'family':'five primary reverse contrasts within this cell',
            'multiplicity':'Bonferroni alpha0.05/5; exploratory across cells',
            'same_pass_only':True,'contrasts':contrasts}


def successful_eval(run):
    try:
        summary=json.loads((run/'summary.json').read_text())
        name=f"ev_{summary['domain']}_{summary['model']}_s{summary['seed']}_contrastive_pilot"
        for path in (ROOT/'runs'/'status').glob(name+'.*.json'):
            status=json.loads(path.read_text())
            if status['exit_code']==0 and status['finished_utc']>=summary['finished_utc']:
                return True
    except (OSError,KeyError,ValueError): pass
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--scan',action='store_true')
    parser.add_argument('--n-boot',type=int,default=5000)
    args=parser.parse_args()
    runs=[args.run] if args.run else sorted(RESULTS_DIR.glob('*/*/*/contrastive_pilot_s*')) if args.scan else []
    written=0
    for run in runs:
        out=run/'contrastive_contrasts.json'
        if args.scan:
            if not successful_eval(run): continue
            if out.exists():
                try:
                    cached=json.loads(out.read_text())
                    if (cached.get('trial_sha256')==hashlib.sha256((run/'trials.jsonl').read_bytes()).hexdigest()
                        and cached.get('analysis_script_sha256')==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        and cached.get('bootstrap_script_sha256')==hashlib.sha256((ROOT/'scripts'/'50_contrasts.py').read_bytes()).hexdigest()
                        and cached.get('n_boot')==args.n_boot):
                        continue
                except (OSError,ValueError): pass
        result=analyse_run(run,args.n_boot)
        temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(out)
        written+=1;print(f'[contrastive analysis] wrote {out}',flush=True)
    if written:
        try:
            subprocess.run([sys.executable,str(ROOT/'scripts'/'94_archive_trials.py')],
                           capture_output=True,timeout=60,check=True)
        except (subprocess.SubprocessError,OSError) as exc:
            print(f'[contrastive analysis] archive retry needed: {exc}',file=sys.stderr)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
