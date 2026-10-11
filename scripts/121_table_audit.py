#!/usr/bin/env python
"""Compute omitted descriptive ratios and track every missing numerical table field."""
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
spec=importlib.util.spec_from_file_location('audited_rates',ROOT/'scripts/117_main_results.py')
rates=importlib.util.module_from_spec(spec);spec.loader.exec_module(rates)
PAPER=ROOT/'paper'


def normalized_change(base,sft,mixed):
    """A signed descriptive ratio, including null/adverse cells; zero is undefined."""
    denominator=base-sft
    return None if denominator==0 else 100*(mixed-sft)/denominator


def legacy_tables(snapshot,sources):
    available={(c['domain'],c['model'],c['seed']):c for c in snapshot['core_cells']}
    cached={};ratios=[]
    for filename in ['main.tex','dose.tex','mixture.tex']:
        path=PAPER/'tables'/filename
        if not path.exists():continue
        source=path.read_text();match=re.search(r'\\midrule\n(.*?)\n\\bottomrule',source,re.S)
        if not match:continue
        header=source.split(r'\toprule',1)[1].split(r'\midrule',1)[0].strip().removesuffix(r'\\').strip()
        headers=[x.strip() for x in header.split('&')]
        new_rows=[]
        for line in match.group(1).splitlines():
            if not line.strip().endswith(r'\\'):new_rows.append(line);continue
            columns=[x.strip() for x in line.strip().removesuffix(r'\\').split('&')]
            k=(columns[0].replace(r'\_','_'),columns[1],int(columns[2]))
            c=available[k]
            if k not in cached:
                run=Path(c['run'])
                cached[k]=rates.campaign_rates(run,snapshot['sources'][str(run/'trials.jsonl')],sources)
            means=cached[k]
            for i,h in enumerate(headers[3:],3):
                if filename=='mixture.tex' and i==len(headers)-1:
                    ratio=normalized_change(means['base']['reverse'],means['sft']['reverse'],means['mixedtask']['reverse'])
                    columns[i]=r'\textit{undefined}' if ratio is None else f'{ratio:+.1f}'+r'\%'
                    ratios.append(dict(domain=k[0],model=k[1],seed=k[2],ratio_percent=ratio,run=c['run']))
                    continue
                arm=h.split('$',1)[0].replace(r'\_','_')
                direction='forward' if r'\rightarrow' in h else 'reverse'
                columns[i]=f'{100*means[arm][direction]:.1f}' if arm in means else '--'
            new_rows.append(' & '.join(columns)+r' \\')
        source=source[:match.start(1)]+'\n'.join(new_rows)+source[match.end(1):]
        if filename=='mixture.tex':
            source=source.replace(' & recovered ', ' & Normalized change ')
            source=re.sub(r'\\caption\{.*?\}\n\\label',
                lambda _:r'\caption{Reverse strict success under mixed-task SFT. Normalized change is '
                r'$100(\mathrm{mixedtask}-\mathrm{sft})/(\mathrm{base}-\mathrm{sft})$, calculated '
                r'for every nonzero denominator, including null/adverse cells. Outside collapse '
                r'cells this is a signed descriptive normalization, not a recovery verdict; '
                r'negative or near-zero denominators limit interpretation. All rates in a row '
                r'use one complete evaluation campaign.}\n\label'.replace(r'\n','\n'),source,flags=re.S)
        if filename=='dose.tex':
            source=re.sub(r'\\caption\{.*?\}\n\\label',
                lambda _:r'\caption{Backward strict success by nominal replacement dose. Instances '
                r'and optimizer steps are matched; realized token counts are audited separately. '
                r'Dashes are pending measurements. Low-dose relation/execution additions are '
                r'exploratory below-batch extensions under Amendment57, excluded from the '
                r'original registered knee analysis.}\n\label'.replace(r'\n','\n'),source,flags=re.S)
        path.write_text(source)
    return ratios


def missing_fields(directory):
    fields=[]
    for path in sorted(directory.glob('*.tex')):
        for row,line in enumerate(path.read_text().splitlines(),1):
            if '&' not in line or not line.strip().endswith(r'\\'):continue
            for col,value in enumerate(line.strip().removesuffix(r'\\').split('&'),1):
                if value.strip() in ['--',r'$\varnothing$']:
                    fields.append(dict(table=path.name,row=row,column=col,source_row=line))
    return fields


def main():
    snapshot=json.loads((PAPER/'EVIDENCE_SNAPSHOT.json').read_text())
    sources={};ratios=legacy_tables(snapshot,sources)
    main=json.loads((PAPER/'MAIN_RESULTS.json').read_text())
    runnable=[];structural=[]
    for c in main['core']:
        for arm in rates.CORE_ARMS:
            if arm not in c['means']:runnable.append(dict(table='main_adaptations.tex',domain=c['domain'],model=c['model'],seed=c['seed'],arm=arm,directions=['forward','reverse']))
    for c in main['auxiliary']:
        for arm in ['cft','unlikelihood','roundtrip']:
            if arm not in c['means']:
                target=dict(table='main_loss_baselines.tex',domain=c['domain'],model=c['model'],seed=c['seed'],arm=arm,directions=['forward','reverse'])
                (structural if arm=='cft' and c['domain']!='code' else runnable).append(target)
    atomic_json(PAPER/'TABLE_COMPLETION_AUDIT.json',dict(
        generator_sha256=rates.sha(__file__),missing_numeric_fields=missing_fields(PAPER/'tables'),
        pending_campaign_arms=runnable,code_specific_cft_entries=structural,
        computed_normalized_changes=ratios,source_sha256=sources,
        note='All generated tables inventoried, including legacy tables outside the current PDF. The user chose code-specific CFT: translation/SQL entries are not applicable and no extension is queued. Undefined ratios are mathematical boundaries.'))
    print('Table audit:',len(runnable),'pending campaign arms;',len(structural),'code-specific CFT entries;',len(ratios),'normalized changes calculated')
    return 0


if __name__=='__main__':
    import fcntl
    with (ROOT/'runs/feeder/.table_audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raise SystemExit(main())
