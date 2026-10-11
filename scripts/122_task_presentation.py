#!/usr/bin/env python
"""Describe actual tasks, training doses and uncertainty without changing any experiment."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import domains, prompts
from bidir.config import load_config, resolve_thresholds, RESULTS_DIR
from bidir.pipeline_state import atomic_json
from bidir.train import adapter_dir
spec=importlib.util.spec_from_file_location('presentation_rates',ROOT/'scripts/117_main_results.py')
rates=importlib.util.module_from_spec(spec);spec.loader.exec_module(rates)
PAPER=ROOT/'paper'
ARMS=['base','sft','mix1','mix5']


def escape(text):
    mapping={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&',
             '#':r'\#','{':r'\{','}':r'\}','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    return ''.join(mapping.get(c,c) for c in str(text))


def training_counts(domain, sources):
    records={}
    for arm in ['sft','mix1','mix5']:
        path=adapter_dir(domain,'llama32-3b',arm,32,17)/'training_summary.json'
        if not path.exists():continue
        t=json.loads(path.read_text());sources[str(path)]=rates.sha(path)
        if arm=='sft':records['train']=t['lengths']['n_kept']
        else:
            # Balance is before length filtering. Report only the reversed rows retained.
            n=t['balance']['by_task'].get('rev',0)-t['lengths'].get('dropped_by_task',{}).get('rev',0)
            records[arm]=dict(reverse_pairs=n,retained_pairs=t['lengths']['n_kept'])
    return records


def task_table(counts):
    rows=[
      ('code','Code transformation',r'Readable Python $\to$ obfuscated Python; e.g. renaming \texttt{count} to \texttt{a0} without changing execution. Backward requests readable code.',
       'Execution agrees on stored tests and output is not a copy. Original identifier names are not required.'),
      ('fmt','Format conversion',r'JSON $\to$ YAML/XML; Markdown table $\to$ CSV. Backward converts back; e.g. \texttt{\{"id": 3\}} $\leftrightarrow$ \texttt{id: 3}.',
       'Target parser accepts output, data structures agree, and output is not a copy. YAML syntax need not be exclusive to YAML.'),
      ('mt_en-de','Translation',r"English $\leftrightarrow$ German/Chinese; both training orientations. E.g. ``Good morning'' $\leftrightarrow$ ``Guten Morgen''.",
       'Target-language check and COMET-22 above a fixed model/direction threshold; backward additionally rejects copies.'),
      ('sql',r'Question $\to$ SQL',r"Question plus schema $\to$ SQLite query; backward produces a question. E.g. ``How many rows?'' $\leftrightarrow$ \texttt{SELECT COUNT(*) FROM t}.",
       'Forward: equal database results. Backward: a frozen parser maps the question to a query with equal results.'),
      ('relation','Factual relations',r'Country $\to$ capital, ISO code or TLD; backward identifies the country. E.g. France $\leftrightarrow$ Paris.',
       'Normalized first-line exact match, excluding copies. Ambiguous relation values are removed before splitting.'),
      ('exec','Execution prediction',r'Function and argument $\to$ return value; backward finds an argument. For $f(x)=x^2$, $3\to9$; backward accepts $3$ or $-3$.',
       'Forward: return value agrees with execution. Backward: any argument executing to the requested value is correct.'),
      ('algebra','Algebra',r'Factored $\to$ expanded polynomial, and the opposite training orientation. E.g. $(x+1)(x-1)\leftrightarrow x^2-1$.',
       'Symbolic equivalence and no copy; factoring also requires a nonexpanded product. Expansion has no separate form check.')]
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
      r'\begin{tabular}{p{0.15\textwidth}p{0.38\textwidth}p{0.34\textwidth}r}',
      r'\toprule',r'Task & Input and required output & Primary success check & Train \\',r'\midrule']
    for d,label,task,criterion in rows:
        lines.append(' & '.join([label,task,criterion,str(counts[d]['train'])])+r' \\')
        lines.append(r'\addlinespace[3pt]')
    lines += [r'\bottomrule',r'\end{tabular}',
      r'\caption{Task definitions and small illustrative examples (not scored test items). '
      r'Train is the number of retained forward-SFT pairs for the Llama3B reference run. '
      r'Sources: local Python transformation corpus, synthetic formats/algebra, News Commentary '
      r'(training) and FLORES-200 (test), Spider, country-relation instruction datasets, '
      r'and CRUXEval, respectively. Appendix~\ref{app:task-scoring} gives splits, exact prompt '
      r'templates, thresholds, parsing and exclusions.}\label{tab:task-definitions}',r'\end{table*}']
    return '\n'.join(lines)+'\n'


DETAILS={
 'code':('Python transformation','The local Python corpus supplies readable/obfuscated pairs over five transformations: misleading identifier renaming (L1b), random hexadecimal renaming (L1r), sequential minification (L2), control-flow flattening (S1), and opaque predicates/dead code (S2). Training/validation use the transformation-generation pool; held-out tests include only programs available under all five transformations. A program is the uncertainty cluster. Renaming destroys the original names, so the inverse is not unique. Both primary directions require agreement on every stored execution test and reject copies. Transformation-style compliance, original-name recovery, CodeBLEU and readability are not conjuncts of primary success; the workshop readability/similarity predicate is secondary.'),
 'fmt':('Serialization conversion','A seeded generator creates lossless JSON/YAML, JSON/XML, and Markdown-table/CSV pairs. JSON uses json.loads, YAML yaml.safe_load, XML an ElementTree decoder, CSV csv.reader, and Markdown a pipe-table parser that removes separator rows. Equality requires matching dictionary keys, container shapes and list order; scalar strings are stripped and lowercased to accommodate XML/CSV type conversion. Copies are rejected in both directions. JSON can parse as YAML: the copy guard excludes unchanged or near-unchanged inputs, but this scorer does not enforce YAML-exclusive syntax or case-sensitive scalar identity.'),
 'mt_en-de':('Machine translation','Training uses the Helsinki-NLP/news_commentary Hub export, German--English and English--Chinese configurations. Both sides must have 20--400 characters; normalized source duplicates are removed. Test data use FLORES-200 devtest from the haoranxu/FLORES-200 mirror. Each orientation trains its designated source-to-target mapping. COMET uses Unbabel/wmt22-comet-da with source, generated translation and reference. Success requires nonempty output, the target-language check, and COMET at or above the frozen threshold. Backward additionally rejects copies; the original forward predicate does not. Chinese requires a Han-character fraction above 0.15; English/German use langdetect probability above 0.5 and reject Han characters. A language-detector exception accepts the language check, a limitation. chrF++ and BLEU are secondary. The threshold is the 25th percentile of scores from 200 untouched-model outputs for each model and direction, before tuned evaluation. Thus success percentages compare training methods within a model/direction, not absolute translation quality across models.'),
 'sql':('Semantic parsing and verbalization','Questions and queries come from the official database-disjoint xlangai/spider train/validation splits; SQLite databases come from the prem-research/spider mirror, whose pooled question files are not used. The official training split is shuffled into training and validation; official development databases form the test set. Both prompts include a schema summary. Forward outputs must contain the whole word SELECT (case-insensitive) and pass the Spider test-suite execution evaluator on the example database, with plug_value=False and keep_distinct=False; alternatives with equal denotations can pass. Reverse questions are parsed by frozen ibm-granite/granite-3.1-8b-instruct and checked by the same execution comparison. Empty outputs, copied queries and outputs containing that whole word are rejected in reverse. This keyword guard can reject a valid question, as can parser errors. Its reference-question accuracy is a calibration diagnostic, not a mathematical ceiling for every generated paraphrase.'),
 'relation':('Country relations','Lots-of-LoRAs/task1146_country_capital, task1314_country_abbreviation and task1320_country_domain_tld provide country--capital, country--ISO-code and country--TLD pairs. Values associated with multiple countries and identical input/target strings are removed; many-to-one currency/calling-code relations are excluded. Splits and uncertainty clusters are countries, keeping their relations together. The scorer uses the first output line, removes a leading "the" or "answer" marker and a trailing period, then compares after Unicode/whitespace normalization. It is case-sensitive and does not accept unrestricted aliases or alternative names. Copying is rejected.'),
 'exec':('Output and input prediction','The cruxeval-org/cruxeval test export contains 800 Python function/input/output problems. Corpus construction removes identical input/output strings, unexecutable programs, inconsistent reference outputs, and problems where feeding the output back into the function is a valid reverse answer. The retained corpus is split by program. Forward compares the predicted literal with the canonicalized executed return value. Reverse executes the generated argument and accepts any preimage producing the required return value; matching the stored input is unnecessary. Code fences are stripped. Forward answers and reference return values are parsed with ast.literal_eval, falling back to stripped text; generated reverse arguments are passed directly to the execution harness. Execution-based canonicalization determines equality.'),
 'algebra':('Expansion and factorization','A seeded SymPy generator multiplies primitive integer-coefficient linear factors into degree-2--4 polynomials in one or two variables, using the configured root range. Expanded symbolic expressions are deduplicated across splits. The expansion orientation has factored inputs and expanded targets; the factorization orientation swaps the sides of the same generated pairs. Answers allow caret notation and implicit multiplication. Success requires symbolic equivalence after expansion and excludes copies. Factoring also requires a non-Add top level whose expansion differs from itself; it does not require a canonical or complete irreducible factorization. The expansion scorer has no separate expanded-form conjunct, despite the prompt asking for a polynomial.')}


def task_appendix(counts, sources):
    lines=[r'\section{Task data, prompts and correctness}',r'\label{app:task-scoring}',
      'Strict success is a binary task-specific predicate, not a common metric across tasks. '
      'The primary evaluation uses greedy decoding and the simple instruction without demonstrations. '
      'A shared system message and the same backward instruction are used in training and evaluation. '
      'Output extraction strips complete or opened Markdown code fences. Echo detection compares '
      'Unicode-NFKC/whitespace-normalized strings and also rejects character-trigram Jaccard overlap '
      'at least 0.95 when both strings have at least twelve characters. Domain-specific exceptions '
      'are stated below; no new scoring rule is applied to the reported results.',
      r'\paragraph{Dataset versions and splits.} '
      'The saved paired JSONL corpora, resolved build configurations, prompt/scorer sources and SHA-256 '
      'hashes are listed in the accompanying task-methods metadata. Some original upstream dataset '
      'revisions were not recorded; current cached Hub revisions cannot establish the historical '
      'training version. The frozen paired corpora therefore specify the reproducible analysis '
      'inputs, without implying exact reconstruction of all historical training data.',
      r'\paragraph{Shared system message.}',r'\begin{quote}\small',escape(prompts.SYSTEM),r'\end{quote}']
    metadata={}
    for d,(title,description) in DETAILS.items():
        cfg=load_config('domains/'+d+'.yaml');mod=domains.get(d)
        config_path=ROOT/'configs/domains'/(d+'.yaml');sources[str(config_path)]=rates.sha(config_path)
        file=Path(mod.__file__);sources[str(file)]=rates.sha(file)
        buildpath=ROOT/'data'/d/'build_report.json';build=json.loads(buildpath.read_text())
        sources[str(buildpath)]=rates.sha(buildpath)
        for split in ['train','val','test']:
            p=ROOT/'data'/d/(split+'.jsonl');sources[str(p)]=rates.sha(p)
        n=build['counts']
        lines += [r'\subsection{'+title+'}',escape(description),
          f"Built train/validation/test pairs: {n['train']}/{n['val']}/{n['test']}. "
          f"Retained reference forward-SFT pairs: {counts[d]['train']}. "]
        dose=[]
        for arm,pct in [('mix1',1),('mix5',5)]:
            if arm in counts[d]:dose.append(f"{pct}\\%: {counts[d][arm]['reverse_pairs']} reversed pairs among {counts[d][arm]['retained_pairs']} retained pairs")
        if dose:lines.append('Realized reference doses: '+ '; '.join(dose)+'.')
        templates=[]
        subtasks=cfg.get('subtasks',['capital','abbrev','tld'] if d=='relation' else ['<TRANSFORMATION>'])
        for sub in subtasks:
            inst=dict(side_a='<INPUT A>',side_b='<INPUT B>',subtask=sub,
              meta=dict(schema='<DATABASE SCHEMA>',code='<FUNCTION CODE>',entry_point='f',relation=sub if d=='relation' else 'capital'))
            for direction in ['forward','reverse']:
                text=mod.instruction(direction,inst);templates.append(dict(subtask=sub,direction=direction,user=text))
                lines += [r'\paragraph{'+direction.capitalize()+(' / '+escape(sub) if len(subtasks)>1 else '')+' prompt.}',
                          r'\begin{quote}\small\ttfamily',escape(text).replace('\n',r'\par '),r'\end{quote}']
        metadata[d]=dict(build=build,training_counts=counts[d],prompt_templates=templates,resolved_configuration=cfg,
                        upstream_revision=cfg.get('revision') or cfg.get('hf_revision'),scorer_source=str(file))
    lines += [r'\subsection{Other translation and algebra orientations}',
      'The following templates complete the opposite training directions and Chinese pair. '
      'Placeholders identify the input side in that orientation; the shared system message is unchanged.']
    for d,parent in [('mt_de-en','mt_en-de'),('mt_en-zh','mt_en-de'),('mt_zh-en','mt_en-de'),('algebra_rev','algebra')]:
        mod=domains.get(d);sources[str(Path(mod.__file__))]=rates.sha(mod.__file__)
        inst=dict(side_a='<INPUT A>',side_b='<INPUT B>',subtask='',meta={})
        templates=[]
        for direction in ['forward','reverse']:
            text=mod.instruction(direction,inst);templates.append(dict(direction=direction,user=text))
            lines += [r'\paragraph{'+rates.TASKS[d]+' / '+direction+' prompt.}',
                      r'\begin{quote}\small\ttfamily',escape(text).replace('\n',r'\par '),r'\end{quote}']
        metadata[parent].setdefault('other_orientations',{})[d]=templates
    lines += [r'\subsection{Translation thresholds}',
      'Thresholds below are frozen separately by model, orientation and direction; each row '
      'uses the base model calibration described above. They must not be used to rank models '
      'by absolute translation quality.',r'\begin{table*}[t]\centering\small',
      r'\begin{tabular}{llrr}\toprule Task orientation & Model & Forward $\tau$ & Backward $\tau$ \\\midrule']
    thresholds=[]
    for d in ['mt_en-de','mt_de-en','mt_en-zh','mt_zh-en']:
        path=ROOT/'configs/domains'/(d+'.yaml');sources[str(path)]=rates.sha(path)
        cfg=load_config('domains/'+d+'.yaml')
        for model in rates.MODELS:
            resolved=resolve_thresholds(cfg,model);v=resolved['thresholds']
            thresholds.append(dict(domain=d,model=model,thresholds=v))
            lines.append(' & '.join([rates.TASKS[d],rates.MODELS[model],f"{v['forward']['comet']:.6f}",f"{v['reverse']['comet']:.6f}"])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Frozen COMET-22 strict-success thresholds.}\label{tab:translation-thresholds}\end{table*}',
      r'\subsection{Model-based correctness checks}',
      'SQL-to-question is the model-based check in the main comparison: a frozen Granite parser '
      'maps a generated question back to SQL, whose database result is compared with the target. '
      'The other proposed round-trip direction is WebNLG triples-to-text: the same model family '
      'extracts triples from generated text. Its extractor fails on reference texts, so this '
      'task supplies no tuned-model result and is absent from the main task table. Neither check '
      'is executable or symbolic verification of the generated natural-language meaning itself.']
    ceiling=[]
    for p in sorted((RESULTS_DIR/'base_gates/sql').glob('*.json')):
        t=json.loads(p.read_text())
        if 'roundtrip_ceiling' in t:
            ceiling.append(dict(model=t['model'],n=t['n'],reference_accuracy=t['roundtrip_ceiling'],source=str(p)))
            sources[str(p)]=rates.sha(p)
    if ceiling:
        lines += [r'\begin{table}[t]\centering\small',
          r'\begin{tabular}{lrr}\toprule Model & $n$ & Reference accuracy \\\midrule']
        for t in ceiling:lines.append(' & '.join([rates.MODELS[t['model']],str(t['n']),f"{100*t['reference_accuracy']:.1f}\\%"])+r' \\')
        lines += [r'\bottomrule\end{tabular}',r'\caption{Frozen SQL parser calibration on reference questions in the corresponding initial evaluation subset. Variation reflects the subsets and execution layout, not a parser tuned to each evaluated model.}\label{tab:parser-calibration}\end{table}']
    return '\n'.join(lines)+'\n',metadata,thresholds,ceiling


def paired_intervals(rows, arms=ARMS, reps=2000):
    """Shared cluster resampling preserves pairing across every arm and direction."""
    import numpy as np
    groups={(a,d):{} for a in arms for d in ['forward','reverse']}
    cluster={}
    for r in rows:
        if r.get('strategy')!='simple' or r['system'] not in arms:continue
        k=(r['system'],r['direction']);pid=r['pair_id']
        if pid in groups[k]:raise ValueError('duplicate presentation trial')
        groups[k][pid]=r['strict'];cid=r.get('cluster_id') or pid
        if pid in cluster and cluster[pid]!=cid:raise ValueError('inconsistent presentation cluster')
        cluster[pid]=cid
    ids=sorted(groups[('base','forward')])
    if not ids or any(set(g)!=set(ids) for g in groups.values()):raise ValueError('incomplete presentation coverage')
    units=defaultdict(list)
    for i,pid in enumerate(ids):units[cluster[pid]].append(i)
    units=list(units.values());draws=np.random.default_rng(17).integers(0,len(units),(reps,len(units)))
    den=np.array([len(u) for u in units])[draws].sum(axis=1)
    values={k:np.array([g[i] for i in ids]) for k,g in groups.items()}
    def interval(v):
        sums=np.array([v[u].sum() for u in units]);lo,hi=np.quantile(sums[draws].sum(axis=1)/den,[.025,.975])
        return dict(mean=float(v.mean()),lo=float(lo),hi=float(hi))
    return dict(rates={a:{d:interval(values[(a,d)]) for d in ['forward','reverse']} for a in arms},
                deltas={a:{d:interval(values[(a,d)]-values[('sft',d)]) for d in ['forward','reverse']} for a in arms if a!='sft'})


def evidence_presentation(main,snapshot,sources):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panel=['fmt','code','mt_de-en','sql'];comparisons={};fig,axes=plt.subplots(1,4,figsize=(7.2,2.35))
    colors=['#333333','#be3b34','#287eae','#26835b'];names=['Base','Forward SFT','1% reversal','5% reversal']
    for ax,d in zip(axes,panel):
        cell=next(c for c in main['core'] if c['domain']==d);run=Path(cell['run'])
        rates.campaign_rates(run,cell['trial_sha256'],sources)
        rows=[json.loads(l) for l in (run/'trials.jsonl').open()]
        ci=paired_intervals(rows);comparisons[d]=dict(run=str(run),intervals=ci)
        for a,color in zip(ARMS,colors):
            x,y=[100*ci['rates'][a][dr]['mean'] for dr in ['forward','reverse']]
            fx,ry=ci['rates'][a]['forward'],ci['rates'][a]['reverse']
            ax.errorbar(x,y,xerr=[[max(0,x-100*fx['lo'])],[max(0,100*fx['hi']-x)]],
              yerr=[[max(0,y-100*ry['lo'])],[max(0,100*ry['hi']-y)]],fmt='o',color=color,ms=4,elinewidth=.7,capsize=1.5)
        ax.set(xlim=(-4,104),ylim=(-4,104),title={'fmt':'Formats','code':'Python','mt_de-en':'German → English','sql':'Question → SQL'}[d],
               xlabel='Forward success (%)',xticks=[0,50,100],yticks=[0,50,100])
        ax.tick_params(labelsize=7);ax.set_title(ax.get_title(),fontsize=8);ax.xaxis.label.set_size(7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Backward success (%)',fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],marker='o',color=c,linestyle='none',markersize=4) for c in colors],names,
               loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.07,right=.99,top=.86,bottom=.28,wspace=.35)
    fig.savefig(PAPER/'figures/preservation_focus.pdf');fig.savefig(PAPER/'figures/preservation_focus.png',dpi=180)
    plt.close(fig)
    seed_cells=[]
    for cell in snapshot['core_cells']:
        if cell['model']!='llama32-3b':continue
        run=Path(cell['run']);m=rates.campaign_rates(run,snapshot['sources'][str(run/'trials.jsonl')],sources)
        if not set(ARMS)<=set(m):continue
        seed_cells.append(dict(domain=cell['domain'],seed=cell['seed'],run=str(run),means=m))
    lines=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llrrrr}\toprule',
      r'Task & Runs & $\Delta F_{1-S}$ & $\Delta B_{1-S}$ & $\Delta F_{5-S}$ & $\Delta B_{5-S}$ \\\midrule']
    variation=[]
    for d in rates.TASKS:
        cells=[c for c in seed_cells if c['domain']==d]
        if len(cells)<2:continue
        vals=[[100*(c['means'][a][dr]-c['means']['sft'][dr]) for a in ['mix1','mix5'] for dr in ['forward','reverse']] for c in cells]
        ranges=[(min(v[i] for v in vals),max(v[i] for v in vals)) for i in range(4)]
        variation.append(dict(domain=d,seeds=sorted(c['seed'] for c in cells),runs=[c['run'] for c in cells],ranges=ranges))
        lines.append(' & '.join([rates.TASKS[d],str(len(cells)),*[f'[{lo:+.1f}, {hi:+.1f}]' for lo,hi in ranges]])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Training-run variation for every Llama3B task orientation with at least two runs measuring both 1\% and 5\% reversal. Entries are minimum/maximum within-run differences from forward SFT, in percentage points; they are not confidence intervals. Every available qualifying run is included regardless of outcome. $F/B$ means forward/backward; subscripts 1 and 5 identify reversal proportions. Other tasks and model coverage remain in Table~\ref{tab:core-inventory}.}\label{tab:main-seed-variation}\end{table*}']
    (PAPER/'tables/main_seed_variation.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llrr}\toprule',
      r'Task & Mixture & Forward change & Backward change \\\midrule']
    for d in panel:
        for a,label in [('mix1',r'1\%'),('mix5',r'5\%')]:
            q=comparisons[d]['intervals']['deltas'][a]
            lines.append(' & '.join([rates.TASKS[d],label,*[f"{100*q[dr]['mean']:+.1f} [{100*q[dr]['lo']:+.1f}, {100*q[dr]['hi']:+.1f}]" for dr in ['forward','reverse']]])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Paired reversal-mixture changes from forward SFT for Figure~\ref{fig:dose-tradeoff-draft}, in percentage points with 95\% cluster-bootstrap intervals (2,000 draws). These exploratory intervals are not adjusted for multiple comparisons and measure test-instance uncertainty, not training-run variability.}\label{tab:preservation-paired}\end{table*}']
    (PAPER/'tables/preservation_paired.tex').write_text('\n'.join(lines)+'\n')
    return comparisons,variation


def failure_reasons(sources):
    records=[]
    for d in ['units','logic','py_cpp']:
        for m in ['llama32-3b','gemma3-4b']:
            paths=sorted((RESULTS_DIR/'base_gates'/d).glob(m+'_*.json'))
            if not paths:continue
            p=paths[-1];t=json.loads(p.read_text());sources[str(p)]=rates.sha(p)
            reasons=[]
            for direction,label in [('forward','forward'),('reverse','backward')]:
                q=t['directions'][direction]
                if q['rate']<.1:reasons.append(label+' success below 10\\%')
                if q['format_fail']>.15:reasons.append(label+' format failures above 15\\%')
            if t.get('max_echo_probe_strict',0)>.02:reasons.append('copy success above 2\\%')
            if not t['passes_gate'] and not reasons:raise ValueError('unexplained task exclusion: '+str(p))
            records.append(dict(domain=d,model=m,reasons=reasons,source=str(p)))
    lines=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llp{0.62\textwidth}}\toprule',
      r'Task & Model & Reason an original comparison is excluded \\\midrule']
    for t in records:lines.append(' & '.join([escape(t['domain']),rates.MODELS[t['model']],'; '.join(t['reasons']) or 'All initial requirements satisfied'])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Scientific reasons for excluding the original small-model task extensions. Correctness criteria are unchanged; revised-output versions are separate exploratory comparisons.}\label{tab:initial-failure-reasons}\end{table*}']
    (PAPER/'tables/initial_failure_reasons.tex').write_text('\n'.join(lines)+'\n')
    return records


def auxiliary_intervals(main,sources):
    records=[]
    lines=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{lllrr}\toprule',
      r'Task/model & Objective & Compared with & Forward change & Backward change \\\midrule']
    for c in main['auxiliary']:
        run=Path(c['run']);path=run/'followup_contrasts.json';report=json.loads(path.read_text())
        if report['trial_sha256']!=c['trial_sha256']:raise ValueError('auxiliary interval trials changed')
        rates.campaign_rates(run,c['trial_sha256'],sources);sources[str(path)]=rates.sha(path)
        for method in sorted(set(c['means']) & {'cft','unlikelihood','roundtrip'}):
            for reference in ['sft','mix5']:
                pair=[]
                for d in ['forward','reverse']:
                    contrast=next(q for q in report['contrasts'] if
                        (q['a'],q['b'],q['direction'],q['metric'])==(method,reference,d,'strict'))
                    if contrast['confidence']!=.95:raise ValueError('wrong auxiliary confidence')
                    expected=100*(c['means'][method][d]-c['means'][reference][d])
                    if abs(expected-contrast['delta_pp'])>1e-8:raise ValueError('auxiliary interval means changed')
                    pair.append(contrast)
                records.append(dict(domain=c['domain'],model=c['model'],method=method,reference=reference,
                                    run=str(run),contrasts=pair))
                lines.append(' & '.join([rates.TASKS[c['domain']]+'/'+rates.MODELS[c['model']],
                    {'cft':'CFT','unlikelihood':'Unlikelihood','roundtrip':'Round-trip'}[method],
                    'Forward SFT' if reference=='sft' else r'5\% reversal',
                    *[f"{q['delta_pp']:+.1f} [{q['ci_lo']:+.1f}, {q['ci_hi']:+.1f}]" for q in pair]])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Auxiliary-objective differences from forward-only and reversed-pair SFT, in percentage points with 95\% paired cluster-bootstrap intervals. Every completed compatible objective comparison is included. Intervals are exploratory and unadjusted for multiplicity; small point-estimate gains do not establish an objective-specific benefit.}\label{tab:auxiliary-intervals}\end{table*}']
    (PAPER/'tables/auxiliary_intervals.tex').write_text('\n'.join(lines)+'\n')
    return records


def main():
    sources={str(Path(__file__)):rates.sha(__file__),str(ROOT/'src/bidir/prompts.py'):rates.sha(ROOT/'src/bidir/prompts.py')}
    snapshot=json.loads((PAPER/'EVIDENCE_SNAPSHOT.json').read_text());main=json.loads((PAPER/'MAIN_RESULTS.json').read_text())
    counts={d:training_counts(d,sources) for d in ['code','fmt','mt_en-de','sql','relation','exec','algebra']}
    (PAPER/'tables/task_definitions.tex').write_text(task_table(counts))
    appendix,tasks,thresholds,ceiling=task_appendix(counts,sources)
    (PAPER/'task_methods.tex').write_text('% GENERATED by scripts/122_task_presentation.py\n'+appendix)
    comparison,variation=evidence_presentation(main,snapshot,sources)
    excluded=failure_reasons(sources)
    auxiliary=auxiliary_intervals(main,sources)
    numbers={}
    for c in variation:
        prefix='main-repeat-'+c['domain'].replace('_','-')
        trials=[str(Path(run)/'trials.jsonl') for run in c['runs']]
        numbers[prefix+'-runs']=dict(value=str(len(c['seeds'])),sources=trials)
        for index,name in enumerate(['mix1-forward','mix1-backward','mix5-forward','mix5-backward']):
            for value,label in zip(c['ranges'][index],['lo','hi']):
                numbers[prefix+'-'+name+'-'+label]=dict(value=f'{value:.1f}',sources=trials)
    atomic_json(PAPER/'TASK_METHODS.json',dict(tasks=tasks,translation_thresholds=thresholds,parser_calibration=ceiling,
      presentation_intervals=comparison,training_run_variation=variation,initial_exclusions=excluded,
      auxiliary_intervals=auxiliary,numbers=numbers,source_sha256=sources,
      generator_sha256=rates.sha(__file__),note='Presentation only: original prompts, training and scorers unchanged. Missing historical upstream revisions are not inferred.'))
    print('Task presentation:',len(tasks),'families;',len(variation),'orientations with repeat-run low-dose evidence')
    return 0


if __name__=='__main__':raise SystemExit(main())
