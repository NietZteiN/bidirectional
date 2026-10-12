#!/usr/bin/env python
"""Apply concise publication captions after table generation; preserve all table bodies."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
PAPER=ROOT/'paper/bidir'

CAPTIONS={
 'main-adaptations':r'Forward (F) and backward (B) strict success (\%), Llama 3.2-3B. Blue: reversal mixtures; red: SFT backward success $\leq$ half the base rate.',
 'main-loss-baselines':r'Preservation objectives: forward (F) and backward (B) strict success (\%). L/G: Llama/Gemma; sizes in billions. Methods: \S\ref{sec:attrib}. Colors follow Table~\ref{tab:main-adaptations}.',
 'full-adaptations':r'Full dose ladder and controls: strict success (\%), Llama 3.2-3B. F/B: forward/backward.',
 'task-definitions':r'Tasks, illustrative examples, scoring, and training pairs (Llama3B).',
 'auxiliary-intervals':r'Method minus reference strict success (percentage points), with 95\% paired bootstrap intervals (unadjusted).',
 'main-seed-variation':r'Minimum/maximum reversal-mixture minus SFT success across training runs (percentage points), Llama3B. F/B: forward/backward; subscripts: reversal percentages.',
 'preservation-paired':r'Reversal-mixture minus SFT success (percentage points), with 95\% paired cluster-bootstrap intervals (unadjusted).',
 'reference-methods':r'Corpus sizes and forward-SFT training counts.',
 'core-inventory':r'Experimental coverage. Severe loss: SFT backward success $\leq$ half the base rate.',
 'contrastive-snapshot':r'Contrastive learning minus directional SFT: backward-success changes (percentage points), with 99\% paired cluster-bootstrap intervals adjusted for five comparisons per task/model/run.',
 'contrastive-controls':r'Contrastive learning minus additional-CE or shuffled-pair controls: backward-success changes (percentage points), with multiplicity-adjusted 99\% paired cluster-bootstrap intervals.',
 'sprint-likelihood':r'Loss per token on correct answers (128 pairs). Differences have 95\% bootstrap intervals; negative values favor the first system.',
 'sprint-gradients':r'Forward/backward gradient cosine in active LoRA coordinates (16 pairs).',
 'sprint-candidates':r'Fraction of pairs where correct answers outrank copied inputs by total/per-token loss (128 pairs).',
 'sprint-layer-controls':r'Layer removal minus norm-matched controls: strict-success changes (percentage points), with 95\% paired intervals; formatting/Llama3B, 512 pairs.',
 'paper-finish-protocol':r'Additional evaluation coverage.',
 'preservation-summary':r'Representative forward/backward strict success (\%).',
 'preservation-costs':r'Training runtime, optimizer steps, and CE-corpus tokens before epoch repetition. Tokens exclude auxiliary exposure.',
 'publication-probe-ranges':r'Minimum/maximum general-ability changes across training runs (percentage points).',
 'transfer-boundaries':r'OPUS transfer: base-model capability and scoring checks.',
 'revised-domain-results':r'Revised-task strict success (\%); reversal uses 50\% replacement. F/B: forward/backward; red: severe backward loss; underlined: SFT gains in both directions.',
 'publication-robust-ranges':r'Prompt/template sensitivity: minimum/maximum success changes across runs (percentage points). S/B/M: SFT/base/mixture; F/R: forward/backward.',
 'publication-recipe-ranges':r'Training-recipe sensitivity: minimum/maximum success changes across runs (percentage points). S/B/M: SFT/base/mixture; F/R: forward/backward.',
 'initial-failure-reasons':r'Reasons for excluding initial task extensions.',
 'domain-gates-snapshot':r'Initial base-model forward/backward strict success (\%).',
 'generality-snapshot':r'Larger-model forward/backward strict success (\%).',
 'fullft-snapshot':r'Full-weight fine-tuning: forward/backward success (\%), Llama3B.',
 'paired-judgment':r'Paired-judgment baselines: forward (F) and backward (B) strict success (\%). PJ-5 uses 5\% judgment targets.',
 'paired-judgment-intervals':r'Method minus reference strict success (percentage points), with 95\% paired cluster-bootstrap intervals (unadjusted).',
 'translation-thresholds':r'Frozen COMET-22 success thresholds.',
 'parser-calibration':r'Frozen SQL parser accuracy on reference-question subsets.',
 'main':r'Forward ($\rightarrow$) and backward ($\leftarrow$) strict success (\%) by task and model.',
 'dose':r'Backward strict success (\%) by reversed-pair percentage.',
 'mixture':r'Mixed-task SFT: backward strict success (\%). Normalized change: $100(\mathrm{mixedtask}-\mathrm{sft})/(\mathrm{base}-\mathrm{sft})$.',
 'publication_probes':r'General-ability success changes (percentage points), with 95\% paired cluster-bootstrap intervals.',
 'publication_robust':r'Prompt/template success changes (percentage points), with 95\% paired cluster-bootstrap intervals.',
 'publication_recipe':r'Training-recipe success changes (percentage points), with 95\% paired cluster-bootstrap intervals.',
}


def spans(text, fallback=None):
    result=[]
    for match in re.finditer(r'\\caption\{',text):
        start=match.end();depth=1;i=start
        while i<len(text) and depth:
            if text[i]=='\\':i+=2;continue
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            i+=1
        if depth:raise ValueError('Unclosed caption')
        label=re.match(r'\s*\\label\{tab:([^{}]+)\}',text[i:])
        key=label[1] if label else fallback
        if key in CAPTIONS:result.append((start,i-1,key))
    return result


def stripped(text, fallback=None):
    for start,end,_ in reversed(spans(text,fallback)):text=text[:start]+text[end:]
    return text


def main():
    tables={};labels=[]
    paths=sorted((PAPER/'tables').glob('*.tex'))
    if (PAPER/'task_methods.tex').exists():paths.append(PAPER/'task_methods.tex')
    for path in paths:
        fallback=path.stem if path.stem in ['publication_probes','publication_robust','publication_recipe'] else None
        original=path.read_text();text=original;found=spans(text,fallback)
        if not found:continue
        for start,end,label in reversed(found):text=text[:start]+CAPTIONS[label]+text[end:]
        if stripped(original,fallback)!=stripped(text,fallback):raise ValueError('Caption edit changed table body: '+str(path))
        if text!=original:path.write_text(text)
        tables[str(path)]=hashlib.sha256(text.encode()).hexdigest();labels += [s[2] for s in found]
    atomic_json(PAPER/'PUBLICATION_CAPTIONS.json',dict(
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        table_sha256=tables,labels=labels))
    print(f'Publication captions: {len(labels)}; all table bodies preserved.',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
