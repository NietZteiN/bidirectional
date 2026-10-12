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
 'main-adaptations':r'Forward (F) and backward (B) strict success (\%) on Llama 3.2-3B (seed 17). Blue: reversal mixtures; red: SFT backward success at most half the base rate.',
 'main-loss-baselines':r'Preservation objectives: forward (F) and backward (B) strict success (\%), seed 17. CL-F/M: contrastive alignment with forward/mixed CE; CE-F+/M+: additional CE; Shuffle: shuffled pairs. L/G: Llama/Gemma; suffix: billions of parameters. CFT is code-specific (n/a elsewhere). Colors follow Table~\ref{tab:main-adaptations}.',
 'full-adaptations':r'Full reversal-dose ladder and controls: forward (F) and backward (B) strict success (\%), Llama 3.2-3B, seed 17. Control definitions appear in Appendix~\ref{app:dose-ladder}.',
 'task-definitions':r'Task definitions, illustrative examples, scoring, and retained training pairs (Llama3B). Data and scoring details: Appendix~\ref{app:task-scoring}.',
 'auxiliary-intervals':r'Objective differences in strict success (percentage points), with 95\% paired bootstrap intervals. Positive values favor the first method; intervals are unadjusted.',
 'main-seed-variation':r'Ranges across training runs of reversal-mixture minus SFT success (percentage points), Llama3B. F/B: forward/backward; subscripts: reversal proportions. Ranges are not confidence intervals.',
 'preservation-paired':r'Reversal-mixture minus SFT success (percentage points), with 95\% paired cluster-bootstrap intervals (2,000 draws; unadjusted).',
 'reference-methods':r'Paired-corpus sizes and recorded reference training counts. Retained rows, optimizer steps, and batch sizes describe the actual SFT runs.',
 'core-inventory':r'Model--task--training-run coverage. Severe loss: SFT backward success at most half the base rate. Counts are descriptive.',
 'contrastive-snapshot':r'Contrastive learning minus directional SFT: backward-success changes (percentage points), with 99\% paired cluster-bootstrap intervals. Five primary contrasts are adjusted within each task/model/run.',
 'contrastive-controls':r'Contrastive learning versus additional-CE and shuffled-pair controls: backward-success changes (percentage points), with 99\% paired cluster-bootstrap intervals adjusted within each task/model/run.',
 'sprint-likelihood':r'Gold-answer negative log-likelihood per token on 128 pairs (seed 17). Differences have descriptive 95\% bootstrap intervals; negative values favor the first system.',
 'sprint-gradients':r'Forward/backward validation-gradient cosine on 16 pairs in LoRA coordinates. Inactive Gemma vision factors are excluded.',
 'sprint-candidates':r'Fraction of 128 pairs where the gold answer outranks the copied input by total/per-token negative log-likelihood (seed 17).',
 'sprint-layer-controls':r'Layer-removal effects relative to controls matched on retained norm: strict-success differences (percentage points), with descriptive 95\% paired intervals; formatting/Llama3B, 512 pairs, seed 17.',
 'paper-finish-protocol':r'Additional evaluation coverage by task/model, training seeds, and runs.',
 'preservation-summary':r'Representative preservation results: forward and backward strict success (\%), seed 17.',
 'preservation-costs':r'Training runtime (minutes), optimizer steps, and CE-corpus sequence tokens before epoch repetition (seed 17). Token counts exclude auxiliary encoded-side exposure.',
 'publication-probe-ranges':r'General-ability changes in strict success (percentage points): minimum/maximum across training runs. Ranges are not confidence intervals.',
 'transfer-boundaries':r'OPUS transfer: base-model capability and scoring-validity checks under the original thresholds. Failed checks exclude tuned comparisons.',
 'revised-domain-results':r'Revised-output task results: forward (F) and backward (B) strict success (\%). Reversal uses 50\% replacement. Red: SFT backward success at most half the base rate; underlined: SFT gains in both directions.',
 'publication-robust-ranges':r'Prompt/template sensitivity: ranges of strict-success changes across training runs (percentage points). S/B/M: forward SFT/base/reversal mixture; F/R: forward/backward. Mixtures use 5\% for translation/format and 50\% for units/Python--C++. Ranges are not confidence intervals.',
 'publication-recipe-ranges':r'Training-recipe sensitivity: ranges of strict-success changes across training runs (percentage points). S/B/M: forward SFT/base/reversal mixture; F/R: forward/backward. Ranges are not confidence intervals.',
 'initial-failure-reasons':r'Reasons for excluding the initial task extensions.',
 'domain-gates-snapshot':r'Initial task eligibility: base-model forward and backward strict success (\%).',
 'generality-snapshot':r'Larger-model comparisons: forward and backward strict success (\%), with methods evaluated on the same examples.',
 'fullft-snapshot':r'Full-weight fine-tuning: forward and backward success (\%), Llama3B. Comparisons span engine restarts.',
 'paired-judgment':r'Paired-judgment baseline (PJ-5): forward (F) and backward (B) strict success (\%), seed 17. PJ-5 replaces 5\% of generation targets with Yes/No judgments; shuffled labels use identical inputs.',
 'paired-judgment-intervals':r'Paired-judgment differences in strict success (percentage points), with 95\% paired cluster-bootstrap intervals (2,000 draws; unadjusted). Positive values favor the first method.',
}


def spans(text):
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
        if label and label[1] in CAPTIONS:result.append((start,i-1,label[1]))
    return result


def stripped(text):
    for start,end,_ in reversed(spans(text)):text=text[:start]+text[end:]
    return text


def main():
    tables={};labels=[]
    for path in sorted((PAPER/'tables').glob('*.tex')):
        original=path.read_text();text=original;found=spans(text)
        if not found:continue
        for start,end,label in reversed(found):text=text[:start]+CAPTIONS[label]+text[end:]
        if stripped(original)!=stripped(text):raise ValueError('Caption edit changed table body: '+str(path))
        if text!=original:path.write_text(text)
        tables[str(path)]=hashlib.sha256(text.encode()).hexdigest();labels += [s[2] for s in found]
    atomic_json(PAPER/'PUBLICATION_CAPTIONS.json',dict(
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        table_sha256=tables,labels=labels))
    print(f'Publication captions: {len(labels)}; all table bodies preserved.',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
