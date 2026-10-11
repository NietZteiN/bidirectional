# Provenance for every number in the paper

*Started 2026-09-10. Empty until the gate runs — by design.*

Rule for this file, carried over from the workshop paper: **every figure that appears in the
paper is traceable to a file on disk that a recorded run produced.** Nothing is quoted from a
report or from memory. Where a report and a result file disagree, the file wins and the
discrepancy is noted here.

That rule is not bureaucracy. Writing the workshop paper turned up three numbers that had
drifted from their source — a `base` rate that differed between passes and was explained
wrongly in a first draft, budget ratios that were estimates carried forward as measurements,
and a corpus count quoted where a split count was meant. Each was caught by trying to fill in
this table and failing.

## Runs the paper draws on

| tag | model | what | trials | result dir |
|---|---|---|---|---|
| *(pending the decision gate)* | | | | |

## Section by section

| paper location | numbers | source |
|---|---|---|
| *(pending)* | | |

## Generated, not hand-typed

| artifact | generator | reads |
|---|---|---|
| `tables/main.tex` | `scripts/51_tables.py` | every `trials.jsonl` in the tier |
| `tables/dose.tex` | `scripts/51_tables.py` | same |
| `tables/ladder.tex` | `scripts/51_tables.py` | the `fmt*` runs |
| `figures/fig_dose.tex` | `scripts/52_figs.py` | same |
| `figures/fig_alpha.tex` | `scripts/52_figs.py` | `results/mech/alpha_scale/*/curve.json` |
| `tables/PROVENANCE.json` | `scripts/51_tables.py` | written on every generation |

Do not hand-edit any of those. Re-run the generator.

## Discrepancies resolved while writing

*(none yet)*

## Claims deliberately not made

*(to be filled as the drafting turns up boundaries — the workshop paper's list of these was
one of the more useful things in it, because it is what a reviewer presses on)*
