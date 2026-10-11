# Task-first manuscript revision — October 10, 2026

The manuscript now leads with the experiment: what each input contains, the requested
forward/backward output, its data source, and the actual binary correctness predicate.
The central argument is directional degradation, preservation with small reversed-pair
mixtures, and what specialized objectives add beyond that supervision. Diagnostics follow
the objective comparison as supporting explanation.

- Added a seven-family task table with schematic examples and retained training sizes.
- Added a consolidated task appendix with all main prompt templates, source/split details,
  realized integer doses, parser rules, frozen model/direction COMET thresholds, and SQL
  parser calibration. Unrecorded historical upstream revisions remain explicitly unknown.
- Reduced the main comparison to base, forward SFT, 1% reversal and 5% reversal, preserving
  all eleven orientations. The full twelve-arm grid moved to the appendix.
- Replaced the crowded frontier figure with four tasks and four systems, including marginal
  cluster-bootstrap intervals. Added paired contrasts and training-run ranges for every
  orientation with repeated complete low-dose measurements; no outcome-based subset selection.
- Put repeat-run formatting/code benefits and the repeated adverse algebra result in the
  main text. Added all measured auxiliary-objective paired intervals, including small/null gains.
- Removed the reporting checklist, queue/admission narrative, sprint instructions and internal
  bibliography reading notes. Detailed experimental selection/exclusion rules remain methods.
- Corrected the swapped algebra orientation labels and the two missing citation author records.

This is a presentation and analysis revision. Training, paired corpora, prompts, scoring
predicates and capability requirements are unchanged. No new GPU result is inferred.
The seven missing-table campaigns continue in exclusive mode; earlier jobs stay held.

The descriptions deliberately expose scorer limits: execution does not require code-style
or original-name recovery; YAML parsing need not enforce YAML-exclusive syntax; scalar
format comparisons normalize case; expansion has no separate expanded-form conjunct;
factorization need not be irreducible; factual exact matching can reject aliases. SQL
verbalization uses a frozen model, whereas the proposed WebNLG text scorer failed on
references and provides no tuned comparison. These are not retrospectively strengthened
criteria. Original annotations remain in `papers/references.bib` and Git history.

`scripts/122_task_presentation.py` generates the task appendix, presentation uncertainty,
training-run ranges and source metadata. The regular paper-refresh pipeline runs it before
numeric resolution and compilation. Targeted regression checks cover filtered reverse-pair
counts, paired cluster resampling, completeness and the algebra orientation labels.

Validation: 48 targeted tests pass. The rebuilt PDF has seven body pages and 26 total
pages, a 165-word abstract, embedded fonts, and resolved numeric macros and references.
All five background-pipeline health checks pass. Missing experiment cells remain explicit
dashes until their queued measurements complete; the format check does not certify their
completion.

## Focused narrative revision

The second editorial pass keeps the task/scoring explanations and section order. The
abstract states the seven-family scope and one measured formatting result. The introduction
ends on the empirical contribution; compute accounting and the replacement objective sit
in setup. Section 5 progresses from large low-dose gains to other tasks and the adverse
algebra orientation, with the knee criterion beside the appendix dose ladder. Section 6
first explains why reverse-mapping exposure can confound attribution, then describes the
methods and controls. Section 7 leads with the Gemma likelihood/generation discrepancy,
followed by the output-selection hypothesis and supporting diagnostics. SFT and CE are
defined at first use, and unit conversion is identified as a supplementary boundary task.
These changes do not alter experiments, scoring, exclusions, or measured results.
The rebuilt draft has seven body pages, 25 total pages, and a 126-word abstract;
strict PDF checks pass with resolved numbers/references and embedded fonts.
