> Current scope, October 10: Amendment57 calculates the missing numerical table entries first.
> The exclusive table-completion controller is active; the 32 earlier jobs remain held.
> See [table completion plan](../docs/TABLE_COMPLETION_PLAN.md) and
> [remaining table gaps](TABLE_COMPLETION_AUDIT.json).
> Amendment 55's resumed campaign and the completed sprint remain historical records.

# Current paper story — October 10, 2026

Forward-only SFT can improve the trained task while damaging its backward counterpart.
The practical intervention is to reverse a small share of existing pairs and keep ordinary
cross-entropy. Lead with the fixed-seed formatting and code examples: 1% reversal gives
substantial backward recovery with little forward change. Use source-generated numbers;
do not claim a universal 1% optimum or universal superiority over auxiliary losses.

The framing builds from our local [ATTRIB/NeurIPS workshop draft](../papers/theflipflip_workshop2026.tex):
objective changes and data direction must be separated. The extension now makes three
questions explicit: transfer across task families, the required backward dose, and why
backward examples can help. The main text reports source-generated translation, SQL and
algebra examples alongside code/formatting and connects candidate-ranking and matched
layer interventions to a scoped output-selection hypothesis. See the workshop-framing
section of [ARGUMENT.md](ARGUMENT.md) for the provenance and limits of this extension.

The main presentation now has two percentage tables:

- Preservation table: all eleven Llama3B reference task orientations, forward/backward rows,
  base, forward SFT, 1% reversal and 5% reversal. The full ladder and controls are in the appendix. Blue identifies low
  doses; red identifies SFT backward rates at or below half of the same-pass base.
- Objective table: every eligible original reference contrastive task/model comparison and every measured
  seed-17 auxiliary-objective cell, with both directions and fresh within-pass baselines.
  Preserve mixed-direction contrastive gains, round-trip tradeoffs and the units null.

A task-definition table comes before results. `task_methods.tex` consolidates datasets,
splits, exact prompts, actual predicates, model/direction translation thresholds and
round-trip checks. `TASK_METHODS.json` records source hashes, realized doses, paired
uncertainty and repeat-run variation. The simpler four-system figure and repeat-run findings
support the central preservation comparison. See [revision details](../docs/PAPER_REVIEW_REVISION.md).

Highlighting is descriptive. Missing dose rungs are dashes. Model/seed coverage and paired
intervals remain in the appendix and source audit. Memorization or memory erasure is not
established by failed backward generation. Mechanism evidence motivates selective
interference in one cell; it is supporting diagnosis, not the central practical claim.
[MAIN_RESULTS.json](MAIN_RESULTS.json) records every displayed campaign and source hash;
`scripts/117_main_results.py` regenerates both main tables before the paper builds.
The earlier campaign stays held under Amendment56; Amendment57 authorizes and prioritizes
the missing-table campaigns. Earlier sprint scope below is historical.

# Current delivery scope — October10, 2026

The completed [24-hour paper sprint](../docs/PAPER_SPRINT_PLAN.md) retained existing
evidence and five fixed seed17 explanatory diagnostics. Amendment57 adds the missing
table experiments through an exclusive detached controller, with GPU checks and fresh
same-pass baselines. The previous larger queue remains held.
The original larger registration remains visible as unfinished scope. This is a reviewed draft target, not an
automatic conference submission or proof of a general causal mechanism.

All five selected diagnostics and both revised Gemma smokes are complete and proof-valid.
The draft integrates likelihood, gold/copy ranking, gradient and matched layer-control
results, including the Gemma counterexample and units null. Final delivery receipts live
in [SPRINT_COMPLETION.json](SPRINT_COMPLETION.json); the broader unrun panel stays deferred.

# ACL ARR LaTeX folder

**Directional Collapse in Fine-Tuning: Preserving the Way Back** — an ARR **long paper**, with a **8-page main-content limit**.

`main.tex` uses the standard 11pt article class and `\usepackage[review]{acl}`:
anonymous authors, line numbers, page numbers, A4 paper and two columns throughout,
including appendices. Wide figures and tables use full-width floats. Limitations follow
the conclusion, before references; appendices follow references. The abstract is limited
to 200 words. No template margins, fonts or vertical spacing are overridden.

The requirements were checked against the [ARR call for papers](https://aclrollingreview.org/cfp)
and [ACL formatting guidelines](https://acl-org.github.io/ACLPUB/formatting.html) on 2026-10-02.
`acl.sty` and `acl_natbib.bst` match the [official ACL files](https://github.com/acl-org/acl-style-files)
byte for byte; their SHA-256 hashes are recorded in `check_arr.py`.

## Files

- `main.tex`, `main.pdf` — manuscript and compiled review draft.
- `planned_evaluations.tex` — completed findings and registered protocols for dose/cost,
  general ability, independent-transfer boundaries, robustness and local reproducibility;
  reference methods and all completed versioned-domain pilots. Included after references.
- `acl.sty`, `acl_natbib.bst`, `refs.bib` — local style and bibliography dependencies.
- `numbers.tex`, `tables/`, `figures/` — generated results and assets; update them through
  their generators rather than editing measured values by hand.
- `Makefile`, `check_arr.py`, `page_limit.txt` — standalone build and checks.
- `PUBLICATION_ANALYSIS.json`, `PUBLICATION_PROVENANCE.json` — paired preservation
  frontiers, measured cost records, and completed paper-extension evidence.
- `MANUSCRIPT_COMPLETION.json` — all versioned-domain pilot campaigns, verified corpus
  counts and recorded reference-training metadata, with source hashes. Regenerated by
  `scripts/118_manuscript_completion.py` during the evidence-backed build.
- `TABLE_COMPLETION_AUDIT.json` — every missing numerical table position, required
  campaign arms, code-specific CFT boundaries and calculated mixed-task normalizations.
- `artifact/` — local anonymous evidence bundle, original/redacted hash mapping, standalone
  hash verifier and strict-mean replay. Weights are excluded; external release remains separate.

See [ARGUMENT.md](ARGUMENT.md) for the argument and drafting decisions.

This is an evidence-backed working draft. Contribution, dose and probe interpretation
are integrated from audited evidence; unmeasured diagnostics supply no claimed findings.
The strict format check passes; scientific review and deferred diagnostics are separate. Transfer, probes, prompts/templates and recipe sensitivity
were separately registered in Amendment 50 and admitted to the autonomous smoke-gated queue.
Pending allocation is not a successful GPU test. `refs.bib` is a local snapshot of
`../papers/references.bib`; keep it current when adding citations. Older ACM files under
`Router_Merger (1)/` are reference material and are not inputs to this draft.

## Build and check

From the repository root, using the existing cluster environment:

```bash
make paper             # compile and check formatting; errors propagate
make paper-check       # check the existing PDF
make paper-submission  # also reject unfilled numbers, missing assets and unresolved references
```

The folder also builds independently. With Tectonic and Python with `pypdf` installed:

```bash
cd paper
make TECTONIC=/path/to/tectonic PY=/path/to/python
make submission TECTONIC=/path/to/tectonic PY=/path/to/python
```

For Overleaf, upload the manuscript, `planned_evaluations.tex`, local style/bibliography files, `numbers.tex`,
`tables/` and `figures/`, and select `main.tex` as the main document. All manuscript
inputs live inside this folder.

The automated checks verify the official styles, review mode, section order, page budget,
abstract length, A4 portrait pages and embedded fonts, including fonts in figures.
They report draft placeholders separately; `--strict` makes these a failure. Visual
legibility, citation accuracy and scientific completeness still require review.

Story revised October 3 around directional loss, economical preservation, new-domain generality,
contrastive attribution, and recovery evidence. October 8 additions now include the audited
preservation figure, available cost records, and frozen evaluation/reporting protocols. The
core inventory selects the latest valid campaign per model/task/training seed, retaining
superseded passes in the audit. Measured values come from evidence generators; missing
dose rungs remain explicit dashes. The evidence refresher tracks manuscript and method
sources. Run `python scripts/99_paper_evidence.py --build` for a checked CPU-only refresh
during the pause. See [the execution plan](../docs/PAPER_FINISH_PLAN.md)
for campaign scope, validation, monitoring, and artifact limitations.
