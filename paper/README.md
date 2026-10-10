# Current delivery scope — October10, 2026

The [24-hour paper sprint](../docs/PAPER_SPRINT_PLAN.md) stops additional training and
replication, retains completed evidence and five fixed seed17 explanatory diagnostics.
The collector generates source-backed diagnostic and general-ability summaries; the
detached controller rebuilds the draft as validated results arrive. The original larger
registration is retained as deferred scope. This is a reviewed draft target, not an
automatic conference submission or proof of a general causal mechanism.

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
- `planned_evaluations.tex` — registered protocols, figure/table slots, and conclusion placeholders
  for the dose/cost synthesis, general-ability controls, independent transfer, robustness,
  and anonymous reproducibility package. Included as an appendix after references.
- `acl.sty`, `acl_natbib.bst`, `refs.bib` — local style and bibliography dependencies.
- `numbers.tex`, `tables/`, `figures/` — generated results and assets; update them through
  their generators rather than editing measured values by hand.
- `Makefile`, `check_arr.py`, `page_limit.txt` — standalone build and checks.
- `PUBLICATION_ANALYSIS.json`, `PUBLICATION_PROVENANCE.json` — paired preservation
  frontiers, measured cost records, and completed paper-extension evidence.
- `artifact/` — local anonymous evidence bundle, original/redacted hash mapping, standalone
  hash verifier and strict-mean replay. Weights are excluded; external release remains separate.

See [ARGUMENT.md](ARGUMENT.md) for the argument and drafting decisions.

This is an evidence-backed working draft. The sprint fills the contribution, dose and probe
interpretation from audited evidence, and the collector reports diagnostic progress explicitly.
The strict format check passes; scientific review and remaining diagnostics are separate. Transfer, probes, prompts/templates and recipe sensitivity
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
superseded passes in the audit. Fill
measured values through the evidence generators; replace red `*-sentence` placeholders only
after the associated audit/analysis. The evidence refresher tracks the appendix so later edits
trigger the autonomous PDF rebuild. See [the execution plan](../docs/PAPER_FINISH_PLAN.md)
for campaign scope, validation, monitoring, and artifact limitations.
