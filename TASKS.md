# Experiment and task checklist

Updated October 10, 2026, 18:37 CDT. Background campaign reopened by user request, Amendment55.
The completed sprint draft is frozen at Git commit34de8d0; its
[completion checklist](archive/task_checklists/TASKS-2026-10-10-sprint-complete.md) and
[receipts](paper/SPRINT_COMPLETION.json) remain historical evidence.

## Active background queue

Resume all unfinished previously registered eligible experiments, including replications.
New-domain and contrastive work retains first priority. Completed adapters are reused;
training packs resume only missing arms. Engineering proofs, capability gates, quarantine,
bounded retries, same-pass base comparisons and null/adverse reporting remain enforced.
The detached Slurm controller and CPU-host renewal continue after logout.

- [x] Reopen the original campaign by disabling the completed sprint's admission guard.
- [ ] Finish the remaining19 registered explanatory production jobs across all three seeds,
      both modes and the mandatory units null control. The original panel remains24 jobs.
- [ ] Finish repaired-format Gemma12B contrastive seed17, then automatically admit42/1234
      only after a proof-valid seed17 evaluation, regardless of measured effect sign.
- [ ] Finish eight ready repaired-domain seed42/1234 evaluations: typed Python/C++ at
      Llama8B/Gemma12B, formatting contract at Llama8B and units contract at Gemma12B.
- [ ] Resume six missing invertibility-ladder training/evaluation packs at Llama8B,
      covering fmt_det75/50/25 and seeds42/1234; retain fully saved arms.
- [ ] Collect checked contrasts, preserve trials and refresh the working paper evidence.
- [x] Verify a complete healthy controller pass, detached stdin and CPU-host renewal.
      All five health checks and43 distinct targeted regression tests pass. Initial admission queues
      41 GPU jobs across34 experiment cells; Slurm priority/dependencies currently govern starts.

Live states: [mechanism scheduler](runs/feeder/mechanism_status.json),
[repair scheduler](runs/feeder/gate_repair_status.json), and
[controller status](runs/feeder/STATUS.md). See [background plan](docs/BACKGROUND_RUN_PLAN.md).
Failed capability gates, the failed D2T extractor oracle and unimplemented/unregistered
new task proposals remain blocked/design work; reopening admission does not make them
runnable or scientific successes. Catalogue-wide new task construction and new scorer
protocols require implementation and registration before GPU admission.

## Evidence already available

- [x] Main grid: 91 distinct audited model/task/seed cells across five models and eleven
      orientations, from103 valid campaigns.52 cells meet the descriptive reverse-loss threshold.
- [x] Original contrastive panel: 24/27 evaluations complete; three original fmt/Gemma12B
      cells remain failed-gate boundaries. All five primary comparisons remain reported.
- [x] Four repaired new-domain seed17 pilot evaluations:typed Python/C++ at Llama8B and
      Gemma12B, formatting contract at Llama8B, and units contract at Gemma12B.
      These variants reuse semantic corpora and are exploratory, not independent new domains.
- [x] Original units/Gemma12B and Python/C++ Llama8B campaigns already include three seeds.
      Preserve the unit-conversion null-collapse result and seed-dependent forward costs.
- [x] Six full-FT evaluations and the saved-checkpoint repair campaign.
- [x] Original paper-completion panel: 39/39 production jobs and13/13 engineering smokes.
      Preservation/cost summaries, probes, prompts, training recipes and transfer boundaries retained.
- [x] All six stricter OPUS transfer production evaluations; failed eligibility retained.
- [x] Cancel 37 out-of-scope jobs, including four active training packs. Completed and partial
      artifacts remain on disk; cancellation does not count as a scientific failure.
- [x] Enforce the exact sprint allowlist through delivery; Amendment55 explicitly reopens
      the registered background campaign. The completed sprint deadline is historical.

## Completed sprint GPU work: five production diagnostics

All use unchanged Amendment52 data, sample sizes, checkpoints and controls. Gemma HF uses
Amendment53 text-graph repair. Training seed17 is fixed throughout; no new model training.

- [x] Formatting/Llama3B:gold, copy and shuffled-target likelihood; gradient geometry and
      all held-out equal-norm local steps across the fixed SFT/replay/mix/CL/CE systems.
- [x] German→English/Llama3B:the same likelihood/gradient/local-step tests in a second domain.
- [x] Formatting/Llama3B:all four layer bands versus scaling and both random controls,
      with both directions and strict no-echo generation on512 pairs.
- [x] Both revised Gemma GPU smokes pass, including all gradient and local-step phases.
- [x] German→English/Gemma4B:full HF diagnostic 450906 completes for a second family.
- [x] Units/Gemma12B:full HF diagnostic 450907 completes, retaining the null control.

Live states:[mechanism scheduler](runs/feeder/mechanism_status.json).
Proof-valid counts:[paper diagnostic report](paper/MECHANISM_DIAGNOSTICS.json), sprint subsection.
The sprint snapshot contains five proof-valid production jobs. Amendment55 resumes the
 other19; the full registered production denominator remains24.

## Current manuscript revision — October 10

- [x] Center abstract, introduction and conclusion on forward gains/backward loss and
      small-dose reversed-pair SFT, with measured 1% formatting/code examples.
- [x] Add compact percentage main tables: all eleven reference task orientations and
      all twelve main-grid systems; all eight eligible original contrastive and five
      exposure-audited auxiliary-objective seed-17 cells, with both directions.
- [x] Highlight low-dose arms and descriptive SFT collapse; retain unresolved rungs,
      adverse dose effects, units nulls and loss-baseline wins.
- [x] Keep model/seed counts and paired uncertainty in the appendix/source audit.
- [x] Verify source integrity and controller regressions (32 targeted tests); rebuild
      the seven-page main draft, 168-word abstract, with all numbers resolved.

## Completed sprint CPU/paper delivery — commit34de8d0

- [x] Integrate all five proof-valid diagnostic findings, including the Gemma likelihood
      counterexample, units null, near-zero gradients and all layer-control outcomes.
      Source-generated tables report both total and per-token gold/copy rankings in both directions.
- [x] Tighten the argument around directional information, matched controls, and scoped
      generality. Do not claim a universal mechanism or independent-domain evidence from prompt variants.
- [x] Fill contribution, preservation and general-ability prose from audited results;
      add source-generated probe seed ranges and retain endpoint-specific adverse outcomes.
- [x] Rebuild and strictly check the final ACL draft after integrating the completed diagnostics:
      eight body pages, seventeen total, 183 abstract words, no visible placeholders or
      unresolved references, all fonts embedded.
- [x] Archive all current trials/manifests:zero outstanding files.
- [x] Refresh and verify the final anonymous bundle after all diagnostics; all packaged
      hashes verify and all 219 strict-mean replay reports complete.
- [x] Freeze [the sprint completion snapshot](paper/SPRINT_COMPLETION.json) before the
      deadline, with completed, failed, deferred and unresolved work explicit.
      A queue delay or failed smoke is reported as such, never filled with an invented finding.

## Still blocked or requiring implementation

- D2T extractor/scoring repair and tasks whose unchanged base gates failed.
- New independent task construction, shuffled reverse-CE correspondence follow-up and
  extra probes without a complete registered/tested worker and admission path.

The former sprint deferrals (registered seeds, contrastive packs and layer panels) now
belong to the active background queue above. Configs/paper_sprint.json remains disabled
under explicit user authorization. All experiment artifacts are retained. The immutable
sprint receipts describe delivery at34de8d0; subsequent results update the working evidence.

## Interpretation retained in this delivery

Reverse gold-over-copy preference deteriorates in the three collapse cells and improves
with mixed-direction SFT. Absolute reverse-answer loss instead improves in Gemma translation
despite collapsed generation. Mid-band formatting removals exceed matched controls, but the
earliest band harms forward performance. Local reverse updates also improve loss in the
units null cell; final gradient geometry does not establish persistent training conflict.
These are exploratory single-seed findings, with all controls retained, rather than a
universal mechanism or proof of stored knowledge.
