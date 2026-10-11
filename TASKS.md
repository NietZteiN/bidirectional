# Experiment and task checklist

Updated October 10, 2026. Amendment57 prioritizes calculating every missing table entry.
The table-completion controller is active; the 32 earlier experiment jobs remain held.
The completed sprint draft is frozen at Git commit34de8d0; its
[completion checklist](archive/task_checklists/TASKS-2026-10-10-sprint-complete.md) and
[receipts](paper/SPRINT_COMPLETION.json) remain historical evidence.

## Top priority: calculate the missing tables

- [x] Inventory every generated table, including legacy tables outside the PDF, and record
      gaps in [TABLE_COMPLETION_AUDIT.json](paper/TABLE_COMPLETION_AUDIT.json).
- [x] Calculate all fourteen mixed-task normalized changes from checked same-pass trials,
      including the six previously omitted null/adverse cells; label them descriptively.
- [x] Register seven seed17 campaigns covering twelve missing arm combinations and
      24 main-table forward/backward percentages. Queue first GPU checks for all seven.
- [ ] Evaluate saved relation/Llama3B mix1/mix5/mix10 adapters together with every
      displayed control and a fresh base; refresh both primary and legacy tables.
- [ ] Train execution/Llama3B mix1/mix5/mix10 after valid GPU checks, then evaluate
      all twelve main-grid arms together. Below-batch doses remain exploratory.
- [ ] Train and evaluate code/Llama3B unlikelihood and round-trip; translation/Llama3B
      and Llama8B round-trip; SQL/Llama3B and Llama8B unlikelihood. Keep same-pass controls.
- [x] Keep CFT code-specific, as requested. Mark its four translation/SQL combinations
      as not applicable; no cross-domain CFT variant is queued.
- [ ] Finish all eligible production table cells and verify the remaining-gap audit.
- [x] Run the detached controller in exclusive table mode after logout/renewal;
      skip broad scheduling and old shell-queue launches while this priority is active.
- [x] Pass 43 targeted regression tests and all five detached-controller health checks;
      rebuild the eight-body-page draft with resolved numeric macros and references.
      GPU checks are queued, not yet validated; 33 numeric table positions remain pending,
      including nine duplicate legacy positions.

See [table completion plan](docs/TABLE_COMPLETION_PLAN.md). Current job/phase progress:
[table scheduler](runs/feeder/table_completion_status.json). New results enter the paper
only after full paired coverage, adapter effectiveness, source hashes and successful status checks.

## Earlier experiment queue — remains held

- [x] Stop automatic scheduling and hold all 32 current experiment jobs, including four
      running training packs requeued into held state; verify zero running experiments.
- [x] Preserve completed results, dependencies and the shared CPU host; record the pause
      and resume procedure in [the pause plan](docs/EXPERIMENT_PAUSE.md).
- [x] Complete the remaining manuscript/reporting blanks using existing evidence;
      unmeasured rungs and ineligible comparisons remain explicitly unavailable.

The previous unfinished experiment jobs remain held. The user's clarification explicitly
resumes the table-completion work above; it takes priority over the earlier new-domain,
contrastive and mechanism campaign. Completed adapters are reused;
training packs resume only missing arms. Engineering proofs, capability gates, quarantine,
bounded retries, same-pass base comparisons and null/adverse reporting remain enforced.
The controller's exclusive table mode persists after logout and CPU-host renewal.

- [x] Reopen the original campaign by disabling the completed sprint's admission guard.
- [ ] Finish the remaining19 registered explanatory production jobs across all three seeds,
      both modes and the mandatory units null control. The original panel remains24 jobs.
- [ ] Finish repaired-format Gemma12B contrastive seed17, then automatically admit42/1234
      only after a proof-valid seed17 evaluation, regardless of measured effect sign.
- [x] Finish eight ready repaired-domain seed42/1234 evaluations: typed Python/C++ at
      Llama8B/Gemma12B, formatting contract at Llama8B and units contract at Gemma12B.
- [ ] Resume six missing invertibility-ladder training/evaluation packs at Llama8B,
      covering fmt_det75/50/25 and seeds42/1234; retain fully saved arms.
- [x] Collect completed checked contrasts, preserve trials and refresh the working paper evidence.
- [x] Verify a complete healthy controller pass, detached stdin and CPU-host renewal.
      All five health checks and43 distinct targeted regression tests pass. Initial admission queues
      41 GPU jobs across34 experiment cells at that historical checkpoint. Amendment56 now holds all remaining jobs.

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
- [x] All twelve repaired-domain pilot evaluations across three seeds: typed Python/C++ at
      Llama8B/Gemma12B, formatting contract at Llama8B and units contract at Gemma12B.
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
The sprint snapshot contains five proof-valid production jobs. The other19 registered
jobs are now held under Amendment56; the full registered production denominator remains24.

## Current manuscript revision — October 10

- [x] Integrate all twelve completed versioned-domain pilots, including all seeds,
      replay wins, mixture failures and forward costs; distinguish reused-corpus
      diagnostics and 50% mixtures from independent domains and low-dose results.
- [x] Add source-generated split sizes, actual reference training counts and shared
      recipe details; retain the historical/current code-corpus count discrepancy.
- [x] Replace stale pending interpretation and running/validation prose with completed
      findings or explicit unmeasured/ineligible status; remove the draft figure fallback.
- [x] Rebuild and visually inspect the manuscript and independently copied paper folder:
      eight body pages, twenty total, 175 abstract words, all numbers/references resolved
      and fonts embedded. All 34 targeted reporting/evidence regressions pass.
- [x] Verify local packaged hashes and replay all 227 retained trial reports;
      upstream licenses and missing historical training versions remain external-release requirements.
- [x] Build from the local ATTRIB/NeurIPS workshop's objective-versus-data-direction
      framing; extend the argument across fields, doses and explanatory diagnostics.
- [x] Add source-generated translation, semantic-parsing and algebra examples, and
      state the conditional-CE/output-selection hypothesis with counterevidence explicit.
- [x] Center abstract, introduction and conclusion on forward gains/backward loss and
      small-dose reversed-pair SFT, with measured 1% formatting/code examples.
- [x] Add compact percentage main tables: all eleven reference task orientations and
      all twelve main-grid systems; all eight eligible original contrastive and five
      exposure-audited auxiliary-objective seed-17 cells, with both directions.
- [x] Highlight low-dose arms and descriptive SFT collapse; retain unresolved rungs,
      adverse dose effects, units nulls and loss-baseline wins.
- [x] Keep model/seed counts and paired uncertainty in the appendix/source audit.
- [x] Verify source integrity and controller regressions (32 targeted tests); rebuild
      the seven-page main draft, 175-word abstract, with all numbers resolved.

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
belong to the paused experiment queue above. Configs/paper_sprint.json remains disabled
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
