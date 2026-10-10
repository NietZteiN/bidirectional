# Experiment and task checklist

Updated October 10, 2026, 18:23 CDT. All five diagnostics complete and proof-valid.
Active scope: user-directed 24-hour paper sprint, Amendment54.
The full earlier checklist is preserved in [the pre-sprint snapshot](archive/task_checklists/TASKS-2026-10-10-before-sprint.md).

## Current priority: distinct evidence and paper integration

No additional training seeds or replication campaigns. Reuse completed new-domain,
contrastive, preservation, robustness and probe evidence. Five fixed seed-17 explanation
jobs address different questions; required Gemma engineering smokes passed before production.
Unrun work is deferred, not relabelled complete. Completed adverse, null and failed-gate
results remain in the paper. See [the sprint plan](docs/PAPER_SPRINT_PLAN.md).

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
- [x] Enforce an exact job allowlist in all automatic planners and the central submitter.
      Replications cannot be silently requeued. Admission stops at the sprint deadline.

## Completed GPU work: five production diagnostics

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
The original24-job denominator remains historical; the active scope has five production jobs.

## CPU/paper delivery

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

## Deferred beyond this sprint

- Seeds42/1234 for repaired domains, the invertibility ladder and explanatory diagnostics.
- Unfinished repaired-format Gemma12B contrastive pilot and both future replication packs.
- Translation/Gemma and units layer-generation panels; translation/Llama layer panel.
- D2T extractor/scoring repair, remaining failed-gate tasks, new task construction,
      shuffled reverse-CE follow-up, and extra probes.

Disabling or editing configs/paper_sprint.json is an explicit future scope change. No
experiment artifacts are deleted by this sprint. The detached controller, analysis,
archiving and paper snapshots continue after logout.

## Interpretation retained in this delivery

Reverse gold-over-copy preference deteriorates in the three collapse cells and improves
with mixed-direction SFT. Absolute reverse-answer loss instead improves in Gemma translation
despite collapsed generation. Mid-band formatting removals exceed matched controls, but the
earliest band harms forward performance. Local reverse updates also improve loss in the
units null cell; final gradient geometry does not establish persistent training conflict.
These are exploratory single-seed findings, with all controls retained, rather than a
universal mechanism or proof of stored knowledge.
