# Experiment and task checklist

Updated 2026-10-09 13:32 CDT, reconciled against adapter manifests, result files,
`95_runner.py --status`, Slurm, and `runs/feeder/STATUS.md`.
Previous checklist: [September 14 snapshot](archive/task_checklists/TASKS-2026-09-14.md).

Completed campaign counts below are reconciled against complete adapter manifests, successful
evaluation status, summaries and paired analyses. They do not certify final scientific
interpretation or publication readiness. Mechanism artifact counts are inventories, not claims.

## Current priority order — user decision, October 3

**New domains and contrastive baselines are the first priority.** Implement, validate,
preregister, gate, and run their pilots before starting additional legacy experiments,
full-FT evaluation/repair, d2t repairs, or paper integration. Build the two tracks alongside
one another where their dependencies allow. Existing submitted jobs continue; no running
or queued experiment is cancelled by this reprioritization.

1. **Priority 1A:** Python ↔ C++, Boolean formula ↔ truth table, and unit conversion.
2. **Priority 1B:** paired contrastive objectives and their matched baselines/ablations.
3. Finish and audit existing experiment gaps, then refresh paper analyses and the ARR draft.

The feeder handles the existing campaign and gate-dependent new-domain pilots. All six
original small-model domain gates failed. All six explicit-format diagnostics also failed;
Gemma12B passes the original unit-conversion gate and all three seed campaigns are complete.
Llama8B passes Python/C++ translation; all three seed campaigns are complete.
Contrastive phase1 has twelve of twelve complete evaluations. Larger-model/new-domain
contrastive extensions and validity-based unit-conversion replications are first priority.
Original failures and the retained null-collapse pilot are preserved.

## Live to-do checklist — October 9 afternoon

The controller is healthy: fresh heartbeat, all six stages exit0, no active operational
alarms or quarantined cells, and CPU-host renewal remains queued. Four existing H200 GPUs
train contrastive replications for translation/Gemma12B and format/Llama8B.
All39 original publication production jobs and13/13 smokes are complete, including all
nine probe campaigns. Units/Gemma12B contrastive replication is complete at all three seeds.
Versioned blocked-cell diagnostics are newly registered and enter automatic admission below.
The artificial GPU cap remains disabled.

- [x] Original contrastive campaign:12/12 complete, including all five primary contrasts.
- [x] Main grid:91 distinct audited model/task/seed cells across five models and eleven orientations,
      from103 validated evaluation campaigns. Latest valid completion time selects repeated
      passes;12 earlier campaigns remain in the audit.52 latest cells meet the descriptive
      reverse-loss threshold. Earlier103/58 snapshots counted campaigns,not unique cells.
- [x] Full-FT evaluation:6/6 complete; paired repair evaluation complete.
- [x] Corrected unlikelihood:both registered Llama3B seeds complete and exposure verified.
- [x] Units/Gemma12B:seeds17,42,1234 complete, all five systems in each fresh pass.
      SFT reverse success70.0%,73.7%,72.7% versus same-pass base35.8% for each seed;
      forward96.9–97.4%. Preserve this replicated null-collapse result.
- [x] All eight larger-model CL/CE engineering profiles pass. Validated microbatch4 is
      active in production, with effective batch64 preserved; both control and objective
      improve runtime over microbatch1 and fit the measured memory requirement.
- [x] All A30 trainer/evaluation/SVD checks complete, including matching proof validation
      for the full ten-system Gemma campaign and largest projection spectrum.
- [x] Python/C++ Llama8B seed17 training448716 and fresh evaluation448717 complete;
      validated paired analysis admits seed42/1234 replication independent of effect sign.
      Replications449074→449075 and449076→449077 are complete.
- [x] Finish eligible larger-model contrastive seed17 training/evaluation:
      de→en/Llama8B448932→448933;de→en/Gemma12B448937→448938;
      units/Gemma12B448939→448940. All three pilots and fmt/Llama8B449065→449066 are complete.
- [x] Finish fmt feasibility gates449015 (Llama8B) and449016 (Gemma12B), using
      unchanged1000-row criteria. Llama8B passes;pack449065→449066 is complete.
      Gemma12B fails with forward format-failure26.1%;tuned runs remain blocked.
- [ ] Replicate valid larger-model contrastive cells at seeds42/1234 automatically.
      Original phase12/12 complete;extension8/15 evaluations complete. Translation/Llama8B
      and units/Gemma12B are complete across all three seeds. Four eligible replication packs run with
      dependent evaluations queued;fmt/Gemma12B and its two replications remain gate-blocked.
- [x] Replicate Python/C++ at seeds42/1234 once its seed17 pilot passes validity checks,
      independently of a favorable, adverse or null outcome.
- [ ] Finish mechanism interpretation, joint forward/reverse contrasts, seed ranges,
      attribution audit reconciliation and final paper/provenance checks after results arrive.
- [ ] Blocked:original small-model generality cells,failed larger-model logic/units cells,
      D2T frozen-extractor scoring, and failed-gate invertibility/coverage experiments.
      They receive no tuned runs under the current criteria.

## First-priority blocked-cell repair extension — Amendment51

This is an exploratory extension after inspecting failures. Original failed gates stay
unchanged; see [docs/GATE_REPAIR_PLAN.md](docs/GATE_REPAIR_PLAN.md).

- [x] Register four isolated repairs: signed C++ literals, serialization contract, exact
      unit-conversion contract and Boolean assignment/minterm contract. Prepare eval-disjoint splits.
- [x] Compute-local40-case gold/echo/empty/garbage oracles pass exactly in both directions
      for all four new cells. Valid observed C++ literal output now compiles and matches all33 inputs;
      unsafe syntax, wrong behavior and overflow remain rejected.
- [x] Implement source/data/trial/status admission proofs and bounded watchdog quarantine.
      Automatic passing-cell pilots and42/1234 replication depend on validity, not effect sign.
- [x] Deploy tested repair admission:390 regression tests pass;182 data/repair checks pass.
      Eleven GPU diagnostics submitted (450183–450193), currently awaiting shared cluster
      allocation. Four existing contrastive packs keep running;artificial caps remain disabled.
      The detached controller and renewal checks pass after deployment.
- [x] Extend verified spare-H100 QoS fallback to repair diagnostics and prevent status-only
      timestamps from rebuilding the4.7 GB review bundle. All18 routing/bundle checks pass.
- [x] Register versioned Gemma12B format contrastive runs and isolated training-only negative
      alias with checked identical paired content; no competing pilot writes the same adapters.
- [ ] Complete four model gate packs, covering16 repaired domain/model combinations.
- [ ] Complete unchanged legacy feasibility gates: six domains at each of Llama8B/Gemma12B.
- [ ] Complete WebNLG train-schema extractor oracle; admit Llama3B base gate only on success.
- [x] Complete the two revised Gemma12B format objective/control GPU trainer smokes
      (450190/450191): both completed two real training steps with proof-valid receipts.
      Production still requires the model/domain feasibility gate.
- [x] Complete stricter OPUS engineering smokes (450192/450193): both pass eligibility,
      paired coverage and adapter-effectiveness checks. Smoke outcomes are not production findings.
- [x] Queue all six stricter OPUS production campaigns (450425–450430), three seeds per model.
- [ ] Complete the six fresh OPUS production campaigns; retain failed new gates as boundaries
      and any eligible same-pass tuned comparisons separately.
- [ ] Run every admitted new/legacy domain pilot, then two valid replications; failed gates
      remain blocked. Four-arm pilots use base/sft/rev/mix50/replay; Gemma12B format uses the
      existing ten-system contrastive panel at all three seeds.
- [ ] Integrate new outcomes after validity/paired analysis, including adverse/null results.
      Existing39-job publication completion does not certify this new conditional work complete.

## Publication priorities — EACL/NAACL assessment, October 8

These publication additions now have a registered automatic experiment queue.
Current new-domain/contrastive jobs keep their existing first priority.
User authorized queueing/testing all draft additions. Amendment50 now freezes the panel;
its executable scope and validation are in [docs/PAPER_FINISH_PLAN.md](docs/PAPER_FINISH_PLAN.md).
Official review criteria emphasize supported claims, impact and reproducibility:
[ARR review form](https://aclrollingreview.org/reviewform).

- [x] Draft these publication additions in the manuscript: main preservation figure slot,
      matched general-ability controls, independent real-data transfer, fixed prompt/recipe
      robustness, sharper contribution framing, and anonymous reproducibility plan.
      See [paper/planned_evaluations.tex](paper/planned_evaluations.tex). Red placeholders
      reserve pending conclusions. This completes drafting only; the analysis/evaluation
      tasks below remain open until measurements/interpretation finish. Their protocols are
      now registered and implemented in the autonomous smoke-gated planner.

- [x] Freeze corpus revisions and overlap rules;retain541 IFEval,1319 GSM8K,1688 OPUS pairs.
      Full compute-node gold oracles pass for1000 magnitude and1000 piecewise-quadratic cases.
- [x] Implement idempotent submission/dependencies, isolated recipe outputs, same-campaign
      coverage/effectiveness checks, current source/data hashes, paired analysis, and archiving.
- [x] CPU scorer/data preflight and15 targeted tests pass;GPU smokes still require allocation.
- [x] All thirteen allocation-local smokes pass,including the units/CL probe smoke.
- [x] Complete39 production jobs after matching smokes:9 probe,6 OPUS-transfer,
      12 prompt/template and6 recipe-evaluation campaigns,plus6 training packs/48 adapters.
- [x] Source-backed preservation figure/cost synthesis added;deduplicate rerun campaigns.
- [x] Detached controller debug checks pass;GPU caps remain disabled and P1 work continues.
- [x] Acceleration deployment:widen seven pending engineering smokes and four dependent
      contrastive evaluations across verified/eligible GPU partitions;19 targeted checks pass.
      Future submissions retain the wider routing. Shared-queue start predictions remain tentative.
      See `runs/feeder/paper_finish_acceleration_20261008.json` for original specifications/updates.
- [x] Initial local anonymous evidence bundle built;packaged hashes verified and strict means replayed.
      Final refresh and release/reconstruction audit remain pending.
- [x] Afternoon continue pass:39/39 publication production jobs complete;13/13 engineering smokes pass.
      All12 prompt/template campaigns and all nine probes are complete.
      Six OPUS campaigns close as failed echo-validity gates;retain base-only trials and block tuned comparisons.
      All six recipe packs and their six evaluations are complete,covering48 new recipe adapters.
      Both Python/C++ replication seeds42/1234 have completed training/evaluation.
- [x] Repair probe-panel lookup in the paper updater;31 validated evaluation campaigns now enter synthesis.
      Add regression coverage and an idle-H100 fallback for ready jobs blocked by the eight-job pri QoS.
      The fallback requires a physically free GPU,CPU/memory headroom,no dependency,and owned pending work.
      All18 targeted checks and detached-controller debug checks pass.
- [ ] Finalize scientific claims after completed panels;draft remains visibly incomplete.
- [x] Integrate completed prompt/template and recipe panels as generated within-campaign
      changes with descriptive seed ranges. Preserve adverse Python/C++ forward costs and
      recipe-dependent units reverse loss;the original units null-collapse claim is recipe-scoped.
      Fill the completed robustness/OPUS interpretation while leaving the incomplete probes
      and final contribution open. Add source hashes for the paired-analysis inputs.
      All19 targeted checks pass;the PDF builds with three explicit conclusion placeholders.
      Displayed endpoint means match the preserved trial hashes for all18 robustness/recipe campaigns.

1. [ ] Integrate the existing audited dose results into a main forward-versus-reverse tradeoff
   figure and paired comparisons. Show base,sft,replay,mix doses and contrastive controls,
   with seed ranges/paired intervals. Report measured runtime and token exposure separately;
   extra CE remains a compute proxy. Current figure,cost records and paired analyses are integrated;
   final knee/retention interpretation remains open.
2. [ ] Audit and integrate matched IFEval/general-ability probes for a fixed declared panel.
   Compare a frozen base,sft,replay,mix5 and registered auxiliary/control panel in a fresh comparable
   probe campaign where existing artifacts are insufficient. Check overlap first,including
   replay/mixedtask filler. Do not claim disproportionate directional loss without this evidence.
3. [ ] Design one independent real-data transfer evaluation for existing checkpoints on a
   separately sourced translation or natural-code corpus. Freeze inclusion/scoring and overlap
   audit before results;no held-out tuning. OPUS protocol is now frozen and smoke-gated.
4. [ ] Design a small fixed prompt/generalization stress test:equivalent rewordings and novel
   representation/template families,including collapse and null cells. Separate instruction
   failure,echo/off-target and semantic errors. Freeze prompts before scoring;never choose the
   best test prompt and present it as the primary endpoint. Frozen panel is now smoke-gated.
5. [ ] If the claim extends across training recipes,register a small fixed learning-rate/LoRA
   rank sensitivity panel that preserves effective batch/directional budgets and includes a
   null cell. Frozen four-setting panel is now smoke-gated;ordinary adapters remain unchanged.
6. [ ] Rewrite the abstract/conclusion around the completed supported contribution relative
   to prior directional-forgetting/reverse-training work. Preserve the units null-collapse
   result and failed gates;replace pending-result prose after analysis,not with stronger promises.
7. [ ] Prepare the reproducible anonymous artifact package:corpora/scorers,exact split and
   trial hashes,manifests,environment,analysis entrypoints and compute accounting. Verify
   citation accuracy and ensure the main paper contains the evidence its central claims need.

Timing verified against official calls:NAACL2027 ARR submission October12,2026 AoE,
commitment December23. EACL2027 commitment October11 requires ARR reviews/meta-review
from the August2026 cycle or earlier;the upcoming October cycle is not eligible for it.
[NAACL call](https://2027.naacl.org/calls/main_conference_papers/),
[EACL call](https://2027.eacl.org/calls/papers/).
No submission/commitment is implied or performed by this checklist update.

## Task expansion coverage

The implemented expansion is units, Python↔C++ and Boolean formula↔truth table.
Units/Gemma12B has three complete seed campaigns;Python/C++/Llama8B has completed
all three seed campaigns;Boolean cells remain blocked after failed gates.
Both eligible domains and contrastive baselines remain first priority in the feeder.

- [ ] Broader Python/C++ conditionals/loops/lists:design proposal,not implemented or queued.
- [ ] External natural-code evaluation:design proposal,not adopted or queued;requires its
      own corpus/overlap audit and recorded scoring setup before measurement.
- [ ] Numerical magnitude/composition generalization split:proposal,not implemented or queued.

These future proposals are distinct from the implemented expansion;their scope is not
silently included in the automatic queue or the paper's completed-task count.

## October 8 continuation — new eligible work first

- [x] Original contrastive campaign:12/12 model/task/seed evaluations complete, each with
      all ten systems scored together and all five primary reverse contrasts generated.
- [x] Full-FT campaigns447368–447373:6/6 complete with fresh within-campaign baselines,
      isolated full-weight engines, coverage/effectiveness checks and paired analyses.
- [x] Spectral-repair evaluation447374 complete. Both directions improve relative to
      original full SFT in this single exploratory campaign; saved-weight audit also passes.
- [x] Corrected unlikelihood447380/447381 training and447382/447383 evaluation complete.
      Negative-objective exposure is verified; reverse generation is near zero. Retain this
      adverse result and keep the invalid original Llama3B results withdrawn.
- [x] Larger-model Gemma12B gate447996:units PASS;logic and py_cpp FAIL.
- [x] Gemma12B units pilot448218→448219 complete over1000 pairs and five systems.
      Base forward/reverse33.7%/35.8%;SFT97.4%/70.0%;mix50 94.9%/90.9%;
      replay94.0%/56.6%;rev44.0%/95.0%. This is a valid null-collapse pilot.
- [x] Register Amendment47:seeds42/1234 for valid larger-model pilots regardless of
      favorable, null or adverse outcomes. Preserve failed base gates.
- [x] Unit-conversion replications448701→448702 (seed42) and448703→448704 (seed1234)
      complete with successful evaluation status and paired analyses. Training took1h13/1h11;
      evaluations each6m23. All three seeds improve both directions under SFT.
- [x] Fix Llama8B logic parser-stack exhaustion; reject malformed formulas while propagating
      real allocation failures. Regression/oracle tests pass; original failed gate retained.
- [x] Retry unchanged Llama8B logic/py_cpp gates448705:logic FAIL;py_cpp PASS
      (85.4%/100.0% strict,3.3%/0.0% format failure). Its failed units gate is preserved.
- [ ] Llama8B Python/C++ pilot448716→448717:training complete;fresh evaluation awaits allocation.
      Valid seed17 results will admit seeds42/1234 automatically under Amendment47.
- [x] Register Amendment48:contrastive fmt/de→en at8B/12B and units/Gemma12B,
      seeds17 first then validity-based42/1234. Fifteen extension cells; original twelve complete.
- [x] Larger-model trainer checks448707/448708 (Llama8B CL/CE) and448709/448710
      (Gemma12B CL/CE) all pass. The feeder admitted three seed17 packs with dependent fresh evaluations;
      later seeds require validated same-pass analyses and recorded auxiliary exposure.
- [x] Exact-quantity negative audit passes for all6500 units training anchors at all three
      seeds, including source/target equivalence and disjoint6500/500/1000 physical quantities.
      Equivalent numeric renderings and held-out negatives are covered by the regression test.
- [x] Register Amendment49:matched larger-model microbatch1-versus4 engineering profiles.
- [x] Profiles448718/448719 (Llama8B CL/CE) and448720/448721 (Gemma12B CL/CE) pass.
      Microbatch4 is admitted:CL/CE runtimes6.48→5.34s/9.86→3.30s for Llama and
      11.56→10.98s/16.97→6.57s for Gemma,at matched effective4;peak20.3–33.5GiB.
      Production uses effective64 with accumulation16. All eight new engineering jobs have bounded infrastructure retries and
      successful retry proof resolution, so a transient failure cannot silently block admission.
- [ ] Complete the new27-cell total contrastive scope, retaining all null/adverse controls.
- [x] All A30 checks447988/447989/447990/447997/447993 complete on the verified node.
      Full Gemma evaluation and largest-matrix proof validators both pass. Placement remains
      limited to proven workloads/nodes;large-model CL production stays on H200.
- [x] D2T raw extractor diagnostic447994 complete. Predicate/entity normalization mismatches,
      missing facts and inverse relations prevent exact reference recovery. Keep D2T blocked
      under the unchanged exact criterion; raw extraction matches do not bypass echo guards.

## Priority 1A — new domains

Expansion plan: [New domains for generality](docs/DOMAIN_EXPANSION_PLAN.md), drafted October 2.

- [x] Bounded Python ↔ C++ translation implemented: single-return integer functions;
      compiler/interpreter equivalence on all33 declared inputs, template/behavior-disjoint
      6500/500/1000 corpus, Amendment39, compute-node scorer/budget audits and GCC14.2.0.
      Existing code/exec overlap audit records all programs as unsupported by the restricted
      grammar; it does not prove unrestricted semantic disjointness.
- [x] Python/C++ gates440727/440728 completed; both FAIL eligibility. Llama strict
      forward/reverse78.6%/74.1%, format failures20.3%/25.9%; Gemma5.1%/99.9%.
      Preserve failures;small-model py_cpp pilots remain blocked.
- [x] Boolean formula ↔ truth table: implemented exact logical equivalence, alternate
      formulas accepted; quantity/function-disjoint corpus 6500/500/1000 and Amendment 36.
      Compute-node oracle and revised symmetric-prompt budget audits passed.
- [x] Boolean gates completed: 440627 (Llama 3B), 440628 (Gemma 4B). Both failed;
      strict accuracy 0% in both directions. Preserve results; pilots remain blocked.
- [x] Unit conversion implementation, exact scorer, quantity-disjoint 6500/500/1000 corpus,
      preregistration Amendment 35, and compute-node oracle/budget audit.
- [x] Unit conversion base gates completed: 440554 (Llama 3B), 440555 (Gemma 4B).
      Both FAIL: Llama forward/reverse strict rates 0.4%/1.9%; Gemma 37.6%/3.5%.
- [x] Inspect unit-conversion failure outputs before proposing a preregistered revision;
      small-model pilots remain blocked under the unchanged eligibility rules.
- [x] Preregister designs, build leakage-free corpora, and audit scorers before base gates.
- [ ] Small-model pilots remain blocked:all six base gates failed. Larger-model eligible
      pilots/replications are tracked above under the same unchanged criteria.
- [x] Promote valid larger-model cells by validity, including the units null-collapse result;
      its three-seed campaign is complete. Python/C++ pilot and replications continue.

## Priority 1B — contrastive baselines (phase1 complete; extensions registered)

Plan: [Contrastive baseline](docs/CONTRASTIVE_BASELINE_PLAN.md).

- [x] Amendment 37 preregisters fixed-setting exploratory InfoNCE, shuffled positives,
      six-pass CE proxies, one-pass evaluation and five primary reverse contrasts.
- [x] Contrastive loss/pooling and deterministic per-pair negative-sampling components;
      four focused logic/contrastive tests pass, including padding invariance, duplicate
      filtering, order-independent schedules, and gradients through both sides.
- [x] Real Llama 3B LoRA GPU loss smoke (440629): gradients through both sides and LoRA;
      peak memory 10.7 GB, no experimental adapter saved.
- [x] Actual two-step SFT trainer smokes pass on Llama/Gemma (440639/440640).
- [x] Integrated trainer/collator, first-batch auxiliary gradient gate, token/pair accounting,
      fixed negatives, shuffled-positive ablation, and repeated CE compute proxies.
- [x] Accumulation/control/mixed/shuffled GPU smokes 440645–440648 all completed successfully.
- [x] Full seed-17 pilots: fmt/Llama 440649→440650, de→en/Llama 440651→440652,
      de→en/Gemma 440653→440654. Five new arms each; existing baselines share one new eval pass.
- [x] fmt/Gemma gate440658 PASS; fourth pilot admitted440785→440786.
- [x] Microbatch4 profiles440706–440709 and Gemma baselines440729/440730 passed.
      Llama CL7.45→3.58s; CE12.53→4.32s. Gemma CL10.60→6.42s; CE18.81→5.23s.
      Later arms automatically use4 where validated; effective batch64 is preserved.
- [x] Matched4-versus8 profiles440966–440973 completed (Amendment41). Auto-admit8
      only after both objective/control profiles pass, fit memory and beat4.
- [x] Contrastive split-batch loss/gradient equivalence and conservative profile-admission
      tests passed. Focused checks:14 tests; program/mixture suite:28 tests.
- [x] Fixed replay matcher exhausted-neighborhood fallback (Amendment40); py_cpp
      sequence budget changed from0.87× to1.00×; existing adapters are preserved.
- [x] Pilot fmt/de→en on Llama 3B/Gemma 4B, with matched CE and shuffled-pair controls.
- [x] Registered same-pass analysis is implemented: five primary reverse contrasts with
      99% paired cluster-bootstrap intervals, trial/script hashes and archive mirroring.
      It rejects missing, duplicate or mismatched trial coverage; three focused tests pass.
      Feeder runs it automatically only after successful pilot evaluations finish.
- [x] Phase1 evaluates all existing/new arms in one fresh pass and replicates all four valid
      task/model cells at seeds17/42/1234:12/12 complete.
- [ ] Complete Amendment48 scale/numerical-domain extensions after trainer/negative-filter checks.

## Autonomous pipeline — October 6 deployment

- [x] Detached Slurm-hosted controller with self-renewing CPU host and pending successor.
- [x] Persistent launcher/supervisor recover controller and supervisor crashes; singleton locks
      prevent duplicate schedulers. Controller passes have deadlines and atomic heartbeats.
- [x] Partial/failed pilot output cannot mark evaluation complete; successful Slurm status is required.
- [x] Live debug pass: controller crash recovery, supervisor crash recovery, detached session,
      pending host renewal, duplicate-start prevention and clean stages all passed.11 focused tests passed.
- [x] Operating/status/stop commands documented in [AUTONOMOUS_PIPELINE.md](docs/AUTONOMOUS_PIPELINE.md).

## October 7 parallel work — fixes, diagnostics and paper

- [x] Amendments42–45 register full-checkpoint layout, D2T metric repair,
      explicit-output prompt diagnostics and fixed-threshold full-FT projection repair.
- [x] Explicit units/logic/py_cpp corpora built (6500/500/1000 each), preserving original
      semantic splits; all split overlaps zero. Compute-node oracle/budget audit passed.
- [x] Llama3B explicit-format gates447365 completed: all three FAIL. Units0.8%/1.5%,
      logic0%/0%, Python/C++83.8%/89.4% forward/reverse strict rates. Python/C++
      forward format failure16.0% exceeds15% eligibility cap. Criteria remain fixed.
- [x] Gemma4B explicit-format gates447366 completed: all three FAIL. Units39.9%/26.9%
      strict forward/reverse, forward format failure15.1% exceeds fixed15% cap;
      logic0%/0.1%; Python/C++29.9%/88.8%, forward format failure70.1%.
      Preserve failures; no explicit-format cell is admitted to training.
- [x] Amendment46 registers original-task Llama8B/Gemma12B feasibility gates and
      gate-dependent seed17 four-arm pilots, retaining every null/adverse result.
- [ ] Larger-model gates447995/447996 pending, each all1000 pairs across three domains.
- [x] Full-FT evaluator routes full weights to isolated processes; synthetic routing tests pass.
- [x] Full-FT GPU loading smoke447349 passed; all five systems/directions covered and
      checkpoint/adapter effectiveness checks passed.
- [x] Six full evaluations447368–447373 complete; full paired comparisons generated.
- [x] Full-FT projection repair implemented and synthetic reconstruction tests passed.
- [x] GPU repair447359 completed on H100 in1m58s. CPU saved-weight audit passes:
      all254 tensors finite with matching coverage/shape/dtype;58 non-projection tensors
      exactly unchanged; all196 projections differ from base and tuned weights. SHA256
      and source-fingerprint checks recorded in results/audit/fullft_saved_repair_447359.json.
- [x] Fresh repair evaluation447374 complete. Forward/reverse improvements observed jointly
      in its new campaign; keep the single-checkpoint exploratory limitation.
- [x] D2T gate uses its exact criteria in both directions and records extractor ceiling.
      Error paths hard-exit vLLM processes. Regression test passed.
- [x] GPU D2T oracle447352 completed and FAILS: forward reference gold0/8;
      reverse gold8/8, all echo/empty/garbage negatives0. Gate447375 cancelled because
      its required audit failed. D2T pilot remains blocked; no scorer relaxation.
- [x] Raw frozen-extractor diagnostic447994 saved expected/observed triples; D2T stays blocked.
- [x] Attribution exposure audit:8 valid result cells;1 withdrawn result. Two invalid
      Llama3B unlikelihood adapters preserved under runs/withdrawn_adapters/2026-10-07.
- [x] Corrected unlikelihood seed17 pair447380→447382 and seed42 pair447381→447383 complete,
      with verified negative auxiliary exposure and adverse reverse-generation outcomes.
- [x] Mechanism report keeps model/seed distinct, uses matched never-had controls, and
      leaves forward-progress ratios undefined without forward gain. Tests pass;
      paper/MECHANISM_AUDIT.json contains descriptive instrument summaries.
- [x] Provenance-backed paper snapshot:103 main cells excluding never-had controls;
      58 meet descriptive reverse-loss threshold;12/12 original contrastive cells complete.
      Generated tables and paper/main.pdf build:7/8 body pages,9 total,0 unresolved numbers/assets.
      Full-FT table shows both directions; unit-conversion table retains the null-collapse pilot.
- [ ] Final results/figures/interpretation await queued experiments. A successful draft build
      does not certify submission readiness or completed GPU validation.
- [x] Extend archive coverage to failure dumps, mechanism reports, training summaries,
      withdrawal markers, repair manifests and actual Slurm status files.
- [x] Full CPU/tokenizer test suite:318 tests pass. Live controller restart/recovery passed;
      cached evidence/PDF refresh stage is active.
- [x] Automatic follow-up analysis checks successful status, paired coverage, effectiveness
      and auxiliary exposure; eight legacy attribution reports refreshed. New full-FT,
      repaired-checkpoint and diagnostic-pilot reports refresh as jobs complete.
- [x] A30 allocation-local diagnostic447401 passed on g-04-01: cuInit, CUDA tensor,
      driver580.178.04, torch2.11.0+cu129. This alone does not certify workload fit.
- [x] A30 actual trainer checks447988 (Llama CL) and447989 (Gemma CE) pass.
- [x] A30 evaluation447990 (Llama),447997 (Gemma full ten-system contrastive campaign),
      and largest real projection SVD/memory check447993 complete on g-04-01. The queued three-system
      Gemma check447991 was superseded before running by447997.
- [x] Controller automatically widens only eligible pending jobs after matching A30 workload
      proofs pass, preserving H100/H200 options and excluding untested A30 nodes. Three
      admission tests passed; placement/proof journal: runs/feeder/a30_admission.json.

## October 7 execution checkpoint (20:34 CDT, historical)

- [x] Detached controller healthy: all six stages exit0, no active operational failures
      or quarantined cells. Three GPU jobs running; no artificial cap.
- [x] Inspect newly completed Gemma explicit-domain failures and full-FT repair results.
      Saved repair audit and five focused audit/spectral tests pass. Repair evaluation447374
      has cleared its dependencies and awaits allocation; no recovery claim yet.
- [ ] Contrastive remains9/12 analyzed evaluations. fmt/Gemma seed42/1234 packs are
      finishing sft_extra_ce, followed by mix5_extra_ce. Corrected unlikelihood seed17
      was231/306 steps at20:33; its fresh evaluation remains dependent on successful training.
      Fifteen-second sampled GPU utilization averages approximately95%/86%/98%.
- [ ] Larger-model domain gates, A30 workload checks and remaining evaluations continue
      waiting on scheduler Priority/Resources. Scientific failed gates stay blocked.
- [x] Evidence snapshot and draft now record all six explicit-format gate failures and
      one audited repair checkpoint with source hashes. ARR checks pass at7/8 body pages,
      eight pages total, no unresolved numbers/references/assets or unembedded fonts.

## October 7 execution checkpoint (20:04 CDT, historical)

- [x] Controller debug checks pass; all six stages exit0. Historical fmt/Gemma seed17
      operational failure reconciled against successful job446060; no active human alarms
      or quarantined cells. Original failure evidence remains recorded.
- [ ] Contrastive evaluations remain9/12. Both fmt/Gemma packs completed cl_fwd,
      cl_mix5 and cl_shuffled and are now training sft_extra_ce, followed by mix5_extra_ce.
      At19:59, seed42/1234 controls were201/306 and161/306 optimizer steps; sampled
      GPU utilization88%/100%. These are training progress, not completed evaluations.
- [x] Shorten only pending engineering jobs447988/447989 and raw extractor diagnostic
      447994 to10minutes using prior72–91second trainer checks and174second oracle.
      Larger-model gates447995/447996 receive one hour instead of two, allowing a wide
      margin over the295second small-model gate. No sample/settings/threshold changes.
      Exact before/update records: runs/feeder/backfill_walltime_updates_2026-10-07.json.
- [ ] New-domain gates, remaining evaluations, A30 validation and full-FT/attribution jobs
      still await cluster allocations. All8 A30 GPUs are allocated to other users at this
      checkpoint; single tested-node placement is retained until workload validation passes.
- [x] Add provenance-backed extra-CE/shuffled control table to the paper: every completed
      cell now displays all five registered primary contrastive comparisons across two tables.
      Keep incomplete replication and observed-floor intervals explicit.
      ARR draft check passes:7/8 body pages,8 total pages, no unresolved references,
      missing assets, unfilled numbers or unembedded fonts.

## October 7 execution checkpoint (18:13 CDT, historical)

- [x] Nine of twelve contrastive evaluations/paired analyses complete. fmt/Llama seed1234
      evaluation446066 finished; de→en/Gemma seed1234 training446071 finished.
- [ ] de→en/Gemma evaluation446072 waits on Priority. fmt/Gemma seed42/1234 packs
      446427/446429 are running at99–100% sampled GPU utilization; dependent evaluations
      446428/446430 follow automatically. Remaining compute controls still need training.
- [x] Fresh heartbeat, all six controller stages, detached session, CPU-host renewal and
      controller crash recovery passed. Fix health-check race during an in-progress stage.
- [ ] Full-FT evaluations/repair, corrected attribution runs and new-domain/A30 validations
      remain queued. No artificial GPU cap; cluster allocation is the present capacity limit.

## October 7 execution checkpoint (11:45 CDT, historical)

- [x] Controller health checks pass; continuous scheduling and detached host renewal active.
- [x] All four seed17 contrastive evaluations and paired analyses now complete.
- [x] Replication training/evaluation complete for fmt/Llama seed42, de→en/Llama
      seeds42/1234, and de→en/Gemma seed42 (446061–64, 446067–70).
- [ ] Four H200 packs running: fmt/Llama seed1234 (446065), de→en/Gemma seed1234
      (446071), fmt/Gemma seeds42/1234 (446427/446429). Dependent evaluations
      446066, 446072, 446428, 446430 automatically follow.
      GPU utilization snapshot: 72%, 100%, 100%, 100% respectively.

## October 6 execution checkpoint (historical)

- [x] Evening health check: detached controller, supervisor, clean stage exits, and CPU-host
      renewal all pass. GPU jobs are waiting on cluster priority; no GPU job is running yet.
- [x] Broaden remaining fmt/Gemma seed17 evaluation (446060) to H100/H200, excluding both
      known bad nodes. Reduce six pending contrastive training packs to 32 GiB host memory
      from 64 GiB; completed seed17 packs peaked below 15 GiB. Future contrastive packs use
      the same request. Evaluations retain 64 GiB.

- [x] Seed17 contrastive training finished for all four cells (20 new adapters).
- [x] Evaluations/registered paired analyses finished for fmt/Llama and de→en/Llama/Gemma.
- [ ] fmt/Gemma evaluation recovery446060 on H100/H200 after repeated CUDA initialization
      failures on g-08-06. Preserve adapters; exclude that node from feeder submissions.
- [ ] Seeds42/1234 replication admitted by implementation validity, including null effects:
      fmt/Llama446061→446062 and446065→446066;
      de→en/Llama446063→446064 and446067→446068;
      de→en/Gemma446069→446070 and446071→446072.
      Six packs,30 new training runs, followed by fresh within-pass evaluations.
- [ ] fmt/Gemma replication waits for successful seed17 evaluation/analysis and validity.
- [x] Eight matched batch4/8 engineering benchmarks completed successfully; larger batches
      remain selected only when measured faster with memory headroom.

## Current execution

- [x] Continuous feeder and watchdog active; CPU host renews automatically.
- [x] H200-capable submissions use `juno-pri` (`OverPartQOS`, eight concurrent jobs).
- [x] Feeder maintains up to 24 cells in flight; each cell has training and dependent evaluation.
- [x] Released five infrastructure-failed cells after moving them from A30 to H100/H200.
- [x] Nine focused runner/control/share tests passed on a compute node on October 2.
- [x] Removed the artificial six-GPU cap and released all capacity holds on October 3.
- [x] Finish the remaining seven legacy cells; the existing eligible campaign is complete.
- [x] Two unit-conversion gates finished; both failed eligibility.
      Unit-conversion and Boolean model gates failed eligibility; pilots remain blocked.
      Contrastive integration checks passed; three pilot packs and dependent evals submitted.
- [x] Diagnose A30 CUDA initialization on g-04-01; actual trainer checks pass.
- [x] Validate complete A30 evaluation/SVD workload proofs;all checks pass on g-04-01.

The old estimate of378 remaining GPU-hours is obsolete. No new completion date is asserted:
queue delay and measured large-model contrastive runtimes determine extension completion.
The cluster has `PriorityWeightQOS=0`; this QoS raises concurrency, not queue priority.

## Main experiment grid

| Priority / experiment | Completed | Outstanding | Status |
|---|---:|---|---|
| P1 Llama 3B seed-17 breadth: relation, en→zh, zh→en | 3/3 | None | Complete |
| P2 Llama 3B formal/format cells, seed 17 | 3/5 | diacritics, fmt_det75 fail base gate | Partial; blocked |
| P3 Llama 3B four core cells, seeds 42 and 1234 | 8/8 | None | Complete |
| P5 Gemma 4B core + breadth, seed 17 | 7/7 | None | Complete |
| P5b Gemma 4B core + breadth, seed 42 | 7/7 | None | Complete |
| P6 OLMo 1B core + breadth, seed 17 | 4/7 | code, en→zh, zh→en fail base gate | All eligible cells complete |
| P6b OLMo 1B core + breadth, seed 42 | 4/7 | Same three failed gates | All eligible cells complete |
| P7 Llama 8B core + breadth, seed 17 | 6/7 | code fails base gate | All eligible cells complete |
| P7b Llama 8B core + breadth, seed 42 | 6/7 | code fails base gate | All eligible cells complete |
| P7 Gemma 12B core + breadth, seed 17 | 7/7 | None | Complete |
| P7b Gemma 12B core + breadth, seed 42 | 7/7 | None | Complete |
| P9 Three small models, core + breadth, seed 1234 | 18/21 | Three OLMo failed-gate cells | All eligible cells complete |
| P9 Llama 3B formal/format replication, seeds 42/1234 | 6/10 | Four failed-gate entries | All eligible cells complete |
| P8/P9 Llama 3B exec, seeds 17/42/1234 | 3/3 | None | Complete |

- [x] Formal cells algebra, algebra_rev, and fmt completed at Llama 3B seed 17.
- [x] Algebra seed 1234 completed training and evaluation overnight.
- [x] Algebra seed 42: training and evaluation complete (438897 → 438898).
- [x] Algebra_rev seed 42: training and evaluation complete (438899 → 438900).
- [x] Fmt seed 42: training and evaluation complete (438901 → 438902).
- [x] Algebra_rev seed 1234: training and evaluation complete (438903 → 438904).
- [x] Fmt seed 1234: training and evaluation complete (438905 → 438906).

## Mechanism experiments

- [x] Review existing explanatory instruments against their implementations and refresh the
      descriptive report (`runs/feeder/mechanism_explanation_20261009.json`). Results disagree:
      prompt/relearning recovery exists, while scaling and spectral summaries do not support
      a universal suppression account. No pooled vote establishes causality.
- [x] Identify a concrete causal lead: format/Llama3B middle-to-late layer ablation restores
      some reverse behavior with forward performance retained (successful job425259).
- [x] Record limits: direction probes confound instruction with input language/format;
      first-token agreement is not output-mode probability; same-input sensitivity uses an
      invalid-direction input; the never-had task is model/seed matched but task-different.
      LoRA leaves base weights fixed, and original diagnostics lack paired trial uncertainty.
- [x] Freeze and test a separate explanatory panel on existing collapse cells plus the units
      null cell, preserving current new-domain/contrastive production priority.
- [ ] Compare correct reverse-answer likelihood/ranking with free generation and output errors
      for base/SFT/replay/mixed-direction/contrastive/control checkpoints.
- [ ] Rerun fixed layer interventions with per-instance trials, both directions, equal-norm
      global scaling and random-removal controls; retain all bands and adverse outcomes.
      Match the actual weight delta (LoRA BA), not separate factor norms.
- [ ] Measure local forward/reverse gradient conflict and verify its predicted effect using
      controlled small updates; calibrate and evaluate on disjoint pairs.
- [ ] If needed, test correct reversed-pair correspondence against shuffled reverse targets
      at matched exposure/steps; existing shuffled contrastive pairs test a different objective.
- [ ] Add pair-grouped, input-controlled probes and seed replication before making a routing
      claim. Integrate the causal tests with hypothesis-specific, falsifiable conclusions.

- [x] P4 Llama 3B seed-17 relearning: 6/6 planned cells complete, plus the model-matched
      never-had control and its relearning ladder.
- [x] P4b Gemma 4B seed-17 relearning: 5/5 planned cells complete, plus matched control/ladder.
- [x] P4c OLMo 1B seed-17 relearning: 2/2 planned cells complete, plus matched control/ladder.
- [x] Mechanism artifacts exist: 13 elicitation reports; 13 alpha-scale reports; 13 LoRA
      spectral reports; six each for layer ablation, sensitivity, and direction probes.
- [ ] Review those reports for completeness, validity, and interpretation before marking
      mechanism claims or paper sections complete.
- [x] P10 seed-42 control training/evaluation: all three models complete.
- [x] P10 seed-42 control ladder training: all three models complete.
- [x] P10 seed-42 control ladder evaluation: all three models complete.
- [x] P10 seed-42 Llama 3B breadth prerequisites and all six mechanism cells complete.
- [x] P10 seed-42 Gemma 4B mechanism: all five planned cells complete.
- [x] P10 seed-42 OLMo 1B mechanism: both planned cells complete.

Mechanism curves describe their own starting checkpoint. Seed-17 collapse does not establish
seed-42 collapse; label each curve using that seed's observed loss.

## P8, attribution, and implementation gaps

- [x] Exec base gate passed; its three Llama 3B seeds have complete grid results.
- [x] Full-fine-tuning training produced 12 final checkpoint directories across six Llama 3B
      seed-17 cells: de→en, en→de, sql, code, en→zh, zh→en.
- [x] Implement full-FT evaluation with isolated checkpoint engines (Amendment42).
- [x] GPU-validate full-FT loading and evaluate all six completed checkpoint cells.
- [x] Generate matched full-FT campaign comparisons; label isolated-engine restarts.
- [x] Implement full-FT projection-delta spectral repair; existing completed spectral artifacts concern LoRA.
- [x] Audit attribution manifests and withdraw the invalid original unlikelihood artifacts;
      corrected Llama3B seed17/42 runs are complete with verified auxiliary exposure.
- [ ] Finish attribution interpretation and reconcile the audit inventory with corrected runs.
- [x] Fix the d2t gate's asymmetric metric handling. Job 426814 crashed with `KeyError:
      'chrf2'` in reverse scoring and then held a GPU during teardown; no valid gate report exists.
- [ ] D2T remains blocked after its failed frozen-extractor oracle; require a separately
      registered, validated measurement design before any new tuned experiment.
- [ ] RQ3 format ladder: full tuned experiment remains blocked. Llama 3B gates fail for
      fmt_det75/50/25/00 and automata; preserve descriptive base results and do not bypass gates.
- [ ] Coverage: Llama 3B base gate fails; leave tuned experiments blocked unless justified by
      new diagnostic evidence.

## Established foundations and final analysis

- [x] Data build and leakage checks, adaptive training batch, separate COMET environment,
      decision gate, determinism measurement, and initial general-ability probes were completed
      before this campaign. See the archived checklist for historical measured findings.
- [ ] Refresh contrasts, tables, figures, mechanism report, and paper provenance from valid
      results after the queued replications finish.
- [ ] Audit trial counts, summaries, adapter effectiveness, withdrawal markers, and archive
      coverage before publication; a Slurm COMPLETED status alone is insufficient.
- [ ] Report seed ranges for partial collapses (preregistration Amendment 34).
- [ ] Preserve the relation country-level bootstrap and avoid unsupported equivalence claims.
- [ ] State the replay supervised-token imbalance and mixedtask quantity confound.
- [ ] Confirm scratch retention policy; keep mirroring irreplaceable trials and manifests.

## Evidence and live status

- `scripts/95_runner.py --status`: planned cell completion, prerequisites, and base-gate checks.
- `runs/feeder/STATUS.md`: automatically updated runtime summary.
- `runs/feeder/FIXES.md`: infrastructure fixes and placement changes.
- `runs/feeder/FAILURES.md`: historical failure evidence; resolved failures remain in its history.
- `/scratch/juno/jvl210002/bidir/results`: trial files and mechanism artifacts.
- `/scratch/juno/jvl210002/bidir/runs`: adapters, full-FT checkpoints, and manifests.

This checklist is a dated snapshot. Live queue state can change after it is written.

## Explanatory diagnostics — Amendment52

Live per-job checklist: `runs/feeder/mechanism_status.json`; fixed design:
[docs/MECHANISM_PLAN.md](docs/MECHANISM_PLAN.md).

- [x] Register a fixed four-cell panel at seeds 17/42/1234, including the units reverse-null control.
- [x] Freeze evaluation subsets outside the earlier 200-pair exploratory mechanism slice.
- [x] Implement native completion likelihood, actual delta-W norm controls, validation gradients,
  and reversible equal-norm held-out local updates in an isolated result namespace.
- [x] Pass CPU verification (403 tests) and native-tokenizer/checkpoint preflight (84 adapters); enable the tested scheduler.
- [x] Submit six GPU engineering smokes (450241–450246); verify idempotent queue admission.
- [ ] Complete six GPU engineering smokes; retain technical failures and scientific nulls.
- [ ] Complete 12 likelihood/gradient production jobs and 12 controlled layer/generation jobs.
- [ ] Summarize all fixed contrasts with paired uncertainty; distinguish evidence for selection,
  selective layers and local interference from evidence for preserved internal knowledge.
- [ ] Integrate diagnostic findings and failed hypotheses into the paper after full proof validation.
- [ ] Conditional follow-up: shuffled reverse-CE correspondence control if the registered
  diagnostics and existing contrastive controls leave that attribution unresolved.
