# Paper-completion execution plan — October 8, 2026

User authorized queueing and testing the draft additions. The frozen scientific specification
is [configs/paper_finish.json](../configs/paper_finish.json), registered as Amendment 50 in
[PREREGISTRATION.md](../PREREGISTRATION.md). Existing new-domain and contrastive work keeps
first priority. Failed eligibility gates remain findings; their tuned runs stay blocked.

| Work | Fixed scope | Automatic admission |
|---|---|---|
| General ability | MT/Llama3B, fmt/Gemma4B, units/Gemma12B; three seeds; all matched arms; IFEval and GSM8K | Complete adapters and matching generation/scorer smoke |
| Independent transfer | OPUS-100 de-en; Llama3B and Gemma4B; three seeds | Matching smoke, then unchanged full fresh base gate before tuned generation |
| Prompts/templates | MT/Llama3B, fmt/Gemma4B, units/Gemma12B, py_cpp/Llama8B; three seeds | Matching smoke and complete ordinary/registered auxiliary adapters |
| Recipe sensitivity | fmt/Llama3B and units/Gemma12B; three seeds; four settings, two arms | Rank64 trainer smoke, recipe load/eval smoke, resumable training, fresh eleven-system eval |
| Synthesis | Latest valid campaign per cell/tag; paired intervals, seed variation, measured runtime/exposure | Refresh from validated completed trials; no cross-pass paired differences |
| Local artifact | Code/configs, frozen data, trials/manifests, audits, figures/tables, original/redacted hash mapping | Local build now; final refresh after production closes; external release remains separate |

This registers **33 evaluation campaigns plus six recipe-training packs**, with **48 new
adapters** in separate directories. Thirteen engineering smokes cover the worker/model
combinations. Production admission depends on their successful allocation-local results,
coverage, adapter effectiveness, current implementation/data hashes, and Slurm success.
Smokes are never included in scientific estimates. Null/adverse outcomes are retained.

The frozen data manifest records exact Hugging Face revisions and corpus hashes. All 541
IFEval and 1,319 GSM8K test items survived the recorded lexical overlap rules. The independent
transfer set contains 1,688 OPUS pairs after fixed character filters and overlap/duplicate
checks. Synthetic magnitude/template panels each contain 1,000 instances. Gold outputs
passed the entire synthetic oracle on compute, including C++ compilation/execution. Lexical
audits do not establish semantic/pretraining independence; numerical task structure is shared.

The probe system prompt is neutral and common across adapters; transformations retain their
original system prompt. Primary GSM8K uses the fixed strict `####` extraction; flexible numeric
extraction is secondary. IFEval strict/loose validators remain separately recorded. Do not
pool their scales or select a best auxiliary setting/prompt from test outcomes.

## Execution and monitoring

The detached controller invokes the ordinary runner first. The runner then calls
`104_paper_finish.py`, which deduplicates live jobs, preserves dependencies, and admits only
validated work. `106_paper_synthesis.py` regenerates numerical artifacts after result changes;
the existing evidence refresher rebuilds the PDF. The watchdog supplies bounded infrastructure
retries and quarantines repeated code failures. Complete recipe arms survive pack restarts.

```bash
source scripts/env.sh
python scripts/104_paper_finish.py --max-new 0   # reconcile/report without submitting
python scripts/95_runner.py --status
python scripts/97_pipeline.py debug
python scripts/107_paper_artifact.py --build
python paper/artifact/verify_bundle.py
python paper/artifact/replay_means.py
```

Live per-item states/job IDs are in `runs/feeder/paper_finish_status.json`. The production
enablement marker is set; every workload still needs its matching successful smoke. GPU
utilization caps remain disabled. Pending allocation is distinct from a passing GPU test.

October8 acceleration:small non-COMET engineering smokes may use H100/H200 or the previously
verified g-04-01 A30 environment;the other A30 nodes remain excluded. Gemma12B engineering
smokes may use H100/H200. Production A30 and Gemma12B recipe training remain unpromoted.
Larger-model contrastive evaluations can use H100/H200;the ordinary full Gemma12B multi-LoRA
H100 campaigns already completed. Objective/control training profiles remain H200-specific.
Seven pending smoke specifications and four dependent evaluations were widened without cancelling
running jobs or changing samples,training budgets,scoring,or pilot validity requirements.
Original specifications and update commands are in `runs/feeder/paper_finish_acceleration_20261008.json`.
At17:16 CDT,seven GPUs were active. Slurm's first widened-smoke prediction moved from tomorrow
11:32 to today17:38;these are tentative scheduler predictions,not guaranteed allocations.

October8 evening:24/39 production campaigns have closed and12/13 engineering smokes pass.
All six OPUS campaigns stop at the unchanged echo-validity gate;their base-only boundary
trials are retained and tuned comparisons remain blocked. The updater now handles the
singular `probe_panel` configuration key when processing completed `probes` campaigns.
When the eight-job `juno-pri` QoS blocks ready work,an idle H100 may admit one owned pending
job under the uncapped `normal` QoS. This requires a free physical GPU,at least eight CPUs
and64GiB unallocated memory,keeps bad-node exclusions,and does not modify running jobs.
Every update is recorded in `runs/feeder/h100_overflow_admission.json`;occupied H100s do not
trigger routing changes. Normal-priority placement still respects Slurm's shared queue.

October9 morning:37/39 production jobs and all13 smokes are complete. All six recipe
packs/48 adapters and their evaluations are complete. The remaining two units probes
await seed42/1234 contrastive adapters;those packs and four other contrastive replication
packs are active,with evaluation dependencies already queued. The draft now includes
generated prompt/recipe tables with same-campaign differences and descriptive seed ranges,
including adverse forward costs and recipe-dependent reverse losses. Every displayed
setting is retained. These ranges do not replace paired intervals or establish equivalence.
Displayed endpoint means were replayed against the preserved trial hashes for all18
robustness/recipe campaigns. The completed OPUS gate interpretation is integrated;final
probe/contribution interpretation remains open. Three visible conclusion placeholders
remain in the build.

The corrected core inventory has **91 distinct model/task/training-seed cells from 103
validated campaigns**, with 12 earlier passes retained as superseded. Its descriptive collapse
threshold is met in 52 latest cells. Selection uses completion time, never effect sign.
The historical 103/58 snapshot counted repeated passes and must not describe unique cells.

The local anonymous bundle excludes weights/authentication files and records both original
and redacted hashes. Its standalone verifier and strict-mean replay use standard Python.
Full training/evaluation reconstruction needs the project environments, model access, upstream
data licenses, and recorded historical code versions. Missing historical versions remain
explicit; no external publication or claim of submission readiness is implied.
