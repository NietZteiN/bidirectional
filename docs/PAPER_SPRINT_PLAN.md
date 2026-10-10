# 24-hour paper sprint — Amendment54

User decision, October 10: no additional replication; prioritize a minimal set of distinct
experiments and finish a paper snapshot within 24 hours. This supersedes the scheduling
priorities in earlier plans while preserving their protocols and evidence.

Start UTC: 2026-10-10T16:48:14.672677+00:00
Target UTC: 2026-10-11T16:48:14.672677+00:00

## Scientific scope

Reuse91 audited core cells,24 valid original contrastive campaigns, completed seed17
new-domain variants,39 paper-extension production jobs, and six stricter OPUS transfer
boundaries. Existing training seeds remain reported. More favorable outcomes are not a
selection criterion. Original failed gates, negative results and generation floors remain.

Five selected production jobs reuse the unchanged frozen Amendment52 panel:

| Job | Question | Preserved controls |
|---|---|---|
| fmt/Llama3B HF s17 | Is reverse answer likelihood impaired alongside generation? Do local direction gradients conflict? | All eight systems, gold/copy/shuffled candidates, all local steps |
| mt_de-en/Llama3B HF s17 | Does the pattern appear in translation? | Same full HF controls |
| fmt/Llama3B layers s17 | Is recovery selective to depth beyond removing update norm? | All four bands, scaling, both random removals, both directions |
| mt_de-en/Gemma4B HF s17 | Does the diagnostic extend to another architecture? | Same HF controls; repaired text graph |
| units/Gemma12B HF s17 | What happens in a null-collapse cell? | Same HF controls; repaired text graph |

Two revised Gemma GPU smokes must pass before their production jobs are admitted. Single
training-seed diagnostics remain exploratory. Pair bootstrap intervals do not establish
training-seed stability. The original24-job registration is not retrospectively reduced
or called complete;19 production jobs are deferred.

## Execution

37 out-of-scope jobs were cancelled after checking their Slurm working directory. Four
were running training packs; completed arms and partial artifacts remain untouched.
The unfinished Gemma12B repaired-format contrastive pack is deferred to protect the time
budget; the24 completed original contrasts supply the objective/control comparison.
Other projects and CPU controller renewal jobs are untouched.

configs/paper_sprint.json is the explicit allowlist. Planners reload it every pass, and
scripts/slurm/submit.py independently checks admission. No out-of-scope jobs or new seeds
can be submitted. At the target time admission remains closed; pending selected jobs are
cancelled by the diagnostic planner. Already-running selected jobs retain bounded walltime
and finish recording evidence. The controller continues analysis/archive/draft refreshes.
HF jobs can use full H100/H200 GPUs; excluded MIG/small nodes remain excluded. Existing
capacity settings are retained so neighboring probing work is not disturbed.

## Paper delivery

Automatic analysis reports the five-job scope separately from the original panel, retains
all fixed contrasts, and generates source-backed appendix tables. Completed diagnostics
enter the draft after successful source/data/trial/status proof checks. The evidence
refresher watches the diagnostic report and regenerates the PDF as results arrive.

Finish the argument, preservation/probe interpretation, format checks and local anonymous
bundle using available evidence. If queues or smoke failures prevent a diagnostic finishing,
freeze the snapshot with that limitation explicit. This is a target for a coherent reviewed
paper draft, not a guaranteed conference-ready causal claim or automatic submission.

## Execution checkpoint — October 10, 12:03 CDT

Three production diagnostics completed successfully:formatting HF 5:13, translation HF 5:01,
and formatting layers 7:29. Both revised Gemma smokes pass (1:38 and 2:01); their full
production jobs 450906/450907 are queued. Pending HF reservations are two hours, with
all frozen sample sizes retained. The strict draft format check passes with eight body
pages, 16 total pages, 183 abstract words, no unfilled visible placeholders, no unresolved
references and all fonts embedded. Five controller/debug checks pass; 43 distinct targeted
regression/reporting checks pass across the staged test runs. Archive has zero outstanding
files. Final interpretation/bundle review follows remaining results; registration and
format-check success do not establish conference readiness.
