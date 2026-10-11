> Current Amendment57 priority: [calculate missing table entries](TABLE_COMPLETION_PLAN.md).
> The exclusive table controller is active; all 32 earlier jobs remain held. CFT stays
> code-specific. The broader campaign and pause checkpoints below are historical.

# Registered background campaign — Amendment55

User decision, October10, after completed draft delivery: run all remaining registered
eligible experiments in the background, including formerly deferred replications.
The completed eight-body-page draft and five-job diagnostic snapshot are retained at
Git commit34de8d0. The working evidence/draft can refresh as validated results arrive;
that does not change the historical sprint's selection or completion receipts.

## Queue and order

1. Repaired-format Gemma12B contrastive seed17 resumes eight missing arms; original
   completed arms are retained. Its same-pass evaluation must validate before seed42/1234
   packs are eligible. New seeds are promoted by validity, including null/adverse effects.
2. Eight repaired-domain seed42/1234 evaluations reuse completed adapters: typed Python/C++
   at Llama8B/Gemma12B, formatting contract at Llama8B, units contract at Gemma12B.
3. Six Llama8B invertibility packs resume missing arms and paired evaluations for
   fmt_det75/50/25 at seeds42/1234. No completed adapter is overwritten.
4. The remaining19 fixed explanatory production jobs resume the original24-job panel,
   across all three registered seeds and both HF/layer modes. All required original or
   revised matching engineering smokes are proof-valid. Preserve every fixed candidate,
   local step and band/control contrast, including the units null control. Nice1000 keeps
   these behind the first-priority domain/contrastive panel.

The first three groups comprise15 immediately ready runner cells; later Gemma12B
contrastive seeds are conditional. Capability-gate/oracle failures and incomplete task
implementations remain blocked rather than being repeatedly launched. D2T requires a
new tested scorer/repair protocol before its failed frozen oracle can be superseded;
the task catalogue is not an executable GPU queue.

## Operation and review

configs/paper_sprint.json has enabled=false under Amendment55; its old scope/deadline
are historical. Existing scheduler allowlists therefore return to the original plans.
The renewable CPU host runs the detached supervisor/controller after logout. Slurm owns
training and inference. Packs run missing arms back to back, evaluations depend on
successful training, and planners refill the eligible queue automatically.

The former six-GPU cap remains disabled, as does the artificial per-project QoS share.
Actual Slurm resource/account limits still apply; neighboring projects are untouched.
Original memory validation and excluded MIG/small nodes remain enforced. All HF diagnostic
production reservations use two hours and H100/H200 eligibility: the four full seed17 HF
cells completed in under ten minutes, including validated Gemma12B HF on H100. This resource
change retains every frozen sample/control and substantial runtime headroom. Layer-mode
reservations retain their original bounds/validated partitions. Pending changes and original
Slurm specifications are recorded in background_resume.json; running jobs are untouched.
Do not remove quarantine or retry known failed original Gemma HF implementations; the
registered text-graph repair has its separate successful namespace and smoke proofs.

Controller stages analyze completed campaigns and rebuild audited working evidence.
Contrastive analysis mirrors current trial/manifests to archive after new valid reports;
the local anonymous bundle refreshes through the existing paper-completion planner.
The sprint's narrow appendix remains a disclosed historical subset; additional diagnostic
results are retained in paper/MECHANISM_DIAGNOSTICS.json for subsequent scientific review.
No favorable-outcome selection or automatic conference submission follows scheduling.

Live files: runs/feeder/heartbeat.json, supervisor.json, mechanism_status.json,
gate_repair_status.json, STATUS.md and feeder.log. A dated admission/debug checkpoint is
stored in runs/feeder/background_resume.json. TASKS.md distinguishes completed, queued,
conditional and blocked work. Use python scripts/97_pipeline.py status or debug after
sourcing scripts/env.sh. The CPU host has a queued renewal successor.

## Admission/debug checkpoint

All19 remaining mechanism jobs are submitted. The controller submits15 runner cells,
including seven training packs, eight evaluations reusing completed adapters, and seven
dependent evaluations:41 GPU jobs across34 cells in total. They are initially pending
Slurm priority/dependencies; submission is distinct from an allocated/running GPU.
All five detached-controller/renewal health checks pass, and43 distinct targeted regression tests
pass. The D2T gate stays blocked by its completed failed extractor oracle. The two later
contrastive seeds remain conditional on a valid seed17 campaign. Eight pending HF jobs
were updated to the validated two-hour/H100-H200 reservation without touching running jobs.

Sprint receipt hashes refer to frozen commit34de8d0, not the live files that Amendment55
now updates. Background records and this plan are included in subsequent anonymous bundles
so historical receipt/source changes remain explicit. The narrow sprint appendix stays a
historical subset; the complete diagnostic JSON continues tracking the full24-job panel.

The runner continues planner, overflow and contrastive-analysis refreshes even when
the queue meets its target; max-new0 prevents adding cells during that pass. This fixes
a capacity-edge stall without imposing a GPU cap. A regression check covers both
continued maintenance and the absence of new submissions at a full queue.
