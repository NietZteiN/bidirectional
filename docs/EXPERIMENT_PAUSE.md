# Experiment pause and manuscript priority — Amendment 56

The user paused all current bidirectional experiments on October 10, 2026 and prioritized
completing the paper from existing evidence. This supersedes Amendment 55's active scheduling.

- The persistent `runs/feeder/STOP` sentinel is present; the controller phase is stopped.
- All 32 project experiment jobs are held. Four running training packs were requeued into
  held state; the other 28 submitted jobs were held before they could start.
- There are zero running experiment jobs. The shared CPU host and its renewal remain
  available for writing and analysis. Other projects were not stopped.
- Completed adapters, trials, manifests, job IDs and evaluation dependencies remain saved.
  An unfinished arm can restart on resume; no mid-optimizer checkpoint is promised.
- Read-only collection, checked CPU analyses, manuscript editing and local verification
  can continue. Do not restart the scheduler, release jobs or launch experiments until the
  user requests resume. No experimental result is inferred from a held queue entry.

The detailed before/after Slurm specifications, actions and verification are recorded in
`runs/feeder/experiment_pause.json`. Historical campaign plans and sprint receipts are retained.

On an explicit resume request, inspect that receipt and the current queue, release only the
still-held experiment IDs recorded there, then use `python scripts/97_pipeline.py start`.
Verify prerequisites and controller health; pending work resumes from saved complete arms.
Neither starting the controller alone nor clearing all account-wide holds is an appropriate
resumption procedure. Keep the shared CPU-host jobs outside experiment release operations.

The current paper priority is to resolve stale interpretation/checklist placeholders, replace
planned prose with supported findings, record methodological/reproducibility limits, and
rebuild a complete draft. Missing measurements stay marked unavailable; no GPU runs are needed
for these writing tasks.
