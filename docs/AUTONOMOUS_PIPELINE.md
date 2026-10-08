# Autonomous experiment pipeline

The configured campaign runs without an SSH session or chat prompts. All GPU training and
inference are Slurm jobs; the controller runs in the existing renewable CPU allocation.

The four controller source files under `runs/feeder/` are versioned;generated queues,logs,
heartbeats,locks,and job state remain ignored. A deployment copy of the current shared
CPU-host script is [scripts/slurm/hold_node.sbatch](../scripts/slurm/hold_node.sbatch).
On a new clone,adapt `scripts/env.sh` and that script's cluster paths,then install the host
script at the deployment location expected by `97_pipeline.py` (`../node/hold.sbatch`).
Create `runs/feeder/quarantine.txt` as an empty file only for a fresh deployment;retain it
when resuming an existing deployment. Restore the corpus/artifact roots and pinned environments
before enabling submissions. Versioning the source does not recreate running Slurm jobs or
replace the local archived measurement evidence.

`node/hold.sbatch` queues an `afterany` successor and starts `runs/feeder/feeder.sh`.
The launcher keeps a supervisor alive; the supervisor restarts a crashed or unresponsive
controller. Each controller pass runs the watchdog, capacity check, scheduling, queued
submissions, registered contrastive analyses, and cached paper evidence/PDF refreshes. Heartbeats and stage results are written
atomically. There is no feeder lifetime expiry or artificial GPU cap.

Training packs resume from fully saved adapters, run remaining arms back to back, and
schedule dependent within-pass evaluations. Valid seed-17 contrastive cells automatically
admit seeds42/1234 regardless of measured effect sign. Future arms select larger microbatches
only from successful, faster GPU profiles with memory headroom. Gates, finite retry limits,
and quarantine remain enforced. Interrupted legacy shell-queue transactions are preserved
for review to prevent duplicate GPU submissions; the regular experiment plan continues.

New pilot evaluations count as complete only when trials, a finished summary, and a matching
successful Slurm status exist. A partial output or failed evaluation cannot satisfy that check.
Registered contrastive analyses run after successful evaluations and are mirrored to `archive`.

The October7 explicit-output domain diagnostics enter pilots only after successful gates.
Full-FT evaluation jobs wait for GPU checkpoint-loading validation; repair evaluation waits
for both loading validation and a completed repair. D2T gate waits for its GPU scorer audit.
Withdrawn attribution adapters are preserved in a separate directory before fresh training.
Paper snapshots exclude incomplete/failed evaluations, check identical instance coverage and
adapter effectiveness, and preserve model/seed identity. Cached input signatures avoid rescanning
all trials when no outputs change. `runs/feeder/paper_build_latest.log` records draft builds;
GPU validation and final interpretation remain distinct from a successful PDF compilation.

Amendment46 larger-model gates cover the original three generality tasks; passing cells
automatically enter seed17 four-arm pilots and checked paired analyses. Failed cells stay
blocked. The D2T reference-text oracle failed, so its dependent gate was cancelled.

`103_a30_admission.py` runs each controller pass. Workload checks are pinned to g-04-01;
no other A30 node is admitted. A successful largest-projection SVD/memory check can widen
the pending full-FT repair job. A successful complete ten-system Gemma evaluation check
can widen the pending Gemma explicit-domain gate and two fmt contrastive evaluations.
Each keeps H100/H200 eligibility, excludes the other A30 nodes and records the original
Slurm specification, proof flags and update command in `runs/feeder/a30_admission.json`.
Running jobs are preserved. Unvalidated workloads remain on their existing partitions.

October7 backfill adjustment: the two pending A30 trainer checks and raw extractor diagnostic
use ten-minute limits; pending larger-model gates use one hour. These are scheduling limits,
not changes to sample sizes or objectives. The exact original specifications and updates are
saved in `runs/feeder/backfill_walltime_updates_2026-10-07.json`. The unmeasured A30
evaluation/SVD workloads retain thirty-minute limits.

## Commands

Run from `bidirectional/`:

```bash
source scripts/env.sh
python scripts/97_pipeline.py status
python scripts/97_pipeline.py debug
python scripts/97_pipeline.py start
```

`start` is idempotent. It launches a detached Slurm step inside the existing hold-node job,
or queues a CPU host if none is running. The existing pending successor will use the updated
launcher when it starts. Queued/running GPU jobs continue across CPU-host rollover; Slurm
priority may delay the successor or new GPU allocations.

To stop scheduling while submitted GPU jobs finish:

```bash
python scripts/97_pipeline.py stop
```

Resume with `start`. The CPU host and its renewal chain remain available for SSH. Stopping
GPU jobs or the shared CPU host is a separate action.

## Monitoring and debug evidence

- `runs/feeder/heartbeat.json`: controller PID, host, allocation, phase and last pass results.
- `runs/feeder/supervisor.json`: supervisor/worker identity, heartbeat and restart count.
- `runs/feeder/STATUS.md`: campaign state, queue, failure history and quarantine.
- `runs/feeder/feeder.log`, `supervisor.log`, `launcher.log`: execution and recovery logs.
- `runs/feeder/debug_latest.json`, `debug_history.jsonl`: recorded health checks.

October6 deployment checks: unit/regression checks passed; live controller crash recovery,
supervisor crash recovery, singleton launcher, detached stdin/session, fresh heartbeat, clean
watchdog/capacity/runner stages and pending CPU-host renewal all passed. Fault tests affect
only an idle controller/supervisor, never GPU jobs:

```bash
python scripts/97_pipeline.py debug --fault-test
python scripts/97_pipeline.py debug --supervisor-fault-test
```
