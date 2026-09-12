# Operating rules — `bidirectional/`

*Last updated: 2026-09-12.*

This project measures **directional collapse**: whether fine-tuning on one direction of a
paired task destroys the model's pre-existing ability in the other. The science is
[`acl2027-directional-collapse-plan.md`](acl2027-directional-collapse-plan.md); the execution
is [`RUN_PLAN.md`](RUN_PLAN.md). This file is how to work in the repo.

It does **not** inherit `obtune/CLAUDE.md`. That file governs a different paper, and its
hardest rule — the H1 quarantine — does not apply here because this project never reads
`obtune/data/quarantine/` and never trains on H1.

---

## 1. Compute

Same cluster as obtune: `juno-l-02` login, SLURM 24.11.5, **no GPU on the login node**.

- **Concurrency is the binding constraint, not GPU-hours.** The `h200` partition carries QoS
  `juno` with `MaxJobsPU=4`, shared across every project in this account. `juno-pri` gives 8
  slots at priority 200000 and jumps every other user on a shared cluster — it is available
  and is deliberately not the default.
- **Therefore: pack.** One job = one (domain, model, seed) cell, training every arm back to
  back (`scripts/20_train_pack.py`), then one dependent job evaluating the cell in a single
  vLLM multi-LoRA pass. A job per adapter would leave 594 small-grid adapters queued behind a
  4-wide door.
- Packs **skip arms whose `final/` already exists**, so a job killed at the 2-day walltime
  resumes by resubmission and costs only the arm it was inside.
- Measured on H200: **~18 min per 7B LoRA adapter**; ~12 min at 3-4B, ~30 min at 8B. Ask for
  roughly twice the estimate — over-asking costs queue position, under-asking costs the run.
- Spread across partitions, but know which ones share a pool. **`h200` and `normal` are both
  QoS `juno` — ONE budget of 4, not two**, so moving a CPU analysis to `normal` frees no GPU
  slot. `dev` is QoS `juno-dev`, a separate pool; `a30` (24 GB, <= 3B) and `h100` carry no QoS
  at all, which is where small-model packs go. Exclude `g-06-01` on `h100` for anything large:
  it advertises 3g.47gb MIG slices and obtune measured a 2.8x slowdown there.
- **The account is shared with obtune, which holds 2 of the 4 juno slots.** `$BIDIR_JUNO_SHARE`
  (default 1) is this project's share and `scripts/slurm/submit.py` refuses to exceed it,
  counting *running* jobs only — a job pending on Resources holds nothing.

### The login node caps virtual memory at 8 GB
`ulimit -v 8388608`. Loading a model, or even importing scipy under transformers, dies there
and works fine in a job. **Verify environments and run tokenizer/model tests under `sbatch`**
(`-p dev -t 00:30:00 --mem=32G` is enough; `dev` is CPU-only, 2 h limit).

### The login node is also the WRONG place to validate anything
It differs from a compute node in three ways that have each produced a wrong conclusion:

| | login | compute node |
|---|---|---|
| `sched_getaffinity` | 16 (unrestricted) | whatever `--cpus` gave you (8 of 64) |
| interpreter startup | warm in page cache | cold from MooseFS, seconds |
| GPU | none, so `vllm.platforms` never resolves `CudaPlatform` | resolves, and a CUDA-13 wheel dies |

So: a package that imports here is not verified, a criterion that passes here is not verified,
and a timeout that never fires here will fire there. `-p dev` is the cheap way to find out.

---

## 2. Environment

- Training: `/work/jvl210002/migration/envs/bidir-cu129` — obtune's 222-package lock replayed,
  torch overlaid to `2.11.0+cu129` (juno's driver is r550/CUDA 12.4, and the lock's cu130 build
  reports `cuda.is_available() == False` on every GPU node while `nvidia-smi` works).

### The lock does not capture the vLLM wheel
`vllm==0.26.0` resolves from PyPI to a **CUDA 13** build (`NEEDED libcudart.so.13`), which dies
on a GPU node with `ImportError: libcudart.so.13` the moment `vllm.platforms` resolves
`CudaPlatform`. It imports **fine on the login node**, where there is no GPU and that branch is
never taken — so it survived every check until the first GPU job (2026-09-12). obtune's working
env has `vllm-0.26.0+cu129` from the project's GitHub release asset; `+cu129` satisfies
`==0.26.0`, so nothing in the lock records the difference. `env/setup_env.sh` now installs that
wheel by URL and asserts the `+cu129` suffix. **A package that imports on the login node has not
been verified.**
- Scoring: `/work/jvl210002/migration/envs/bidir-score` — COMET only. `unbabel-comet` pins
  `transformers<5` against the training stack's 5.14.1, so it lives behind a subprocess
  boundary. Do not try to merge them.
- **Never add a dependency to obtune's lock.** New deps go in `env/extras.txt`, installed
  under `-c` the lock so a pinned package cannot move.

### Every variable `scripts/env.sh` exports is prefixed
The cluster itself exports `SCRATCH=/scratch/$USER`, which this account cannot write. An
unprefixed `SCRATCH="${SCRATCH:-...}"` therefore keeps the cluster's value and points `HF_HOME`
at an unwritable path **on every compute node, while working on the login node**. That cost a
whole test job on 2026-09-10. `env.sh` now fails loudly if `HF_HOME` or `TMPDIR` is not
writable, and `tests/test_arms_and_grid.py` asserts no unprefixed names are exported.

---

## 3. The rules that make a result mean something

Each of these is enforced by a test, because each is a way to produce a full, plausible,
wrong table.

1. **Direction is how a row is READ, never field order.** `side_a`/`side_b` are fixed;
   `task` says which way. This is what makes "reverse data is free" literally true: a mix
   arm's reverse rows are byte-identical in content to the forward rows they replaced.
2. **Every `mix*` arm REPLACES forward pairs with their own reversal, partitioned by
   `pair_id`.** Instance count, sequence tokens and optimizer steps stay matched to `sft` at
   every dose; only `flip` and `fwd2x` cost 2x. Partitioning by row instead would let one pair
   be seen both ways, making a low dose a small `flip`.
3. **The reverse training prompt is byte-identical to the reverse eval prompt**, and one
   system prompt serves both directions. Two personas would let disjoint circuits masquerade
   as bidirectionality; a prompt mismatch would make reverse accuracy a measurement of the
   mismatch, and it is invisible in the loss curve.
4. **Echo and off-target never count as success.** Both rose under forward-only training in
   the workshop paper. In `fmt`'s `json-yaml` this is a validity condition, not hygiene: JSON
   is valid YAML, so an echoing model parses *and* compares structurally equal.
5. **Thresholds τ come from the BASE model's own distribution, before any tuned model is
   scored** (`scripts/15_base_gate.py --write`), and are written into the domain config with
   the date. A missing threshold **raises** rather than defaulting.
6. **One eval pass per table; contrasts only within a pass.** Greedy decoding is not bitwise
   reproducible across passes — vLLM batches continuously. Measure the floor once per model
   with `scripts/30_determinism_floor.py` and never quote a cross-pass difference finer than it
   (0.5 pp until measured otherwise).
7. **The overlap check for every probe is written down before it runs.** In the workshop paper
   74 of HumanEval+'s 164 problems were in the training split, which made forward-only tuning
   look free.
8. **Adapter-effectiveness is asserted on every eval, after rows are on disk.** An adapter that
   silently failed to load produces a table that is a perfect copy of the base, row for row.
   The guard must fail the job; the evidence must survive it.
9. **Eval sets never appear in any training split**, checked on CONTENT with a domain-supplied
   `content_key` — and the check is a build-time failure, not a warning. It has already caught
   one real leak (Spider) and one false positive (CRUXEval).
10. **Nothing that executes a program may size its own parallelism.** `exec_workers` comes from
    `len(os.sched_getaffinity(0))`, and a pinned value above the allocation raises. A hardcoded
    32 on an 8-CPU allocation scored 40 known-correct `code` answers at **0.375**; the same 40
    at 4 workers scored **1.000**. obtune's executor kills an oversubscribed child on wall clock
    and `exec_equivalence` folds that timeout into `status="error"`, which the criterion reads as
    a wrong answer — so the failure is silent, one-directional, and would have capped the
    known-positive control.
11. **A timeout is a measurement failure, not a wrong answer.** `coverage`'s budget used to
    cover interpreter startup, so it timed out on all 40 gold answers on a compute node (cold
    MooseFS) and on none on the login node. Startup is now measured per process and added; a
    timeout is retried once, serially.
12. **A criterion is only trusted after its oracle test runs ON A COMPUTE NODE.**
    `scripts/16_audit_criteria.py` passes gold/echo/empty/garbage through every scorer. All
    three code-executing domains passed on the login node and failed on a compute node.

---

## 4. Data sources with a trap in them

- **Spider.** Questions come from `xlangai/spider` (the official, database-disjoint split).
  Databases come from `prem-research/spider`, the only Hub copy with the 169 sqlite files —
  and whose own `train.json` **pools dev**: all 20 dev databases, 384 duplicate (question,
  query) pairs, 130 instances leaking into eval. `build_pairs` asserts disjointness.
- **FLORES-200.** `openlanguagedata/flores_plus` and `facebook/flores` are gated (403 here).
  `haoranxu/FLORES-200` is ungated and carries the same 1,012 devtest sentences.
- **WebNLG** ships as a loading script, which `datasets>=4` refuses; read the
  `refs/convert/parquet` branch.
- **CRUXEval** is 800 problems in one split, so `exec` is a deliberately small cell. It says so
  in its config rather than being resampled up to look uniform.

---

## 5. Destructive commands — human in the loop

**Never run `rm -rf`, or any recursive or bulk deletion, without confirming first.** State the
blast radius, dry-run it with `ls`/`find`, wait for explicit approval, then run the narrowest
command that does the job. Trained adapters and built corpora cost GPU-hours and CPU-days.

---

## 6. Storage

Adapters, trials and results go to **`/scratch/juno/jvl210002/bidir`** (`$BIDIR_OUT`): 29 TB
free, no MooseFS quota. The grid produces ~455 GB, which fits easily there and would not fit on
`/work`, which is at 694 GB of a 1,000 GB soft quota. Numbers in
[`docs/STORAGE.md`](docs/STORAGE.md).

**The scratch path is `/scratch/juno/<user>`, namespaced by cluster, and `~/scratch` points at
it. The cluster's own `$SCRATCH` variable points at `/scratch/<user>`, which does not exist.**
That stale value made `mkdir` fail and produced a confident, wrong conclusion that this account
had no scratch at all. Look one level up before deciding a filesystem is unavailable.

Two rules survive independent of space:

- **`save_strategy: "no"`.** Nothing here loads an intermediate checkpoint — no checkpoint
  selection, no resume, only `final/` — so writing three per adapter is waste wherever it lands.
- **A vLLM script must hard-exit.** `bidir.engine.shutdown_and_exit(main())`, not
  `sys.exit(main())`. With `spawn`, the engine-core child does not reliably come back: job
  391263 printed its final gate summary and then held an H200 for another **6m14s** without
  writing its status file. The share is one running job, and a job killed at the walltime is
  recorded as a failure — so a teardown hang turns a completed eval into a lost one.
- **`trials.jsonl` is what the paper is built from.** Scratch filesystems are usually purged and
  no retention policy for this one has been established; ask. Copy trials to `/work` before the
  writing phase. Everything else is reproducible from the run manifests, at the cost of its
  GPU-hours.

`scripts/91_reap_adapters.py` (dry run by default) is housekeeping rather than necessity now.

## 7. Provenance

`data/` is committed (131 MB, and its build reports are provenance); `runs/` and `results/` are
generated and live outside the tree. Provenance lives in the files
themselves: every adapter carries a `run_manifest.json` with the git sha, the resolved config,
the script hashes and the realised direction balance; every result carries a `summary.json`
with the engine version, the systems and the adapter-effectiveness report.

**No number in the paper is hand-typed.** Tables come from `scripts/51_tables.py` and figures
from `scripts/52_figs.py`, both reading `trials.jsonl`; `paper/PROVENANCE.json` maps each table
to the runs it was computed from. Re-running the eval regenerates them.
