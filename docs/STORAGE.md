# Storage: what the campaign produces, and where it goes

*Measured 2026-09-10, before any adapter was trained.*

## The situation

`/work/jvl210002` holds **694 GB against a 1,000 GB soft quota** (1,100 GB hard), so there is
about **306 GB of headroom**. MooseFS keeps two replicas, so `realsize` is 1,387 GB, but the
quota counts logical size.

**`/scratch` exists on this cluster and this account cannot use it.** The filesystem is mounted
(401 TB, 72 % full, from `172.20.239.231`), the cluster exports `SCRATCH=/scratch/jvl210002`, and
that directory **does not exist and cannot be created** -- `mkdir` returns permission denied on
`/scratch`. Provisioning it is an administrator action.

> **Worth asking for.** A scratch directory would move every number in the table below off the
> quota entirely. It is the single change that would most simplify this campaign's storage, and
> it costs nothing but an email.

## What the grid produces

A LoRA adapter is bigger than it looks: **96 MB measured** for `olmo2-1b` at r=32 over seven
target modules in bf16, scaling with layers x hidden.

| model | per adapter |
|---|---|
| olmo2-1b | 96 MB |
| llama32-3b | 253 MB |
| gemma3-4b | 256 MB |
| llama31-8b | 386 MB |
| gemma3-12b | 543 MB |

| | GB |
|---|---|
| LoRA adapters, `final/` only | 275 |
| **...with three epoch checkpoints kept** | **1,099** |
| full fine-tune checkpoints, final only | 180 |
| ...with epoch checkpoints | 720 |
| `trials.jsonl` across ~120 cells | 11 |
| **total, as originally configured** | **1,819** |
| **total, final-only** | **455** |
| available | **306** |

## What was changed

**1. `save_strategy: "no"`.** TRL's default kept a checkpoint per epoch with
`save_total_limit=None`, quadrupling the footprint. Nothing in this project loads an
intermediate checkpoint: there is no checkpoint selection and no resume, only `final/`. This
alone takes 1,819 GB to 455 GB.

**2. Adapters and results left the git working tree.** `BIDIR_OUT` (default
`/work/jvl210002/migration/bidir_out`) holds `runs/` and `results/`. Same filesystem, so it saves
no quota -- what it buys is that `git status` never scans hundreds of gigabytes, the disposable
material is obviously disposable, and repointing at a real scratch filesystem later is one
environment variable rather than a rewrite. `data/` stays in the repo: it is 131 MB and its build
reports are provenance.

**3. `scripts/91_reap_adapters.py`.** 455 GB still exceeds 306 GB, so adapters are deleted once
their evaluation has landed. **Dry run by default**, per `CLAUDE.md` §5. It never touches an
adapter whose eval has not completed, an `sft` adapter in a mechanism domain (experiments 1, 2,
4, 5, 6 and 7 all start from `sft`, and experiment 7 needs the weights themselves), or any arm
another arm initialises from. It keeps `run_manifest.json` and `training_summary.json` -- the
provenance record is kilobytes -- and leaves a `REAPED` marker.

What that costs: re-scoring an arm means retraining it. The manifest records the git sha, the
resolved config and the seed, so it is reproducible, but reproducibility is not free.

## Running order

Reap after each tier's evaluations finish, not at the end:

```bash
python scripts/slurm/pipeline_grid.py --tier small
# ... evals land ...
python scripts/91_reap_adapters.py           # read the blast radius
python scripts/91_reap_adapters.py --yes     # then delete
```

`scripts/00_status.py` reports the current footprint so this does not have to be remembered.
