# Storage: what the campaign produces, and where it goes

*Measured 2026-09-10.*

## Where things live

| what | where | why |
|---|---|---|
| adapters, trials, results | **`/scratch/juno/jvl210002/bidir`** (`$BIDIR_OUT`) | 29 TB free, no MooseFS quota |
| models and datasets | `/work/jvl210002/migration/hf_home` (`$HF_HOME`) | 304 GB, shared with obtune |
| the repo, incl. `data/` | `/work/jvl210002/migration/bidirectional` | 145 MB; build reports are provenance |

### The scratch path is namespaced, and the cluster's own variable is stale

The real path is **`/scratch/juno/<user>`** — namespaced by cluster — and `~/scratch` symlinks to
it. The cluster exports `SCRATCH=/scratch/<user>`, which **does not exist**.

That stale variable cost real time on 2026-09-10: `mkdir /scratch/jvl210002` returns permission
denied, which reads as "this account has no scratch", and a whole storage plan was built around a
quota problem that does not exist. **Look one level up before concluding a filesystem is
unavailable.** `scripts/env.sh` sets `BIDIR_OUT` to the correct path and never reads `$SCRATCH`
— which it already avoided for a different reason, since an unprefixed `SCRATCH` also collides
with the cluster's value (see `CLAUDE.md` §2).

## What the grid produces

A LoRA adapter is bigger than it looks: **96 MB measured** for `olmo2-1b` at r=32 over seven
target modules in bf16, scaling with layers × hidden.

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
| full fine-tune checkpoints, final only | 180 |
| `trials.jsonl` across ~120 cells | 11 |
| **total** | **~455** |
| scratch headroom | **~29,000** |

It fits with room to spare. On `/work` it would not have: that volume is at 694 GB of a 1,000 GB
soft quota, about 306 GB of headroom.

## One setting still matters

**`save_strategy: "no"`.** TRL's default kept a checkpoint per epoch with
`save_total_limit=None`, which would put the campaign at **1,819 GB**. That is now comfortable on
scratch, but the setting stays for a reason that has nothing to do with space: **nothing in this
project ever loads an intermediate checkpoint.** There is no checkpoint selection and no resume,
only `final/`. Writing three copies of every adapter for nobody to read is waste wherever it
lands, and it slows every job by the time it takes to serialise them.

## Reaping is now optional

`scripts/91_reap_adapters.py` deletes adapters once their evaluation has landed. On `/work` it
was necessary; on scratch it is **housekeeping**, worth running if scratch fills or a tier is
finished with. It stays dry-run by default (`CLAUDE.md` §5) and never touches an unevaluated
adapter, an `sft` adapter in a mechanism domain, or an arm another arm initialises from.

## The risk that replaces the quota risk

**Scratch filesystems are usually purged.** This one has no stated policy that has been found,
and 868 GB of other work is currently sitting on it — but "no purge policy found" is not "no
purge policy". Two consequences:

- **Ask what the retention window is.** If it is 30 days, a campaign spanning October and
  November needs adapters copied somewhere durable before the writing phase, or regenerated.
- **`trials.jsonl` is what the paper is built from, not the adapters.** It is ~11 GB and belongs
  on `/work` at the point where tables are generated. Everything else is reproducible from the
  run manifests, at the cost of the GPU-hours.
