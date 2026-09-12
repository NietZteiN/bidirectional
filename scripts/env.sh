# Sourced by every sbatch script. The roots are defined ONCE here (obtune's rule); if the
# project moves, this file changes and nothing else does.
#
# EVERY name here is prefixed. That is not style: the cluster itself exports `SCRATCH`
# (= /scratch/$USER, which this account cannot write), so an unprefixed
# `SCRATCH="${SCRATCH:-...}"` silently keeps the cluster's value and sends HF_HOME to an
# unwritable path on every compute node while working fine on the login node. Measured
# 2026-09-10: it failed 8 tests in job 388948 and would have failed every job in the campaign.
export BIDIR_ROOT="${BIDIR_ROOT:-/work/jvl210002/migration/bidirectional}"
export BIDIR_ENV="${BIDIR_ENV:-/work/jvl210002/migration/envs/bidir-cu129}"
export BIDIR_SCORE_ENV="${BIDIR_SCORE_ENV:-/work/jvl210002/migration/envs/bidir-score}"
export BIDIR_SCRATCH="${BIDIR_SCRATCH:-/work/jvl210002/migration}"
# Large, disposable outputs: adapters and trial files. On SCRATCH -- 29 TB free and no MooseFS
# quota -- not on /work, which is at 69 % of a 1 TB quota.
#
# The path is /scratch/juno/<user>, namespaced by cluster. Note that the cluster's own $SCRATCH
# variable points at /scratch/<user>, which does NOT exist -- that stale value is what made
# `mkdir` fail and led to a day's work being planned around a quota problem that does not exist.
# `~/scratch` symlinks to the real path.
export BIDIR_OUT="${BIDIR_OUT:-/scratch/juno/jvl210002/bidir}"
export OBTUNE_ROOT="${OBTUNE_ROOT:-/work/jvl210002/migration/obtune}"

# $BIDIR_ENV/bin on PATH, not just the interpreter by absolute path: vLLM's engine core
# shells out to `ninja` from a child process, and without the bin dir on PATH engine startup
# dies as a bare FileNotFoundError nine frames down (obtune log/setup/2026-08-30).
export PATH="$BIDIR_ENV/bin:$PATH"
export PYTHONPATH="$BIDIR_ROOT/src:$OBTUNE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# HF_HOME is on SCRATCH, not /work, for three reasons obtune paid for first
# (obtune/scripts/env.sh, 2026-09-10):
#   * OWNERSHIP. The /work cache is shared by every project on this account, and on 2026-09-10 a
#     neighbour deleted 87 GB of CodeLlama weights from it to make room. A cache we do not own
#     can be reclaimed by someone else mid-campaign.
#   * SPEED. Measured on a compute node: /scratch reads at 14,186 MB/s against /work's 1,289
#     (11x), WekaFS with 32 MB readahead against MooseFS.
#   * SPACE. /work is a 1.1 TB per-user quota shared across projects; /scratch is a separate
#     30 TB one.
# THE TOKEN MUST TRAVEL WITH THE CACHE: it lives at $HF_HOME/token, and moving HF_HOME without
# it fails every gated repo (both Llamas, both Gemmas) with a 401 at the first hub call -- even
# when the weights are already on disk. Verified present before this was switched.
export HF_HOME="${HF_HOME:-/scratch/juno/$USER/hf_home}"
export TMPDIR="${TMPDIR:-$BIDIR_SCRATCH/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BIDIR_SCRATCH/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BIDIR_SCRATCH/cache/triton}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"  # needs nvcc; not present

# vLLM's engine core is spawned rather than forked. Without this, a job on the h100 partition's
# MIG node (g-06-01) dies with "Cannot re-initialize CUDA in forked subprocess", because several
# jobs share one physical card and a CUDA context already exists when the engine forks.
#
# CONSEQUENCE FOR EVERY SCRIPT THAT BUILDS AN ENGINE: spawn makes the engine-core subprocess
# RE-IMPORT the entry module, so engine construction at module level builds another engine in
# every child, recursively. The tell is the script's own first print appearing twice, and the
# symptom -- a job that starts, reports a healthy GPU and then never finishes -- reads as a
# broken node. `bidir.evaluate`, `bidir.mech.*` and every script here guard with
# `if __name__ == "__main__":`; keep it that way.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Share of the juno QoS pool this project may hold. The account is shared: obtune has claimed 2
# of the 4 running jobs the juno QoS allows, so taking more than 1 here squeezes a neighbour out.
#
# THE POOL IS NOT ONE PARTITION. `h200` and `normal` are both QoS=juno, so they are ONE budget of
# four -- a CPU analysis on `normal` blocks a GPU training job, and moving work there frees
# nothing. `dev` is QoS=juno-dev, a separate pool, which is why the CPU jobs in this project run
# there. `h100` and `a30` carry no QoS at all and are unaffected.
export BIDIR_JUNO_SHARE="${BIDIR_JUNO_SHARE:-1}"
export TOKENIZERS_PARALLELISM=false

# Fail loudly rather than three frames into a download. A cached model tree that is suddenly
# unreachable reads as a network error or a corrupt cache, not as a wrong path.
for _d in "$HF_HOME" "$TMPDIR"; do
  if ! mkdir -p "$_d" 2>/dev/null || [ ! -w "$_d" ]; then
    echo "FATAL: $_d is not writable — check BIDIR_SCRATCH in scripts/env.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
done
unset _d
