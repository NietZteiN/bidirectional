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
# Large, disposable outputs: adapters and trial files. Outside the repo so the git working tree
# stays small, and one variable to repoint if /scratch is ever provisioned for this account.
export BIDIR_OUT="${BIDIR_OUT:-/work/jvl210002/migration/bidir_out}"
export OBTUNE_ROOT="${OBTUNE_ROOT:-/work/jvl210002/migration/obtune}"

# $BIDIR_ENV/bin on PATH, not just the interpreter by absolute path: vLLM's engine core
# shells out to `ninja` from a child process, and without the bin dir on PATH engine startup
# dies as a bare FileNotFoundError nine frames down (obtune log/setup/2026-08-30).
export PATH="$BIDIR_ENV/bin:$PATH"
export PYTHONPATH="$BIDIR_ROOT/src:$OBTUNE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$BIDIR_SCRATCH/hf_home}"
export TMPDIR="${TMPDIR:-$BIDIR_SCRATCH/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BIDIR_SCRATCH/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BIDIR_SCRATCH/cache/triton}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"  # needs nvcc; not present
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
