#!/usr/bin/env bash
# Build the bidir uv environment: obtune's exact lock + scoring extras, in a SEPARATE venv
# so obtune's env (which its live SLURM jobs use) is never touched. RUN this, do not source it.
#   bash env/setup_env.sh            # replay obtune's lock, overlay torch cu129, add extras
set -euo pipefail

BIDIR_ENV="${BIDIR_ENV:-/work/jvl210002/migration/envs/bidir-cu129}"
OBTUNE_ROOT="${OBTUNE_ROOT:-/work/jvl210002/migration/obtune}"
LOCK="$OBTUNE_ROOT/env/lock-obtune.txt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRAS="$HERE/extras.txt"

export TMPDIR="${TMPDIR:-/work/jvl210002/migration/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$(dirname "$BIDIR_ENV")"
UV="${UV:-$(command -v uv)}"

[[ -d "$BIDIR_ENV" ]] || "$UV" venv "$BIDIR_ENV" --python 3.12
PY="$BIDIR_ENV/bin/python"

echo "[1/3] replaying $LOCK ($(wc -l < "$LOCK") packages)"
"$UV" pip install --python "$PY" -r "$LOCK"

# Same overlay as obtune/env/setup_env.sh: the lock pins torch +cu130, juno's driver is r550.
echo "[2/3] overlaying torch 2.11.0+cu129"
"$UV" pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"

# Scoring extras, CONSTRAINED by the lock: a pinned package cannot move. If this step
# fails, an extra genuinely conflicts with the training stack and it goes in a second
# CPU-only scoring env rather than into this one.
echo "[3/3] extras under lock constraints"
"$UV" pip install --python "$PY" -c "$LOCK" -r "$EXTRAS"

"$PY" - <<'PYEOF'
import torch, transformers, trl, peft, vllm, lm_eval, sacrebleu, comet, sqlglot, evaluate, nltk
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__, "vllm", vllm.__version__)
print("lm_eval", lm_eval.__version__, "sacrebleu", sacrebleu.__version__, "comet", comet.__version__, "sqlglot", sqlglot.__version__)
PYEOF
echo "ENV BUILD OK: $BIDIR_ENV"
