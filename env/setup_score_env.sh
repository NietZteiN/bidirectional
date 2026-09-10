#!/usr/bin/env bash
# Second, scoring-only env: unbabel-comet needs transformers<5, which conflicts with the
# training stack's 5.14.1 lock. COMET-22 runs here (GPU via the cu129 torch wheel), invoked as
# a subprocess by bidir.domains.mt after generation. Nothing else goes in this env.
set -euo pipefail
SCORE_ENV="${SCORE_ENV:-/work/jvl210002/migration/envs/bidir-score}"
export TMPDIR="${TMPDIR:-/work/jvl210002/migration/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
UV="${UV:-$(command -v uv)}"
[[ -d "$SCORE_ENV" ]] || "$UV" venv "$SCORE_ENV" --python 3.12
PY="$SCORE_ENV/bin/python"
"$UV" pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.11.0+cu129" "unbabel-comet" "sacrebleu" "sentencepiece" "protobuf"
"$PY" - <<'PYEOF'
import torch, comet, transformers
print("torch", torch.__version__, "comet", comet.__version__, "transformers", transformers.__version__)
PYEOF
echo "SCORE ENV OK: $SCORE_ENV"
