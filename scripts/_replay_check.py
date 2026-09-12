"""Does token-matched replay actually hit sft's budget where character-matched replay missed?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import ensure_obtune, resolve_model  # noqa: E402

ensure_obtune()
from transformers import AutoTokenizer  # noqa: E402

from obtune import prompts as oprompts  # noqa: E402

from bidir import arms as A, domains, prompts  # noqa: E402
from bidir.mixture import build_mixture  # noqa: E402
from bidir.train import measure_lengths  # noqa: E402

m = resolve_model("llama32-3b")
tok = AutoTokenizer.from_pretrained(m["hf_id"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def totals(cell: str, name: str, tokenizer) -> tuple[float, float]:
    mod = domains.get(cell)
    rows = build_mixture(A.resolve(name), cell, "train", seed=17, tokenizer=tokenizer)
    ex = [oprompts.to_trl_example(prompts.build_example(r.model_dump(), mod), tok) for r in rows]
    _, st = measure_lengths(ex, [r.task for r in rows], tok, int(m["max_seq_len"]))
    return st["sequence_tokens_total"], st["supervised_tokens_total"]


print(f"{'cell':<11}{'unit':<7}{'seq/sft':>9}{'sup/sft':>9}", flush=True)
for cell in ("fmt_novel", "automata", "mt_en-de", "sql"):
    s0, p0 = totals(cell, "sft", None)
    for unit, tkn in (("chars", None), ("tokens", tok)):
        s1, p1 = totals(cell, "replay", tkn)
        print(f"{cell:<11}{unit:<7}{s1/s0:>9.3f}{p1/p0:>9.3f}", flush=True)
