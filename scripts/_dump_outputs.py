"""Dump FULL base-model outputs for the format-blocked domains, so the failures can be
categorised offline instead of guessed at from truncated prints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import domains, engine as eng, prompts  # noqa: E402
from bidir.config import RESULTS_DIR, ensure_obtune, load_config  # noqa: E402
from bidir.mixture import load_pairs  # noqa: E402

CELLS = {"diacritics": 120, "algebra": 80, "algebra_rev": 80, "fmt": 80}


def main() -> int:
    ensure_obtune()
    e = eng.get_engine("llama32-3b", load_config("domains/fmt.yaml").get("engine", {}))
    out_dir = RESULTS_DIR / "diag" / "format_blocked"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell, n in CELLS.items():
        mod = domains.get(cell)
        cfg = load_config(f"domains/{cell}.yaml")
        insts = [p.model_dump() for p in load_pairs(cell, "test")][:n]
        rows = []
        for direction in ("forward", "reverse"):
            msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
            raw, _ = eng.generate(e, msgs, [None] * len(msgs), cfg.get("sampling", {}))
            outs = [prompts.extract_answer(r, "simple") for r in raw]
            scored = mod.score_batch(direction, outs, insts, cfg)
            for i, o, r, s in zip(insts, outs, raw, scored):
                rows.append({"direction": direction, "pair_id": i["pair_id"],
                             "gold": prompts.completion_for(i, direction),
                             "output": o, "raw": r,
                             "strict": s["strict"], "off_target": s.get("off_target"),
                             "echo": s.get("echo")})
        (out_dir / f"{cell}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
        print(f"[dump] {cell}: {len(rows)} rows -> {out_dir / f'{cell}.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    eng.shutdown_and_exit(main())
