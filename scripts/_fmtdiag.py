"""What do the base model's outputs actually look like in the domains failing on format?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import domains, engine as eng, prompts  # noqa: E402
from bidir.config import ensure_obtune, load_config  # noqa: E402
from bidir.mixture import load_pairs  # noqa: E402

CELLS = ["diacritics", "automata", "algebra", "fmt_det00"]


def main() -> int:
    ensure_obtune()
    e = eng.get_engine("llama32-3b", load_config("domains/diacritics.yaml").get("engine", {}))
    for cell in CELLS:
        mod = domains.get(cell)
        cfg = load_config(f"domains/{cell}.yaml")
        insts = [p.model_dump() for p in load_pairs(cell, "test")][:6]
        for direction in ("forward", "reverse"):
            msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
            raw, _ = eng.generate(e, msgs, [None] * len(msgs), cfg.get("sampling", {}))
            outs = [prompts.extract_answer(r, "simple") for r in raw]
            scored = mod.score_batch(direction, outs, insts, cfg)
            print(f"\n===== {cell} / {direction} =====", flush=True)
            for i, (o, s) in enumerate(zip(outs, scored)):
                if i >= 3:
                    break
                tgt = prompts.completion_for(insts[i], direction)
                print(f"  GOLD  : {tgt[:110]!r}")
                print(f"  OUT   : {o[:180]!r}")
                print(f"  strict={s['strict']} off_target={s.get('off_target')} "
                      f"echo={s.get('echo')}", flush=True)
    return 0


if __name__ == "__main__":
    eng.shutdown_and_exit(main())
