"""Experiment 6: does the model still listen to the direction instruction?

    python -m bidir.mech.sensitivity --domain mt_en-de --model llama32-3b --arms base,sft,mix5

Hold the INPUT fixed and vary only the instruction, then measure how much the output changes.
The unconditional-mapping hypothesis (H1) predicts near-zero sensitivity after forward-only
training — the model applies the trained mapping whatever it is asked — and restored
sensitivity after a small reverse dose.

Sensitivity is measured two ways, because either alone is arguable:
  * `output_divergence` — 1 - chrF similarity between the outputs under the two instructions.
    Zero means the instruction changed nothing.
  * `applied_forward_rate` — the share of REVERSE-instructed generations that are in fact the
    forward output. This is the direct reading of "it applied the trained mapping regardless",
    and it is what echo rate approximates without being able to name it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bidir import arms as arm_registry
from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config
from bidir.mixture import load_pairs
from bidir.train import adapter_dir


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="base,sft,mix5,mix50")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)

    ensure_obtune()
    import sacrebleu

    from bidir import engine as eng

    dcfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    chrf = sacrebleu.CHRF(word_order=2)

    systems: dict[str, Optional[str]] = {}
    for name in [x.strip() for x in a.arms.split(",") if x.strip()]:
        spec = arm_registry.resolve(name)
        if not spec.trains:
            systems[name] = None
            continue
        p = adapter_dir(a.domain, a.model, name, a.rank, a.seed) / "final"
        if not p.exists():
            raise SystemExit(f"adapter missing: {p}")
        systems[name] = str(p)

    e = eng.get_engine(a.model, dcfg.get("engine", {}))
    # The SAME input under both instructions. For a forward instance the input is side_a; the
    # reverse instruction is then asking the model to invert something it was not given, which
    # is the point — a direction-sensitive model should refuse or echo, not translate.
    fwd_msgs = [prompts.build_messages(i, "forward", mod, "simple") for i in insts]
    rev_on_same_input = [prompts.build_messages({**i, "side_b": i["side_a"]}, "reverse", mod, "simple")
                         for i in insts]

    rows = []
    for name, adapter in systems.items():
        f_raw, _ = eng.generate(e, fwd_msgs, [adapter] * len(insts), dcfg.get("sampling", {}))
        r_raw, _ = eng.generate(e, rev_on_same_input, [adapter] * len(insts), dcfg.get("sampling", {}))
        f_out = [prompts.extract_answer(x) for x in f_raw]
        r_out = [prompts.extract_answer(x) for x in r_raw]
        div = [1.0 - chrf.sentence_score(r or "", [f or ""]).score / 100.0 for f, r in zip(f_out, r_out)]
        applied_fwd = [1 if chrf.sentence_score(r or "", [f or ""]).score >= 90.0 else 0
                       for f, r in zip(f_out, r_out)]
        rows.append({"system": name, "n": len(insts),
                     "output_divergence": sum(div) / len(div),
                     "applied_forward_rate": sum(applied_fwd) / len(applied_fwd)})
        print(f"  {name:<10s} divergence={rows[-1]['output_divergence']:.4f} "
              f"applied_forward={rows[-1]['applied_forward_rate']:.4f}", flush=True)

    out_dir = RESULTS_DIR / "mech" / "sensitivity" / f"{a.domain}__{a.model}_s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(json.dumps(
        {"domain": a.domain, "model": a.model, "seed": a.seed, "n_instances": len(insts),
         "hypothesis": "H1 predicts divergence -> 0 and applied_forward_rate -> 1 after sft, "
                       "both restored by a small reverse dose",
         "finished_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2))
    print(f"[sensitivity] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
