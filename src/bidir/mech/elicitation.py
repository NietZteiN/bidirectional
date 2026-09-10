"""Experiment 1: can prompting recover what tuning removed?

    python -m bidir.mech.elicitation --domain mt_en-de --model llama32-3b --arms base,sft,mix5

Four strategies (simple / few-shot / CoT / augmented) on the reverse direction, per arm. The
workshop paper measured partial CoT recovery on a collapsed 7B model — 5.3 % against the
untouched model's 21.9 % — which is the first suppression signal: a capability that is gone
cannot be prompted back at all, and one that is fully intact does not need to be.

The question this extends it to is whether the RECOVERED FRACTION is constant across domains.
A constant fraction says the same thing is being masked everywhere; a fraction that tracks
invertibility says prompting recovers only what the transform left recoverable.

This is just an eval pass with all four strategies, so it reuses `bidir.evaluate` rather than
reimplementing generation — the point is the analysis, and duplicating the pass would risk the
two disagreeing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bidir.config import GLOBAL_SEED, RESULTS_DIR
from bidir.schema import iter_jsonl


def recovered_fraction(run_dir: Path, metric: str = "strict") -> dict:
    """(best-strategy reverse) - (simple reverse), as a share of the base model's headroom.

    Expressed against the UNTOUCHED model's simple-prompt rate, because that is the number the
    capability had before training and therefore what "recovery" is measured against.
    """
    trials = list(iter_jsonl(run_dir / "trials.jsonl"))
    rates: dict[tuple[str, str], list[float]] = {}
    for t in trials:
        if t["direction"] != "reverse" or metric not in t:
            continue
        rates.setdefault((t["system"], t["strategy"]), []).append(float(t[metric]))
    mean = {k: sum(v) / len(v) for k, v in rates.items()}
    strategies = sorted({s for _, s in mean})
    systems = sorted({s for s, _ in mean})
    base_simple = mean.get(("base", "simple"), 0.0)
    out = {"base_simple": base_simple, "strategies": strategies, "systems": {}}
    for sysname in systems:
        by_strat = {s: mean.get((sysname, s), 0.0) for s in strategies}
        best = max(by_strat.values()) if by_strat else 0.0
        simple = by_strat.get("simple", 0.0)
        headroom = base_simple - simple
        out["systems"][sysname] = {
            "by_strategy": by_strat, "simple": simple, "best": best,
            "best_strategy": max(by_strat, key=by_strat.get) if by_strat else None,
            "prompt_gain": best - simple,
            # What share of what training removed can be prompted back. Undefined when the arm
            # lost nothing, which is the honest answer for `base` and for a healthy mix arm.
            "recovered_fraction": ((best - simple) / headroom) if headroom > 1e-9 else None,
        }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="an eval run already produced with all four strategies")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--arms", default="base,sft,mix5,mix50")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--metric", default="strict")
    a = ap.parse_args(argv)

    run = a.run
    if run is None:
        if not (a.domain and a.model):
            raise SystemExit("give --run, or --domain and --model to produce one")
        from bidir import evaluate
        rc = evaluate.main(["--domain", a.domain, "--model", a.model, "--seed", str(a.seed),
                            "--arms", a.arms, "--strategies", "simple,few_shot,cot,augmented",
                            "--limit", str(a.limit), "--tag", "elicitation"])
        if rc != 0:
            return rc
        runs = sorted(RESULTS_DIR.glob(f"*/{a.domain}/{a.model}/elicitation_s{a.seed}"))
        run = runs[-1]

    res = recovered_fraction(run, a.metric)
    print(f"\n=== elicitation ladder: {run} ===")
    for sysname, r in sorted(res["systems"].items()):
        frac = "n/a" if r["recovered_fraction"] is None else f"{r['recovered_fraction']:.1%}"
        print(f"  {sysname:<10s} simple={r['simple']:.4f} best={r['best']:.4f} "
              f"({r['best_strategy']}) gain={r['prompt_gain']:+.4f} recovered={frac}")
    (run / "elicitation.json").write_text(json.dumps(
        {"run": str(run), "metric": a.metric,
         "finished_utc": datetime.now(timezone.utc).isoformat(), **res}, indent=2))
    print(f"[elicitation] wrote {run}/elicitation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
