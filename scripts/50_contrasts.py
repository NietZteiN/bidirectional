#!/usr/bin/env python
"""Paired arm-vs-arm contrasts with cluster-bootstrap CIs, and the gate verdict.

    python scripts/50_contrasts.py --run results/2026-09-20/mt_en-de/llama32-3b/gate_s17
    python scripts/50_contrasts.py --gate            # every gate cell, plus the pass rule

PAIRED, not two independent rates. Both arms are scored on the same pairs, so a resample
draws a PAIR once and takes both arms' trials for it. Bootstrapping the arms separately would
ignore that pairing and inflate every interval — which, for the nulls this design turns on,
is the difference between "no effect, tightly bounded" and "no effect, but we could not have
detected one".

Contrasts are only ever taken WITHIN one evaluation pass. Greedy decoding is not bitwise
reproducible across passes (vLLM batches continuously), so a cross-pass difference finer than
0.5 pp is engine noise, not an effect.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import GLOBAL_SEED, RESULTS_DIR  # noqa: E402
from bidir.schema import iter_jsonl  # noqa: E402

#: (a, b) meaning "a minus b" — the primary contrasts, pre-registered in RUN_PLAN.md §5.
DEFAULT_CONTRASTS = [("sft", "base"), ("mix5", "sft"), ("mix50", "sft"),
                     ("replay", "sft"), ("mix50", "replay"), ("mix50", "flip"), ("rev", "base")]


def paired_delta(trials: Sequence[dict], a: str, b: str, metric: str, direction: str,
                 strategy: str = "simple", n_boot: int = 2000, seed: int = GLOBAL_SEED):
    by_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t["direction"] != direction or t["strategy"] != strategy or metric not in t:
            continue
        if t["system"] in (a, b):
            by_pair[t["pair_id"]][t["system"]].append(float(t[metric]))
    pairs = [p for p, v in by_pair.items() if a in v and b in v]
    if not pairs:
        return None

    def stat(sample: Sequence[str]) -> float:
        va = [x for p in sample for x in by_pair[p][a]]
        vb = [x for p in sample for x in by_pair[p][b]]
        return (sum(va) / len(va) - sum(vb) / len(vb)) * 100.0

    point = stat(pairs)
    rng = random.Random(seed)
    boots = [stat([pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]) for _ in range(n_boot)]
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
    mean_a = sum(x for p in pairs for x in by_pair[p][a]) / sum(len(by_pair[p][a]) for p in pairs)
    mean_b = sum(x for p in pairs for x in by_pair[p][b]) / sum(len(by_pair[p][b]) for p in pairs)
    return {"a": a, "b": b, "metric": metric, "direction": direction, "n_pairs": len(pairs),
            "mean_a": mean_a * 100, "mean_b": mean_b * 100, "delta_pp": point,
            "ci_lo": lo, "ci_hi": hi, "spans_zero": lo <= 0 <= hi}


def analyse_run(run_dir: Path, metric: str = "strict", n_boot: int = 2000) -> dict:
    trials = list(iter_jsonl(run_dir / "trials.jsonl"))
    systems = {t["system"] for t in trials}
    out: dict[str, Any] = {"run": str(run_dir), "systems": sorted(systems), "contrasts": []}
    for direction in ("reverse", "forward"):
        for a, b in DEFAULT_CONTRASTS:
            if a in systems and b in systems:
                d = paired_delta(trials, a, b, metric, direction, n_boot=n_boot)
                if d:
                    out["contrasts"].append(d)
    return out


def fmt(d: dict) -> str:
    star = "" if d["spans_zero"] else " *"
    return (f"  {d['direction']:<8s} {d['a']}-{d['b']:<8s} "
            f"{d['delta_pp']:+7.2f} pp [{d['ci_lo']:+6.2f}, {d['ci_hi']:+6.2f}]"
            f"  ({d['mean_a']:.2f} vs {d['mean_b']:.2f}, n={d['n_pairs']}){star}")


def learnable_floor(base_rev: float) -> float:
    """The bar `rev` must clear for the reverse direction to count as learnable.

    ADDITIVE, NOT MULTIPLICATIVE. This was `max(0.05, 2 * base_rev)`, which is unsatisfiable
    whenever the base already exceeds 0.5 -- and a metric-based tau puts the base at
    `1 - tau_quantile` (0.75) BY CONSTRUCTION, so for every MT cell the bar was 1.50 and no
    rate can reach it. `verdict["passes"]` requires a collapsing NLP cell to also clear this
    bar, so the gate could not have passed on MT evidence however the data fell, and the
    diagnostic would have printed "[KILL-GATE: rev ~ 0]" with `rev` sitting at 0.90.

    The prereg's words are "`rev` strict reverse is well above zero". An additive margin says
    that and degrades correctly at both ends: at base_rev ~ 0 the bar is 0.05, which is
    literally "well above zero"; at base_rev 0.75 it is 0.80, which is attainable and still
    means training on reverse data taught something the untouched model did not have.

    The 0.95 cap is there so the bar stays reachable at the top of the range too: a base
    already at 0.95 leaves less than the margin in headroom, and such a base has ALREADY
    demonstrated the thing the kill-gate asks about -- the untouched model can do the reverse
    direction -- so the clause should be trivially satisfied rather than arithmetically
    impossible. No floor may ever exceed a rate that can be achieved; that is the property the
    old multiplicative form violated.
    """
    return max(0.05, min(base_rev + 0.05, 0.95))


def gate_verdict(runs: list[Path], metric: str = "strict") -> dict:
    """The pre-registered rule from RUN_PLAN.md §5. Stated here in code so the verdict is a
    computation and not a reading."""
    cells: dict[str, dict] = {}
    for run in runs:
        trials = list(iter_jsonl(run / "trials.jsonl"))
        cell = run.parent.parent.name
        rates: dict[tuple[str, str], float] = {}
        for t in trials:
            if t["strategy"] != "simple" or metric not in t:
                continue
            rates.setdefault((t["system"], t["direction"]), [])
            rates[(t["system"], t["direction"])].append(float(t[metric]))
        mean = {k: sum(v) / len(v) for k, v in rates.items()}
        base_rev = mean.get(("base", "reverse"), 0.0)
        sft_rev = mean.get(("sft", "reverse"), 0.0)
        rev_rev = mean.get(("rev", "reverse"), 0.0)
        rel = (sft_rev - base_rev) / base_rev if base_rev > 0 else float("nan")
        cells[cell] = {
            "base_reverse": base_rev, "sft_reverse": sft_rev, "rev_reverse": mean.get(("rev", "reverse"), 0.0),
            "mix5_reverse": mean.get(("mix5", "reverse"), 0.0), "mix50_reverse": mean.get(("mix50", "reverse"), 0.0),
            "sft_forward": mean.get(("sft", "forward"), 0.0), "base_forward": mean.get(("base", "forward"), 0.0),
            "relative_reverse_change": rel,
            "collapses": bool(base_rev > 0 and rel <= -0.5),
            "reverse_learnable": rev_rev >= learnable_floor(base_rev),
            "learnable_floor": learnable_floor(base_rev),
        }
    nlp = [c for c in cells if not c.startswith("code")]
    verdict = {
        "cells": cells,
        "rule": ("PASS if >=1 NLP cell collapses (sft-base <= -50 % relative on strict reverse) "
                 "AND `rev` shows the reverse direction is learnable in that cell. "
                 "The IFEval control is read from the probe job, not from here."),
        "collapsing_nlp_cells": [c for c in nlp if cells[c]["collapses"]],
        "kill_gate_ok": [c for c in cells if cells[c]["reverse_learnable"]],
    }
    verdict["passes"] = bool(verdict["collapsing_nlp_cells"]
                             and any(c in verdict["kill_gate_ok"] for c in verdict["collapsing_nlp_cells"]))
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--gate", action="store_true", help="find every gate_* run and apply the pass rule")
    ap.add_argument("--metric", default="strict")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    runs = [a.run] if a.run else sorted(RESULTS_DIR.glob("*/*/*/gate_s*")) if a.gate else []
    if not runs:
        print("nothing to analyse (pass --run <dir> or --gate)", file=sys.stderr)
        return 1

    all_out = []
    for run in runs:
        if not (run / "trials.jsonl").exists():
            continue
        res = analyse_run(run, a.metric, a.n_boot)
        all_out.append(res)
        print(f"\n=== {run} ===")
        for d in res["contrasts"]:
            print(fmt(d))
        (run / "contrasts.json").write_text(json.dumps(res, indent=2))

    if a.gate:
        v = gate_verdict([r for r in runs if (r / "trials.jsonl").exists()], a.metric)
        print("\n=== GATE ===")
        for cell, c in sorted(v["cells"].items()):
            # The kill-gate note names both numbers. Saying only "rev ~ 0" was how the old
            # multiplicative floor hid the fact that it was unsatisfiable rather than unmet.
            kill = ("" if c["reverse_learnable"] else
                    f"  [KILL-GATE: rev={c['rev_reverse']:.3f} < floor {c['learnable_floor']:.3f}]")
            print(f"  {cell:<12s} base_rev={c['base_reverse']:.4f} sft_rev={c['sft_reverse']:.4f} "
                  f"({c['relative_reverse_change']:+.1%})  rev_ceiling={c['rev_reverse']:.4f}  "
                  f"{'COLLAPSE' if c['collapses'] else 'no collapse'}{kill}")
        print(f"\n  verdict: {'PASS — proceed to Phase 2' if v['passes'] else 'FAIL — reset to the boundary-conditions paper'}")
        print(f"  rule: {v['rule']}")
        out = RESULTS_DIR / "gate_verdict.json"
        out.write_text(json.dumps(v, indent=2))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
