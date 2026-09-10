#!/usr/bin/env python
"""Base gate: can this model do BOTH directions before we touch it, and where is tau?

    python scripts/15_base_gate.py --domain mt_en-de --model llama32-3b --write

Two jobs, and the second is the one that matters for the paper's honesty:

1. GATE. A model enters the panel for a domain only if the untouched checkpoint can already
   do both directions well enough for a collapse to be visible. `format_fail` (off-target or
   unparseable) must be <= 0.15 and both directions must clear `min_base_rate`. A base model
   at 0.02 reverse has no room to lose anything, and a null on it means nothing.

2. THRESHOLDS. tau is read off the BASE model's own score distribution — the quantile named
   by `tau_quantile` — and written into the domain config with the date and the run that
   produced it. Doing this before any fine-tuned model is scored is what stops the criterion
   from being chosen, consciously or not, to suit the result (plan §7).

For SQL it also measures the frozen round-trip parser's accuracy on the REFERENCE questions,
which is the criterion's ceiling and is reported beside every reverse number.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from bidir import domains, engine as eng, prompts  # noqa: E402
from bidir.config import CONFIGS_DIR, GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config  # noqa: E402
from bidir.mixture import load_pairs  # noqa: E402

#: Which continuous metric tau is set on, per domain. Domains whose criterion is exact
#: (parse + structural equality) have no tau and are absent here.
TAU_METRIC = {"mt": "comet", "sql": None, "code": None, "fmt": None, "d2t": "chrf2"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--limit", type=int, default=200, help="instances; the gate does not need the full eval set")
    ap.add_argument("--tau-quantile", type=float, default=0.25,
                    help="tau = this quantile of the base model's scores on the instances it "
                         "did not fail. 0.25 keeps the criterion at 'clearly worse than the "
                         "base typically manages' rather than at the base's median.")
    ap.add_argument("--min-base-rate", type=float, default=0.10)
    ap.add_argument("--max-format-fail", type=float, default=0.15)
    ap.add_argument("--write", action="store_true", help="write tau into the domain config")
    a = ap.parse_args()

    ensure_obtune()
    mod = domains.get(a.domain)
    cfg = load_config(f"domains/{a.domain}.yaml")
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    e = eng.get_engine(a.model, cfg.get("engine", {}))

    metric = TAU_METRIC.get(a.domain.split("_")[0])
    report: dict = {"domain": a.domain, "model": a.model, "n": len(insts), "tau_metric": metric,
                    "tau_quantile": a.tau_quantile, "directions": {}}
    thresholds: dict[str, dict[str, float]] = {}

    for direction in ("forward", "reverse"):
        msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
        raw, _ = eng.generate(e, msgs, [None] * len(msgs), cfg.get("sampling", {}))
        outs = [prompts.extract_answer(r, "simple") for r in raw]

        # tau must exist before score_batch runs, and score_batch is what produces the
        # distribution tau is read from. So: score once with a permissive tau to get the
        # metric column, choose tau from it, then report the rate under the chosen tau.
        scored = mod.score_batch(direction, outs, insts, {**cfg, "thresholds": {direction: {metric or "chrf2": -1e9}}})
        fmt_fail = sum(r.get("off_target", 0) or r.get("empty_output", 0) for r in scored) / max(1, len(scored))
        row: dict = {"format_fail": fmt_fail,
                     "echo": sum(r.get("echo", 0) for r in scored) / max(1, len(scored)),
                     "rate_uncapped": sum(r["strict"] for r in scored) / max(1, len(scored))}
        if metric:
            vals = sorted(r[metric] for r in scored if not r.get("off_target") and not r.get("empty_output"))
            if len(vals) < 20:
                raise SystemExit(f"only {len(vals)} scorable {direction} generations — the model "
                                 f"fails this domain's format, not its content")
            tau = statistics.quantiles(vals, n=100)[max(0, min(98, int(a.tau_quantile * 100) - 1))]
            thresholds[direction] = {metric: round(float(tau), 6)}
            rescored = mod.score_batch(direction, outs, insts, {**cfg, "thresholds": thresholds})
            row.update({"tau": float(tau), "rate": sum(r["strict"] for r in rescored) / max(1, len(rescored)),
                        f"{metric}_mean": sum(vals) / len(vals)})
        else:
            row["rate"] = row["rate_uncapped"]
        report["directions"][direction] = row
        print(f"  {direction:<8s} rate={row['rate']:.4f} format_fail={fmt_fail:.3f} "
              f"echo={row['echo']:.3f}" + (f" tau={row['tau']:.4f}" if "tau" in row else ""), flush=True)

    if a.domain == "sql":
        gold_q = [i["side_a"] for i in insts]
        rt = mod.roundtrip_sql(gold_q, insts, cfg)
        ceiling = sum(mod._exec_match(mod.db_path(i["meta"]["db_id"]), q, i["side_b"])
                      for q, i in zip(rt, insts)) / max(1, len(insts))
        report["roundtrip_ceiling"] = ceiling
        print(f"  round-trip parser on REFERENCE questions: {ceiling:.4f} (the reverse criterion's ceiling)")

    ok = all(r["rate"] >= a.min_base_rate for r in report["directions"].values()) and \
         all(r["format_fail"] <= a.max_format_fail for r in report["directions"].values())
    report["passes_gate"] = ok
    report["gate_rule"] = (f"both directions rate >= {a.min_base_rate} and format_fail <= {a.max_format_fail}")

    out_dir = RESULTS_DIR / "base_gates" / a.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (out_dir / f"{a.model}_{stamp}.json").write_text(json.dumps(report, indent=2))

    if a.write and thresholds:
        path = CONFIGS_DIR / "domains" / f"{a.domain}.yaml"
        doc = yaml.safe_load(path.read_text()) or {}
        doc.setdefault("thresholds", {}).update(thresholds)
        doc["thresholds_provenance"] = {
            "model": a.model, "date": stamp, "quantile": a.tau_quantile, "n": len(insts),
            "note": "read from the BASE model's score distribution before any fine-tuned model was scored",
        }
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        print(f"[gate] wrote thresholds into {path}")

    print(f"[gate] {a.model} on {a.domain}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
