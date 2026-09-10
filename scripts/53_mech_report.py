#!/usr/bin/env python
"""Collect the mechanism results into one verdict on erased-versus-suppressed.

    python scripts/53_mech_report.py

RQ5 is decided by three cheap behavioral experiments read TOGETHER, so this reads them
together rather than leaving the synthesis to prose:

  1  elicitation   a capability that is gone cannot be prompted back at all; one that is
                   intact does not need to be. A non-zero recovered fraction is suppression.
  2  alpha_scale   reverse collapsing at small alpha while forward still climbs means the
                   collapse is a cheap direction in weight space.
  3  relearn       recovery from `sft` that outruns the never-had control (`fmt_novel`) is
                   latent knowledge; curves that match are erasure.
  7  spectral      reverse capability returning after a closed-form filter on the update, with
                   NO retraining and no data, is the strongest form of the suppression claim:
                   the capability was masked by a removable residue rather than removed.

Each experiment votes, and the script says which way and how strongly. It does NOT average
them into a score: they measure different things, and a disagreement is a finding rather than
noise to be smoothed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR  # noqa: E402
from bidir.schema import iter_jsonl  # noqa: E402


def read_elicitation() -> dict:
    out = {}
    for f in RESULTS_DIR.glob("*/*/*/*/elicitation.json"):
        d = json.loads(f.read_text())
        domain = f.parent.parent.parent.name
        sft = d.get("systems", {}).get("sft", {})
        if sft.get("recovered_fraction") is not None:
            out[domain] = {"recovered_fraction": sft["recovered_fraction"],
                           "best_strategy": sft.get("best_strategy"),
                           "prompt_gain_pp": 100 * sft.get("prompt_gain", 0.0)}
    return out


def read_alpha() -> dict:
    out = {}
    for f in (RESULTS_DIR / "mech" / "alpha_scale").glob("*/curve.json"):
        d = json.loads(f.read_text())
        fwd = {r["alpha"]: r["strict"] for r in d["rows"] if r["direction"] == "forward"}
        rev = {r["alpha"]: r["strict"] for r in d["rows"] if r["direction"] == "reverse"}
        if not fwd or not rev:
            continue
        a_max = max(fwd)
        rev0, revmax = rev.get(0.0, 0.0), rev[a_max]
        # The alpha at which reverse has lost half of what it will lose.
        half = rev0 - 0.5 * (rev0 - revmax)
        a_half_rev = next((a for a in sorted(rev) if rev[a] <= half), a_max)
        fwd_at = fwd.get(a_half_rev, 0.0)
        fwd_frac = (fwd_at - fwd.get(0.0, 0.0)) / max(1e-9, fwd[a_max] - fwd.get(0.0, 0.0))
        out[d["domain"]] = {"alpha_half_reverse_loss": a_half_rev,
                            "forward_progress_there": fwd_frac,
                            "reverse_base": rev0, "reverse_full": revmax}
    return out


def read_relearn() -> dict:
    ks = [10, 50, 200, 1000]
    curves: dict[str, dict[str, float]] = defaultdict(dict)
    for run in RESULTS_DIR.glob("*/*/*/relearn_s*"):
        if not (run / "trials.jsonl").exists():
            continue
        domain = run.parent.parent.name
        acc: dict[str, list[float]] = defaultdict(list)
        for t in iter_jsonl(run / "trials.jsonl"):
            if t["direction"] == "reverse" and t.get("strategy") == "simple" and "strict" in t:
                acc[t["system"]].append(float(t["strict"]))
        for sysname, v in acc.items():
            curves[domain][sysname] = sum(v) / len(v)
    out = {}
    control = curves.get("fmt_novel", {})
    for domain, c in curves.items():
        if domain == "fmt_novel":
            continue
        row = {f"relearn{k}": c.get(f"relearn{k}") for k in ks}
        row["sft"] = c.get("sft")
        row["base"] = c.get("base")
        if control:
            row["control"] = {f"relearn{k}": control.get(f"relearn{k}") for k in ks}
            faster = [k for k in ks
                      if c.get(f"relearn{k}") is not None and control.get(f"relearn{k}") is not None
                      and c[f"relearn{k}"] > control[f"relearn{k}"]]
            row["outruns_never_had_at_k"] = faster
        out[domain] = row
    return out


def read_spectral() -> dict:
    """Experiment 7: does removing a spectral component of the update bring the reverse back?"""
    out = {}
    for f in (RESULTS_DIR / "mech" / "spectral").glob("*/spectral.json"):
        d = json.loads(f.read_text())
        rows = d["rows"]
        def at(variant, direction):
            for r in rows:
                if r["variant"] == variant and r["direction"] == direction:
                    return r["strict"]
            return None
        base_rev, un_rev = at("base", "reverse"), at("unrepaired", "reverse")
        un_fwd = at("unrepaired", "forward")
        best, best_tau = un_rev, None
        for r in rows:
            if r["direction"] == "reverse" and r["variant"].startswith("dghard") and r["strict"] > (best or 0):
                best, best_tau = r["strict"], r["variant"]
        fwd_at_best = next((r["strict"] for r in rows
                            if r["direction"] == "forward" and r["variant"] == best_tau), None)
        out[d["domain"]] = {
            "base_reverse": base_rev, "sft_reverse": un_rev, "best_repaired_reverse": best,
            "best_variant": best_tau, "sft_forward": un_fwd, "forward_at_best": fwd_at_best,
            "lowrank_caveat": d.get("delta_is_lowrank_by_construction", True),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    elic, alpha, relearn = read_elicitation(), read_alpha(), read_relearn()
    spectral = read_spectral()
    votes = {}

    print("=== 1. elicitation: can prompting recover what tuning removed? ===")
    for d, r in sorted(elic.items()):
        print(f"  {d:<12s} recovered {r['recovered_fraction']:.1%} of the loss via {r['best_strategy']} "
              f"(+{r['prompt_gain_pp']:.2f} pp)")
    if elic:
        mean_rec = sum(r["recovered_fraction"] for r in elic.values()) / len(elic)
        votes["elicitation"] = "suppressed" if mean_rec > 0.05 else "erased"
        print(f"  -> mean recovered fraction {mean_rec:.1%}: {votes['elicitation']}")

    print("\n=== 2. adapter scaling: is the collapse a cheap direction in weight space? ===")
    for d, r in sorted(alpha.items()):
        print(f"  {d:<12s} half the reverse loss by alpha={r['alpha_half_reverse_loss']:g}, "
              f"forward only {r['forward_progress_there']:.0%} of the way there")
    if alpha:
        cheap = sum(1 for r in alpha.values()
                    if r["alpha_half_reverse_loss"] <= 0.5 and r["forward_progress_there"] < 0.8)
        votes["alpha_scale"] = "suppressed" if cheap > len(alpha) / 2 else "entangled"
        print(f"  -> {cheap}/{len(alpha)} domains show reverse collapsing ahead of forward: "
              f"{votes['alpha_scale']}")

    print("\n=== 3. relearning cost against the never-had control ===")
    for d, r in sorted(relearn.items()):
        curve = "  ".join(f"k={k[7:]}:{r[k]:.3f}" for k in ("relearn10", "relearn50", "relearn200", "relearn1000")
                          if r.get(k) is not None)
        print(f"  {d:<12s} {curve}")
        if r.get("outruns_never_had_at_k"):
            print(f"               outruns the never-had control at k = {r['outruns_never_had_at_k']}")
    if relearn:
        outrun = sum(1 for r in relearn.values() if r.get("outruns_never_had_at_k"))
        votes["relearn"] = "suppressed" if outrun > len(relearn) / 2 else "erased"
        print(f"  -> {outrun}/{len(relearn)} domains relearn faster than a capability never had: "
              f"{votes['relearn']}")

    print("\n=== 7. spectral repair: does removing a component of the update restore the reverse? ===")
    for d, r in sorted(spectral.items()):
        if r["best_repaired_reverse"] is None or r["sft_reverse"] is None:
            continue
        gain = r["best_repaired_reverse"] - r["sft_reverse"]
        fwd_cost = ((r["forward_at_best"] - r["sft_forward"])
                    if (r["forward_at_best"] is not None and r["sft_forward"] is not None) else float("nan"))
        print(f"  {d:<12s} sft_rev={r['sft_reverse']:.4f} -> {r['best_repaired_reverse']:.4f} "
              f"({gain:+.4f} via {r['best_variant']}), forward {fwd_cost:+.4f}"
              + ("   [LoRA delta: weak instrument]" if r["lowrank_caveat"] else ""))
    if spectral:
        restored = sum(1 for r in spectral.values()
                       if r["best_repaired_reverse"] is not None and r["sft_reverse"] is not None
                       and r["best_repaired_reverse"] - r["sft_reverse"] > 0.02)
        votes["spectral"] = "suppressed" if restored > len(spectral) / 2 else "erased"
        print(f"  -> {restored}/{len(spectral)} domains recover reverse capability with NO retraining: "
              f"{votes['spectral']}")

    print("\n=== verdict ===")
    if not votes:
        print("  no mechanism results on disk yet")
    else:
        for k, v in votes.items():
            print(f"  {k:<14s} {v}")
        agree = len(set(votes.values())) == 1
        print(f"  {'unanimous' if agree else 'SPLIT — report the disagreement, do not average it'}: "
              f"{sorted(set(votes.values()))}")

    out = a.out or (RESULTS_DIR / "mech" / "verdict.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"elicitation": elic, "alpha_scale": alpha, "relearn": relearn,
                               "spectral": spectral, "votes": votes}, indent=2))
    print(f"\n[mech] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
