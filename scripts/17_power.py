#!/usr/bin/env python
"""Statistical power for the pre-registered contrasts, before spending the compute.

    python scripts/17_power.py
    python scripts/17_power.py --n 500 --clusters 500

Every primary contrast is a paired difference of two proportions over the same instances, with
a cluster bootstrap by pair. The question this answers is the one that decides whether a rung of
the dose ladder is worth running: **given n evaluation instances, what is the smallest true
difference we could reliably detect?**

If `mix5 - sft` needs 200 instances and `mix25 - mix50` needs 3,000, then part of the ladder is
decoration at our eval size, and the honest response is to widen the eval set for those rungs or
to drop them --- a decision worth making before 391 GPU-hours, not after.

Method. Simulate paired binary outcomes with a given base rate and true difference, under a
correlation between arms induced by shared instance difficulty (the pairing is what buys the
power, so ignoring it would understate what the design can do). Bootstrap the same way
`50_contrasts.py` does, and report the share of simulations whose 95 % interval excludes zero.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR  # noqa: E402

#: (label, rate under the first arm, rate under the second, kind). Rates are the PREDICTIONS in
#: PREREGISTRATION.md §4, not measurements -- this is a design question asked before any data
#: exists, and the file says so.
#:
#: `kind` matters and separating it is the point of this script. For a contrast we expect to be
#: NON-ZERO, the question is power: would we detect it? For one we expect to be ZERO -- "replacing
#: is as good as doubling", "the dose is free on general ability" -- power is the wrong statistic
#: entirely, because failing to reject zero is the predicted outcome. What matters there is
#: whether the interval is TIGHT ENOUGH to support the equivalence claim, and an interval of
#: +/- 4 pp cannot support "as good as".
SCENARIOS = [
    ("sft - base        (the collapse)",            0.003, 0.250, "detect"),
    ("mix5 - sft        (the cure)",                0.230, 0.003, "detect"),
    ("mix50 - replay    (what direction buys)",     0.305, 0.010, "detect"),
    ("mix50 - mix25     (ladder saturates)",        0.305, 0.295, "equivalence"),
    ("mix10 - mix5      (knee resolution)",         0.281, 0.261, "detect"),
    ("replay - sft      (ordinary forgetting)",     0.010, 0.003, "equivalence"),
    ("mix50 - flip      (replace vs double)",       0.305, 0.312, "equivalence"),
    ("general probe     (dose is free vs sft)",     0.700, 0.700, "equivalence"),
]

#: The equivalence margin a claim of "no difference" has to fit inside to be worth making.
EQUIV_MARGIN_PP = 2.0


def simulate(p_a: float, p_b: float, n: int, rho: float, n_boot: int, n_sim: int,
             seed: int = 17) -> dict:
    """Vectorised: the pure-Python version was O(n x n_sim x n_boot) and could not reach the
    eval sizes this script exists to compare."""
    import numpy as np

    rng = np.random.default_rng(seed)
    # Shared per-instance difficulty induces the correlation the pairing exploits; ignoring it
    # would understate what a paired design can do.
    u = rng.random((n_sim, n))
    ea = rng.random((n_sim, n))
    eb = rng.random((n_sim, n))
    a = (rho * u + (1 - rho) * ea < p_a).astype(np.int8)
    b = (rho * u + (1 - rho) * eb < p_b).astype(np.int8)

    # Chunked over simulations: the full index array is n_sim x n_boot x n, which reaches
    # gigabytes at the eval sizes this script is meant to compare.
    los, his = [], []
    chunk = max(1, min(n_sim, int(4e7 // max(1, n_boot * n))))
    for start in range(0, n_sim, chunk):
        end = min(n_sim, start + chunk)
        idx = rng.integers(0, n, size=(end - start, n_boot, n))
        # Resample instances, take BOTH arms for each drawn instance -- that is the pairing.
        ga = np.take_along_axis(a[start:end, None, :], idx, axis=2).mean(axis=2)
        gb = np.take_along_axis(b[start:end, None, :], idx, axis=2).mean(axis=2)
        d = (ga - gb) * 100.0
        los.append(np.percentile(d, 2.5, axis=1))
        his.append(np.percentile(d, 97.5, axis=1))
    lo = np.concatenate(los)
    hi = np.concatenate(his)
    detected = np.mean((lo > 0) | (hi < 0))
    return {"power": float(detected), "mean_ci_width_pp": float(np.mean(hi - lo))}


def mde(p_base: float, n: int, rho: float, n_boot: int, n_sim: int) -> float:
    """Smallest difference detected with power >= 0.8, by bisection on the true effect."""
    lo, hi = 0.0, min(0.5, 1.0 - p_base)
    for _ in range(9):
        mid = (lo + hi) / 2
        r = simulate(min(0.999, p_base + mid), p_base, n, rho, n_boot, n_sim)
        if r["power"] >= 0.8:
            hi = mid
        else:
            lo = mid
    return hi * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="evaluation instances per cell")
    ap.add_argument("--rho", type=float, default=0.5, help="between-arm correlation from shared difficulty")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-sim", type=int, default=200)
    a = ap.parse_args()

    print(f"n = {a.n} instances, rho = {a.rho}, {a.n_sim} simulations x {a.n_boot} bootstrap draws\n")
    print(f"{'contrast':<44}{'kind':>12}{'true':>7}{'power':>8}{'+/- pp':>8}")
    rows = []
    for label, pa, pb, kind in SCENARIOS:
        r = simulate(pa, pb, a.n, a.rho, a.n_boot, a.n_sim)
        half = r["mean_ci_width_pp"] / 2
        rows.append({"contrast": label, "kind": kind, "p_a": pa, "p_b": pb,
                     "true_diff_pp": (pa - pb) * 100, "ci_half_width_pp": half, **r})
        if kind == "detect":
            ok = r["power"] >= 0.8
            flag = "" if ok else "   <-- UNDERPOWERED"
        else:
            # An equivalence claim needs the whole interval inside the margin.
            ok = half <= EQUIV_MARGIN_PP
            flag = "" if ok else f"   <-- CANNOT CLAIM EQUIVALENCE (need +/-{EQUIV_MARGIN_PP})"
        rows[-1]["adequate"] = ok
        print(f"{label:<44}{kind:>12}{(pa - pb) * 100:>+7.1f}{r['power']:>8.2f}{half:>8.1f}{flag}")

    print(f"\nminimum detectable effect (power 0.8) at n = {a.n}:")
    mdes = {}
    for base in (0.05, 0.25, 0.50, 0.75):
        m = mde(base, a.n, a.rho, a.n_boot, a.n_sim)
        mdes[base] = m
        print(f"  from a base rate of {base:.0%}: {m:.1f} pp")

    out = RESULTS_DIR / "power" / f"power_n{a.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": a.n, "rho": a.rho, "scenarios": rows, "mde_pp": mdes,
                               "note": "rates are the PREDICTIONS in PREREGISTRATION.md 4, not "
                                       "measurements; this is a design question asked before "
                                       "any data exists"}, indent=2))
    print(f"\n[power] wrote {out}")
    bad = [r for r in rows if not r["adequate"]]
    if bad:
        print(f"\nINADEQUATE at n = {a.n}:")
        for r in bad:
            why = ("power {:.2f} < 0.80".format(r["power"]) if r["kind"] == "detect"
                   else "interval +/-{:.1f} pp exceeds the +/-{:.1f} margin".format(
                       r["ci_half_width_pp"], EQUIV_MARGIN_PP))
            print(f"  ! {r['contrast']}  --  {why}")
    else:
        print(f"\nEvery pre-registered contrast is adequate at n = {a.n}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
