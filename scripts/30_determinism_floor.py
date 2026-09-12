#!/usr/bin/env python
"""Measure the cross-pass determinism floor, once, and write it down.

    python scripts/30_determinism_floor.py --domain mt_en-de --model llama32-3b --passes 3

Greedy decoding is NOT bitwise reproducible across evaluation passes. vLLM batches
continuously, so which arms share a pass changes reduction order and occasionally an argmax.
The workshop paper measured 6-8 % of generations differing and 4-8 of 1,500 graded trials
flipping, and every table in it is built on the rule that follows: one pass per table,
contrasts only within a pass, never quote a cross-pass difference finer than 0.5 pp.

That rule needs a number behind it on THIS panel, so this script runs the pass N times and
reports what moves. It is cheap, it is run once per model, and its output is what the paper's
methods section cites instead of an assertion.

A "PASS" MUST BE A FRESH ENGINE IN A FRESH PROCESS. This script used to loop N times over one
engine inside one process, submitting a byte-identical request set each time. vLLM schedules an
identical batch identically, so that measures within-engine repeatability, reports a floor of
~0.00 pp, and would license quoting cross-pass differences finer than any real floor -- the
opposite of what the number is for. What actually moves between the paper's passes is engine
instantiation and batch composition, so both are reproduced here:

  * each pass is a SUBPROCESS with its own engine (`--pass-index`, the internal mode below);
  * each pass is padded to `--companion-load` copies of the request set, because a real eval
    pass batches ~18 arms over the same instances and it is that continuous-batching order,
    not the prompts, that changes an argmax. Only the first block is scored.

The floor this reports is therefore an upper bound on what a single arm's numbers can move,
which is the direction an honest bound has to err in.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import domains, engine as eng, prompts  # noqa: E402
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config  # noqa: E402
from bidir.mixture import load_pairs  # noqa: E402


def run_one_pass(a) -> dict:
    """ONE pass, in this process, on an engine this process built. Invoked as a subprocess by
    the driver so that every pass pays a genuine engine instantiation."""
    ensure_obtune()
    cfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    e = eng.get_engine(a.model, cfg.get("engine", {}))

    per_pass = {}
    for direction in ("forward", "reverse"):
        msgs = [prompts.build_messages(x, direction, mod, "simple") for x in insts]
        # Pad the batch the way a real multi-arm pass pads it. The copies are discarded; their
        # only job is to make the scheduler interleave this block with other work, which is
        # what moves an argmax between the paper's passes.
        batch = msgs * max(1, a.companion_load)
        raw_all, _ = eng.generate(e, batch, [None] * len(batch), cfg.get("sampling", {}))
        raw = raw_all[: len(msgs)]
        outs = [prompts.extract_answer(r) for r in raw]
        scored = mod.score_batch(direction, outs, insts, cfg)
        per_pass[direction] = {"raw": raw, "strict": [r["strict"] for r in scored]}
    return per_pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--companion-load", type=int, default=4,
                    help="copies of the request set submitted per pass, emulating the batch "
                         "composition of a real multi-arm pass; only the first block is scored")
    ap.add_argument("--pass-index", type=int, default=None,
                    help="INTERNAL: run a single pass and write it to --pass-out, then exit")
    ap.add_argument("--pass-out", default=None, help="INTERNAL: where the single pass is written")
    a = ap.parse_args()

    if a.pass_index is not None:
        Path(a.pass_out).write_text(json.dumps(run_one_pass(a)))
        return 0

    # Each pass is its own process, so each builds its own engine. Passing --passes 1 here
    # would defeat the point of the script, so it is refused.
    if a.passes < 2:
        raise SystemExit("--passes must be >= 2: a floor is a difference between passes")

    runs = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as td:
        for i in range(a.passes):
            outp = Path(td) / f"pass{i}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--domain", a.domain, "--model", a.model, "--limit", str(a.limit),
                   "--seed", str(a.seed), "--companion-load", str(a.companion_load),
                   "--pass-index", str(i), "--pass-out", str(outp)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"pass {i + 1} failed with rc={r.returncode}")
            runs.append(json.loads(outp.read_text()))
            print(f"  pass {i + 1}/{a.passes} done (fresh engine)", flush=True)

    n_instances = len(runs[0]["forward"]["strict"])
    report = {"domain": a.domain, "model": a.model, "passes": a.passes,
              "n_instances": n_instances,
              "pass_separation": "each pass ran in its own process and built its own engine",
              "companion_load": a.companion_load,
              "directions": {}}
    for direction in ("forward", "reverse"):
        base = runs[0][direction]
        gen_diff, flips, rates = [], [], []
        for r in runs:
            rates.append(sum(r[direction]["strict"]) / len(r[direction]["strict"]))
        for r in runs[1:]:
            gen_diff.append(sum(1 for x, y in zip(base["raw"], r[direction]["raw"]) if x != y) / len(base["raw"]))
            flips.append(sum(1 for x, y in zip(base["strict"], r[direction]["strict"]) if x != y))
        report["directions"][direction] = {
            "strict_rate_per_pass": rates,
            "strict_rate_spread_pp": (max(rates) - min(rates)) * 100,
            "generations_differing_frac": gen_diff,
            "graded_trials_flipped": flips,
        }
        print(f"  {direction:<8s} rates={[f'{x:.4f}' for x in rates]} "
              f"spread={report['directions'][direction]['strict_rate_spread_pp']:.2f} pp  "
              f"gens differing={[f'{x:.1%}' for x in gen_diff]}  flips={flips}")

    worst = max(v["strict_rate_spread_pp"] for v in report["directions"].values())
    report["floor_pp"] = worst
    report["rule"] = (f"never quote a cross-pass difference finer than {max(0.5, worst):.1f} pp; "
                      "one pass per table, contrasts only within a pass")
    out = RESULTS_DIR / "determinism" / f"{a.domain}__{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\n  floor = {worst:.2f} pp -> {report['rule']}")
    print(f"[determinism] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
