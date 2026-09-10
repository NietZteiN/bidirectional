#!/usr/bin/env python3
"""Submit the mechanism experiments (RQ5) as dependency-linked jobs.

    python scripts/slurm/pipeline_mech.py --floor --dry-run     # experiments 1-3 only
    python scripts/slurm/pipeline_mech.py --all --model llama32-3b

The floor and the upside are submitted separately on purpose (plan §8). Experiments 1-3 are
behavioral, cheap, and together they settle erased-versus-suppressed; 4-6 are the mechanistic
upside that makes the section distinctive but are not a dependency for submission. Submitting
them as one blob would let a stalled probe job hold up the result that actually decides RQ5.

Experiment 3 is the one with a prerequisite: relearn-k needs `sft` trained on the domain AND
the never-had control (`fmt_novel`) trained through the same ladder, or its curve has no scale
to be read against.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

#: Domains the mechanism section runs on. One per criterion family: execution (code), a
#: continuous metric (mt), a round-trip criterion (sql), and the exact synthetic (fmt) whose
#: never-had twin gives experiment 3 its control.
MECH_DOMAINS = ["code", "mt_en-de", "sql", "fmt", "algebra"]


def sub(name, argv, *, partition="h200", time="03:00:00", dep=None, mem="64G", extra=None, dry=False):
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem]
    if dep:
        cmd += ["--dependency", f"afterok:{dep}"]
    cmd += list(extra or [])
    if dry:
        cmd.append("--dry-run")
    cmd += ["--argv"] + argv
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if dry:
        print(f"[dry] {name:<38s} {partition:<5s} {time}  {' '.join(argv[:5])} ...")
        return None
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith("submitted "):
            return line.split()[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama32-3b")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--floor", action="store_true", help="experiments 1-3 only (the RQ5 floor)")
    ap.add_argument("--upside", action="store_true", help="experiments 4-6 only")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.floor or a.upside or a.all):
        a.floor = True
    domains_ = a.domains.split(",") if a.domains else MECH_DOMAINS

    if a.floor or a.all:
        print("== experiment 1: elicitation ladder ==")
        for d in domains_:
            sub(f"m1_elicit_{d}_{a.model}",
                ["-m", "bidir.mech.elicitation", "--domain", d, "--model", a.model,
                 "--seed", str(a.seed), "--arms", "base,sft,mix5,mix50"],
                time="04:00:00", dry=a.dry_run)

        print("\n== experiment 2: adapter scaling ==")
        for d in domains_:
            sub(f"m2_alpha_{d}_{a.model}",
                ["-m", "bidir.mech.alpha_scale", "--domain", d, "--model", a.model,
                 "--arm", "sft", "--seed", str(a.seed)],
                time="03:00:00", dry=a.dry_run)

        print("\n== experiment 3: relearning cost, against the never-had control ==")
        # The control must be trained through the SAME ladder, or the comparison has no scale.
        for d in domains_ + ["fmt_novel"]:
            jid = sub(f"m3_relearn_tr_{d}_{a.model}",
                      ["scripts/20_train_pack.py", "--domain", d, "--model", a.model,
                       "--seed", str(a.seed), "--arms", "relearn"],
                      time="04:00:00", dry=a.dry_run)
            sub(f"m3_relearn_ev_{d}_{a.model}",
                ["-m", "bidir.evaluate", "--domain", d, "--model", a.model, "--seed", str(a.seed),
                 "--arms", "base,sft,relearn10,relearn50,relearn200,relearn1000", "--tag", "relearn"],
                time="02:00:00", dep=jid, dry=a.dry_run)

    if a.floor or a.all:
        print("\n== experiment 7: spectral repair of the adapter ==")
        for d in domains_:
            # In the FLOOR, not the upside: it needs no training, only a checkpoint, and it is
            # the sharpest suppressed-vs-erased instrument we have (PREREGISTRATION Amendment 7).
            sub(f"m7_spectral_{d}_{a.model}",
                ["-m", "bidir.mech.spectral", "--domain", d, "--model", a.model,
                 "--arm", "sft", "--seed", str(a.seed)],
                time="03:00:00", dry=a.dry_run)

    if a.upside or a.all:
        print("\n== experiment 4: layer ablation ==")
        for d in domains_:
            sub(f"m4_ablate_{d}_{a.model}",
                ["-m", "bidir.mech.layer_ablate", "--domain", d, "--model", a.model,
                 "--arm", "sft", "--seed", str(a.seed), "--groups", "4"],
                time="04:00:00", dry=a.dry_run)

        print("\n== experiment 5: direction probes ==")
        for d in domains_:
            # HF forward pass, not vLLM (vLLM does not expose hidden states), so it wants
            # headroom rather than a whole card's KV cache.
            sub(f"m5_probe_{d}_{a.model}",
                ["-m", "bidir.mech.direction_probe", "--domain", d, "--model", a.model,
                 "--seed", str(a.seed), "--arms", "base,sft,mix5"],
                time="03:00:00", mem="96G", dry=a.dry_run)

        print("\n== experiment 6: instruction sensitivity, across the dose ladder ==")
        for d in domains_:
            # The whole ladder, not just sft: arXiv:2509.13079 reports that mixing forward and
            # reverse data weakens the directional distinction, and the mix arms ARE that
            # mixture. If the prescription carries that cost it has to be measured on the
            # prescribed doses (PREREGISTRATION.md Amendment 5).
            sub(f"m6_sens_{d}_{a.model}",
                ["-m", "bidir.mech.sensitivity", "--domain", d, "--model", a.model,
                 "--seed", str(a.seed),
                 "--arms", "base,sft,mix1,mix5,mix10,mix25,mix50,flip"],
                time="04:00:00", dry=a.dry_run)

    print("\nRead them with: python scripts/53_mech_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
