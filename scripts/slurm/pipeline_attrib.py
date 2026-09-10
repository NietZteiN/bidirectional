#!/usr/bin/env python3
"""Submit the attribution arms (RQ6): three objectives, each against a direction-matched SFT.

    python scripts/slurm/pipeline_attrib.py --dry-run

The claim being tested is methodological, so the submission shape carries it: every objective
job is paired with the `mix` arm that supplies the SAME directional content and none of the
method. If the matched baseline reaches the same place, the objective is not what did the work.

  cft           code   forward + equivalence judgements (the workshop paper's worked example)
  unlikelihood  mt     Zan et al. 2024's instruction-conflicting unlikelihood term
  roundtrip     sql    reverse supervision built into the objective

Each pair trains under one recipe and is evaluated in ONE pass, because a cross-pass difference
finer than 0.5 pp is engine noise and these are exactly the comparisons where that matters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

#: (arm, domain, the arm it is matched against). The baseline is not a nicety here — it is the
#: experiment.
CASES = [
    ("cft", "code", "mix5"),
    ("unlikelihood", "mt_en-de", "mix5"),
    ("roundtrip", "sql", "mix5"),
]


def sub(name, argv, *, partition="h200", time="08:00:00", dep=None, mem="64G", dry=False):
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem]
    if dep:
        cmd += ["--dependency", f"afterok:{dep}"]
    if dry:
        cmd.append("--dry-run")
    cmd += ["--argv"] + argv
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if dry:
        print(f"[dry] {name:<40s} {time}  {' '.join(argv[:6])} ...")
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
    ap.add_argument("--models", default="llama32-3b,llama31-8b")
    ap.add_argument("--seeds", default="17,42")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for model in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            for arm, domain, matched in CASES:
                jid = sub(f"at_{arm}_{domain}_{model}_s{seed}",
                          ["scripts/20_train_pack.py", "--domain", domain, "--model", model,
                           "--seed", str(seed), "--arms", f"{arm},{matched}"],
                          time="10:00:00", dry=a.dry_run)
                sub(f"at_ev_{arm}_{domain}_{model}_s{seed}",
                    ["-m", "bidir.evaluate", "--domain", domain, "--model", model,
                     "--seed", str(seed), "--arms", f"base,sft,{matched},{arm}",
                     "--tag", f"attrib_{arm}"],
                    time="03:00:00", dep=jid, dry=a.dry_run)

    print("\nEvery objective is paired with the mix arm supplying the same directional content.")
    print("Read them with: python scripts/50_contrasts.py --run <result dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
