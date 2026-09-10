#!/usr/bin/env python3
"""Submit the September decision gate as one dependency-linked DAG.

    python scripts/slurm/pipeline_gate.py --dry-run
    python scripts/slurm/pipeline_gate.py --model llama32-3b

Four cells x {sft, mix5, mix50, rev}, each packed into ONE training job, each followed by an
eval pass that waits on `afterok`. Under 20 GPU-hours total.

The cells are chosen so that a null is interpretable:
  * mt_en-de and mt_de-en  — BOTH training directions of one pair, because Zhu et al. (2024)
    report collapse for X->en but not en->X. If only one direction collapses, that asymmetry
    is the RQ2 moderator result, not a failure of the gate.
  * sql                    — a second, structurally different NLP domain with an execution
    criterion on the forward side.
  * code                   — the known-positive control on the NEW panel. If code does not
    collapse here, the harness changed, not the phenomenon, and nothing else in the gate is
    readable.

Partitions alternate: the per-user job cap is enforced per partition, so spreading across
h200 and h100 gets more slots than either queue allows alone. `g-06-01` is excluded from h100
because it advertises 3g.47gb MIG slices, and obtune measured a 2.8x slowdown there.

PASS RULE, pre-registered (plan §12, and this file is committed before the jobs are read):
in at least one NLP cell, `sft - base` on strict reverse <= -50 % relative, AND IFEval
`sft - base` better than -5 points, AND `rev` strict reverse well above zero. The last clause
is the kill-gate: if `rev` is ~0 the reverse direction is not learnable on this corpus at this
scale and no other arm's null means anything.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

#: (cell, partition, extra sbatch args). h100's g-06-01 is MIG-sliced; exclude it.
CELLS = [
    ("mt_en-de", "h200", []),
    ("mt_de-en", "h200", []),
    ("sql",      "h200", []),
    ("code",     "h100", ["--exclude", "g-06-01"]),
]
ARMS = "gate"          # sft, mix5, mix50, rev — see bidir.arms.GATE


def sub(name, argv, *, partition, time, dep=None, mem="64G", extra=None, dry=False) -> str | None:
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem]
    if dep:
        cmd += ["--dependency", f"afterok:{dep}"]
    cmd += list(extra or [])
    if dry:
        cmd.append("--dry-run")
    cmd += ["--argv"] + argv
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
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
    ap.add_argument("--train-time", default="06:00:00")
    ap.add_argument("--eval-time", default="02:00:00")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    train_ids: list[str] = []
    for cell, partition, extra in CELLS:
        jid = sub(f"tr_{cell}_{a.model}", ["scripts/20_train_pack.py", "--domain", cell,
                                           "--model", a.model, "--seed", str(a.seed), "--arms", ARMS],
                  partition=partition, time=a.train_time, extra=extra, dry=a.dry_run)
        if jid:
            train_ids.append(jid)
        ev = sub(f"ev_{cell}_{a.model}", ["-m", "bidir.evaluate", "--domain", cell,
                                          "--model", a.model, "--seed", str(a.seed),
                                          "--arms", "base,sft,mix5,mix50,rev", "--tag", "gate"],
                 partition=partition, time=a.eval_time, dep=jid, extra=extra, dry=a.dry_run)

    # One probe job for the whole gate: IFEval + GSM8K separate directional loss from a
    # general loss of instruction following (plan RQ1's control).
    sub(f"probe_gate_{a.model}", ["scripts/40_probes.py", "--model", a.model, "--seed", str(a.seed),
                                  "--cells", ",".join(c for c, _, _ in CELLS),
                                  "--arms", "base,sft,mix5", "--tasks", "ifeval,gsm8k"],
        partition="h200", time="03:00:00",
        dep=":".join(train_ids) if train_ids and not a.dry_run else None, dry=a.dry_run)

    print("\nGate submitted. Read it with:  python scripts/50_contrasts.py --gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
