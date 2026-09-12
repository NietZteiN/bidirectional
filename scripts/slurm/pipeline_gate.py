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

PLACEMENT. `h200` and `normal` share ONE juno QoS budget of 4 running jobs across this whole
account, of which obtune holds 2 and this project claims $BIDIR_JUNO_SHARE (default 1).
`h100` and `a30` carry no QoS at all, so that is where the gate lives by default -- and it is
also why the old placement was wrong: three cells on h200 would all submit before any reached
RUNNING, so submit.py's share guard (which counts running jobs) could not catch them, and the
gate would quietly run 3-deep in a pool where it is entitled to 1.

Only `sql` is on h200, because its reverse criterion keeps a frozen granite-3.1-8b resident
beside the model under test: 0.45 + 0.45 of the card, which is 63 GB each on an H200 and does
not fit in an a30's 24 GB at all. Contested jobs beyond the share are CHAINED with `afterany`
rather than dropped, so the gate stays complete and only its wall-clock grows.

`g-06-01` is excluded from h100 because it advertises 3g.47gb MIG slices, and obtune measured
a 2.8x slowdown there.

PASS RULE, pre-registered (plan §12, and this file is committed before the jobs are read):
in at least one NLP cell, `sft - base` on strict reverse <= -50 % relative, AND IFEval
`sft - base` better than -5 points, AND `rev` strict reverse well above zero. The last clause
is the kill-gate: if `rev` is ~0 the reverse direction is not learnable on this corpus at this
scale and no other arm's null means anything.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

#: (cell, partition, extra sbatch args). h100's g-06-01 is MIG-sliced; exclude it.
#: `sql` is the one h200 cell -- see PLACEMENT above; everything else avoids the juno pool.
NO_MIG = ["--exclude", "g-06-01"]
CELLS = [
    ("mt_en-de", "h100", NO_MIG),
    ("mt_de-en", "h100", NO_MIG),
    ("sql",      "h200", []),      # two resident models: needs the 141 GB card
    ("code",     "h100", NO_MIG),
]

#: Partitions drawing on the shared juno QoS pool. Kept in step with scripts/slurm/submit.py.
CONTESTED = {"h200", "normal"}
ARMS = "gate"          # sft, mix5, mix50, rev — see bidir.arms.GATE


def sub(name, argv, *, partition, time, dep=None, mem="64G", extra=None, dry=False,
        dep_kind="afterok") -> str | None:
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem]
    if dep:
        cmd += ["--dependency", f"{dep_kind}:{dep}"]
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

    share = int(os.environ.get("BIDIR_JUNO_SHARE", "1"))
    contested_running: list[str] = []      # job ids already placed in the juno pool

    train_ids: list[str] = []
    skipped: list[str] = []
    for cell, partition, extra in CELLS:
        # Beyond the share, a contested job waits on an earlier one instead of being dropped.
        # `afterany`, not `afterok`: this link exists to cap concurrency, so a failed predecessor
        # must not also cancel the cells behind it. Only the train->eval link is afterok.
        chain = None
        if partition in CONTESTED and len(contested_running) >= share:
            chain = contested_running[-(share)]

        jid = sub(f"tr_{cell}_{a.model}", ["scripts/20_train_pack.py", "--domain", cell,
                                           "--model", a.model, "--seed", str(a.seed), "--arms", ARMS],
                  partition=partition, time=a.train_time, extra=extra, dry=a.dry_run,
                  dep=chain, dep_kind="afterany")
        if partition in CONTESTED and jid:
            contested_running.append(jid)

        # AN EVAL WITHOUT ITS DEPENDENCY IS WORSE THAN NO EVAL. `sub` returns None when the
        # submission was refused -- a full queue, or the juno share guard -- and passing that
        # None through as `dep` produced an eval job with NO dependency at all, which starts
        # immediately and scores adapters that do not exist yet. The adapter-effectiveness
        # assertion would catch it, but only after a wasted GPU allocation and with a failure
        # that reads like a broken adapter rather than a broken submission.
        if not jid and not a.dry_run:
            skipped.append(cell)
            print(f"  !! {cell}: training was not submitted, so its eval is skipped too",
                  file=sys.stderr)
            continue
        train_ids.append(jid) if jid else None

        sub(f"ev_{cell}_{a.model}", ["-m", "bidir.evaluate", "--domain", cell,
                                     "--model", a.model, "--seed", str(a.seed),
                                     "--arms", "base,sft,mix5,mix50,rev", "--tag", "gate"],
            partition=partition, time=a.eval_time, dep=jid, extra=extra, dry=a.dry_run)

    # One probe job for the whole gate: IFEval + GSM8K separate directional loss from a
    # general loss of instruction following (plan RQ1's control).
    sub(f"probe_gate_{a.model}", ["scripts/40_probes.py", "--model", a.model, "--seed", str(a.seed),
                                  "--cells", ",".join(c for c, _, _ in CELLS),
                                  "--arms", "base,sft,mix5", "--tasks", "ifeval,gsm8k"],
        partition="h100", time="03:00:00", extra=NO_MIG,
        dep=":".join(train_ids) if train_ids and not a.dry_run else None, dry=a.dry_run)

    if skipped:
        # The pass rule needs `code` (the known-positive control) and at least one other NLP
        # cell. Saying so here beats discovering it from a half-empty contrast table.
        print(f"\nINCOMPLETE: {', '.join(skipped)} were not submitted. Re-run this script once "
              f"the queue frees; packs skip arms whose final/ already exists, so nothing is "
              f"recomputed.", file=sys.stderr)
        if "code" in skipped:
            print("  `code` is the known-positive control -- without it no other cell's null "
                  "is readable (plan §12).", file=sys.stderr)

    print("\nGate submitted. Read it with:  python scripts/50_contrasts.py --gate")
    return 2 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
