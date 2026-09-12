#!/usr/bin/env python3
"""Submit bidir jobs to SLURM. A trimmed port of obtune/scripts/slurm/submit.py.

    python scripts/slurm/submit.py --name gate_mt --time 06:00:00 --argv scripts/20_train_pack.py --domain mt_en-de ...
    python scripts/slurm/submit.py --dependency afterok:12345 --name ev_mt --argv -m bidir.evaluate ...

Three things are load-bearing and are why this is not just `sbatch`:
  1. `source scripts/env.sh` puts $BIDIR_ENV/bin on PATH — vLLM's engine core shells out to
     `ninja`, and without it startup fails as a FileNotFoundError with the real cause buried.
  2. `trap` on EXIT, so a walltime kill still records the job's outcome instead of leaving
     nothing behind.
  3. The .sbatch file IS the provenance record and carries the exact argv; two jobs sharing a
     --name would otherwise leave a committed script describing the wrong run.

NOTE ON QOS. obtune's copy defaults to `--qos=high-throughput`, which does not exist on this
cluster (jobs silently fall back). The real per-user cap is the h200 partition's QoS `juno`:
MaxJobsPU=4. `juno-pri` doubles it at priority 200000, which jumps every other user on a
shared cluster — so it is available but never the default.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SLURM_DIR = ROOT / "runs" / "slurm"
SLURM_LOGS = ROOT / "runs" / "logs" / "slurm"

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{qos}{extra}{dependency}#SBATCH --output={log_dir}/%j_{job_name}.out
#SBATCH --error={log_dir}/%j_{job_name}.out
set -uo pipefail

source {root}/scripts/env.sh
cd "$BIDIR_ROOT"

STATUS_DIR="$BIDIR_ROOT/runs/status"
mkdir -p "$STATUS_DIR"
finish() {{
  rc=$?
  python - "$STATUS_DIR/{job_name}.$SLURM_JOB_ID.json" "$rc" <<'PYEOF'
import datetime, json, os, sys
json.dump({{"job_name": {job_name!r}, "job_id": os.environ.get("SLURM_JOB_ID"),
           "node": os.environ.get("SLURMD_NODENAME"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
           "exit_code": int(sys.argv[2]),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}},
          open(sys.argv[1], "w"), indent=2)
PYEOF
  exit $rc
}}
trap finish EXIT

echo "# bidir slurm job $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION)"
echo "# started $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
echo "# argv: {argv_display}"
echo
{command}
"""

DEFAULTS = {"partition": "h200", "gres": "gpu:1", "cpus": 8, "mem": "64G", "time": "08:00:00"}

#: Partitions that draw on the shared juno QoS pool. `h200` and `normal` are ONE budget of four
#: running jobs across the whole account, not two -- so a CPU analysis on `normal` blocks a GPU
#: training job. `dev` is QoS=juno-dev and `h100`/`a30` carry no QoS, so none of them count.
CONTESTED = {"h200", "normal"}


def juno_jobs_held() -> int:
    """Running jobs this account holds in the contested pool.

    Counted rather than trusted, because a share agreed with a neighbouring project on this
    account is only worth what something enforces. Pending jobs are not counted: a job pending
    on Priority or Resources holds nothing, and counting them made obtune's equivalent guard
    refuse everything the moment a queue backed up.
    """
    import getpass

    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-t", "RUNNING", "-o", "%P"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 0          # never block a submission because squeue was slow
    return sum(1 for line in out.split() if line.strip() in CONTESTED)


def queues_behind_pool(dependency: Optional[str]) -> bool:
    """True when this job cannot be concurrent with what already holds the pool.

    A job whose dependency names something already in the contested pool queues BEHIND it, so
    it adds nothing to concurrency and the share check must not refuse it. Without this, a
    correctly-chained pipeline is rejected the moment its first job reaches RUNNING -- which is
    exactly what happened to the sql gate on 2026-09-12, one second after the packed gate
    started.

    Deliberately narrow: the dependency must name a job that is itself in a contested
    partition. Depending on a `dev` job says nothing about the pool, and letting that through
    would be a way to bypass the share by accident.
    """
    if not dependency:
        return False
    ids = {tok for part in dependency.split(",") for tok in part.split(":")[1:] if tok.isdigit()}
    if not ids:
        return False
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", ",".join(sorted(ids)), "-o", "%i %P"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    for line in out.splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[1].strip() in CONTESTED:
            return True
    return False


def build_script(argv, *, job_name, partition, gres, cpus, mem, time, dependency=None,
                 qos=None, nodelist=None, exclude=None) -> str:
    # -u, unconditionally. A job's stdout is a FILE, so python block-buffers it at 4-8 KB and a
    # long job shows nothing until it exits or the buffer fills -- which makes a running job
    # indistinguishable from a hung one, and makes a job killed at the walltime lose whatever
    # progress it had printed. Scripts that pass flush=True everywhere are unaffected; the ones
    # that do not are exactly the ones worth watching.
    command = "python -u " + " ".join(shlex.quote(a) for a in argv)
    extra = ""
    if nodelist:
        extra += f"#SBATCH --nodelist={nodelist}\n"
    if exclude:
        extra += f"#SBATCH --exclude={exclude}\n"
    return TEMPLATE.format(
        job_name=job_name, partition=partition,
        gres_line="" if gres in ("", "none", "0") else f"#SBATCH --gres={gres}\n",
        cpus=cpus, mem=mem, time=time,
        qos=f"#SBATCH --qos={qos}\n" if qos else "",
        extra=extra,
        dependency=f"#SBATCH --dependency={dependency}\n" if dependency else "",
        log_dir=shlex.quote(str(SLURM_LOGS)), root=shlex.quote(str(ROOT)),
        argv_display=" ".join(argv).replace("{", "{{").replace("}", "}}"),
        command=command)


def submit(script: str, name: str, dry_run: bool) -> str | None:
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_LOGS.mkdir(parents=True, exist_ok=True)
    path = SLURM_DIR / f"{name}.sbatch"
    if path.exists() and path.read_text() != script:
        n = 2
        while (SLURM_DIR / f"{name}.{n}.sbatch").exists():
            n += 1
        path = SLURM_DIR / f"{name}.{n}.sbatch"
        print(f"note: {name}.sbatch exists with different content; wrote {path.name}", file=sys.stderr)
    path.write_text(script)
    path.chmod(0o755)
    if dry_run:
        print(f"--- {path} (not submitted) ---\n{script}")
        return None
    out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"sbatch FAILED for {name}: {out.stderr.strip()}", file=sys.stderr)
        return None
    jid = out.stdout.strip().split()[-1]
    print(f"submitted {jid}  {name}")
    return jid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="adhoc")
    ap.add_argument("--partition", default=DEFAULTS["partition"])
    ap.add_argument("--gres", default=DEFAULTS["gres"])
    ap.add_argument("--cpus", type=int, default=DEFAULTS["cpus"])
    ap.add_argument("--mem", default=DEFAULTS["mem"])
    ap.add_argument("--time", default=DEFAULTS["time"])
    ap.add_argument("--qos", default=None, help="default: the partition's own QoS (juno, MaxJobsPU=4)")
    ap.add_argument("--dependency", default=None, help="e.g. afterok:12345")
    ap.add_argument("--nodelist", default=None)
    ap.add_argument("--exclude", default=None, help="e.g. g-06-01, which advertises MIG slices")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--argv", nargs=argparse.REMAINDER, required=True,
                    help="GREEDY — must come LAST; everything after it is passed to the project python")
    a = ap.parse_args()
    if not a.argv:
        print("nothing to run", file=sys.stderr)
        return 1

    share = int(os.environ.get("BIDIR_JUNO_SHARE", "1"))
    if share and a.partition in CONTESTED and not a.dry_run and not queues_behind_pool(a.dependency):
        held = juno_jobs_held()
        if held >= share:
            print(f"REFUSED: this project may hold {share} running job(s) in the juno pool "
                  f"(h200 + normal are one budget) and already holds {held}.\n"
                  f"  Wait, or submit to a30/h100/dev, which carry no QoS.\n"
                  f"  Override with BIDIR_JUNO_SHARE=<n> if the share has been renegotiated.",
                  file=sys.stderr)
            return 2
    dep = a.dependency
    if dep and not dep.startswith(("afterok:", "afterany:", "after:")):
        dep = f"afterok:{dep}"
    script = build_script(a.argv, job_name=a.name[:60], partition=a.partition, gres=a.gres,
                          cpus=a.cpus, mem=a.mem, time=a.time, dependency=dep, qos=a.qos,
                          nodelist=a.nodelist, exclude=a.exclude)
    return 0 if (submit(script, a.name, a.dry_run) or a.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
