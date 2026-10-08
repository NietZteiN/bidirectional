#!/usr/bin/env python3
"""Unattended watchdog for the feeder: retry what is infrastructure, record what is code.

Called once per feeder pass. It reads this project's ended jobs from sacct (WorkDir == the repo,
so obtune/probing jobs are never touched) and decides, once per job id:

  TIMEOUT                  queue jobs: resubmit with 2x walltime (cap 48 h)
  NODE_FAIL / BOOT_FAIL /  resubmit unchanged
    PREEMPTED
  OUT_OF_MEMORY            resubmit with 2x host memory (cap 256G)
  FAILED                   read the log tail: GPU OOM -> resubmit on h200 only; a transient
                           signature (CUDA/NCCL/network/filesystem) -> resubmit unchanged;
                           anything else is a CODE failure -> FAILURES.md, never retried

Each queue job gets at most MAX_RETRY resubmissions. Runner jobs (tr_/ev_) are resubmitted by
the runner itself; here they are only counted, and a cell whose jobs keep failing is written to
quarantine.txt, which 95_runner.py reads, so an unattended loop stops paying queue waits for
the same traceback. Jobs pending on DependencyNeverSatisfied are cancelled.

Writes STATUS.md every pass: the file to read after being away.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "runs" / "feeder"
STATE, SUBMITTED, QUEUE = F / "watchdog_state.json", F / "submitted.tsv", F / "queue.txt"
QUAR, FAILURES, STATUS, EVENTS = F / "quarantine.txt", F / "FAILURES.md", F / "STATUS.md", F / "events.log"
LOGDIR = ROOT / "runs" / "logs" / "slurm"
SINCE = "2026-09-24T12:00:00"
MAX_RETRY = 2
RUNNER_FAIL_LIMIT = 3       # FAILED/OOM on one runner cell before it is quarantined
RUNNER_TIMEOUT_LIMIT = 4    # a pack resumes arm by arm, so a timeout is usually progress
TRANSIENT = re.compile(
    r"CUDA error|NCCL|illegal memory access|ECC error|cudaErrorLaunchFailure|"
    r"Connection (reset|refused|aborted)|ReadTimeout|Temporary failure in name resolution|"
    r"HTTPError: 5\d\d|Stale file handle|Input/output error|Bus error|DUE TO NODE FAILURE", re.I)
GPU_OOM = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError", re.I)
ENDED_RETRY = {"TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "OUT_OF_MEMORY", "FAILED"}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def event(msg: str) -> None:
    with EVENTS.open("a") as f:
        f.write(f"{now()}  {msg}\n")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"handled": [], "retries": {}, "runner_fail": {}, "runner_timeout": {}, "code_failures": []}


def submitted_map() -> dict[str, tuple[str, str]]:
    """job id -> (job name, the queue line that produced it)."""
    out = {}
    if SUBMITTED.exists():
        for row in SUBMITTED.read_text().splitlines():
            parts = row.split("\t", 2)
            if len(parts) == 3:
                out[parts[0]] = (parts[1], parts[2])
    return out


def log_tail(jid: str, name: str, n: int = 40) -> str:
    p = LOGDIR / f"{jid}_{name}.out"
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return f"(no log at {p})"


PAIR_SEP = ' && [ -n "$jid" ] && '


def segment_for(line: str, name: str) -> str:
    """The single submit command inside `line` that submits `name` (pairs hold two)."""
    if PAIR_SEP not in line:
        return line
    first, second = line.split(PAIR_SEP, 1)
    if f"--name {name} " in second:
        # The eval alone: its training job finished, so drop the dependency on it.
        return second.replace(" --dependency afterok:$jid", "")
    return line                       # the training job failed: redo the whole pair


def hours(t: str) -> float:
    d, _, hms = t.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return (int(d) if d else 0) * 24 + h + m / 60 + s / 3600


def bump(line: str, name: str, *, time_x: float = 1, mem_x: float = 1, partition: str | None = None) -> str:
    seg_start = line.find(f"--name {name} ")
    seg_end = line.find("--argv", seg_start)
    seg = line[seg_start:seg_end]
    if time_x != 1:
        seg = re.sub(r"--time (\S+)", lambda m: "--time %02d:00:00" %
                     min(48, max(1, round(hours(m.group(1)) * time_x))), seg)
    if mem_x != 1:
        seg = re.sub(r"--mem (\d+)G", lambda m: f"--mem {min(256, int(int(m.group(1)) * mem_x))}G", seg)
    if partition:
        seg = re.sub(r"--partition \S+", f"--partition {partition}", seg)
    return line[:seg_start] + seg + line[seg_end:]


def requeue(line: str) -> None:
    rest = QUEUE.read_text() if QUEUE.exists() else ""
    QUEUE.write_text(line + "\n" + rest)       # front of the queue: retries go first


def record_code_failure(st: dict, jid: str, name: str, state: str, why: str) -> None:
    st["code_failures"].append({"id": jid, "name": name, "state": state, "why": why, "at": now()})
    with FAILURES.open("a") as f:
        f.write(f"\n## {name} (job {jid}, {state}) {now()}\n\n{why}\n\n```\n{log_tail(jid, name)}\n```\n")
    event(f"NEEDS A HUMAN  {name} ({jid}) {state}: {why}")


def line_for_name(name: str, subs: dict) -> str | None:
    """A pair's TRAINING job id is captured by `jid=$(...)` and never printed, so it is missing
    from submitted.tsv. Found by name instead, or its failures would pass silently (2026-09-26:
    eight attribution packs failed and none was filed)."""
    for _, (_, line) in reversed(list(subs.items())):
        if f"--name {name} " in line:
            return line
    return None


def handle(st: dict, jid: str, name: str, state: str, subs: dict) -> None:
    tail = log_tail(jid, name, 80) if state == "FAILED" else ""
    qline = subs[jid][1] if jid in subs else line_for_name(name, subs)
    if qline is not None:                             # a feeder queue job: retry it here
        line = segment_for(qline, name)
        n = st["retries"].get(name, 0)
        if n >= MAX_RETRY:
            record_code_failure(st, jid, name, state, f"gave up after {n} retries")
            return
        if state == "TIMEOUT":
            new = bump(line, name, time_x=2)
        elif state in ("NODE_FAIL", "BOOT_FAIL", "PREEMPTED"):
            new = line
        elif state == "OUT_OF_MEMORY":
            new = bump(line, name, mem_x=2)
        elif GPU_OOM.search(tail):
            new = bump(line, name, partition="h200")
        elif TRANSIENT.search(tail):
            new = line
        else:
            record_code_failure(st, jid, name, state, "code failure (no transient signature in the log)")
            return
        st["retries"][name] = n + 1
        requeue(new)
        event(f"retry {n + 1}/{MAX_RETRY}  {name} ({jid}) after {state}")
        return
    if name.startswith(("tr_", "ev_")):               # a runner job: the runner resubmits it
        key = name.split("_", 1)[1]
        if state == "TIMEOUT":
            c = st["runner_timeout"][key] = st["runner_timeout"].get(key, 0) + 1
            limit = RUNNER_TIMEOUT_LIMIT
        else:
            c = st["runner_fail"][key] = st["runner_fail"].get(key, 0) + 1
            limit = RUNNER_FAIL_LIMIT
        event(f"runner job {name} ({jid}) {state}; {c}/{limit} for {key}")
        already = QUAR.exists() and key in QUAR.read_text().split()
        if c >= limit and not already:
            with QUAR.open("a") as f:
                f.write(f"{key}  # {state} x{c}, last job {jid}, {now()}\n")
            record_code_failure(st, jid, name, state, f"runner cell {key} quarantined after {c} x {state}")


def main() -> int:
    st = load_state()
    handled = set(st["handled"])
    subs = submitted_map()
    rows = sh("sacct", "-u", os.environ["USER"], "-X", "-n", "-P", "-S", SINCE,
              "--format=JobID,JobName,State,Elapsed,WorkDir,Partition").splitlines()
    ours, gpu_h = [], 0.0
    for r in rows:
        f = r.split("|")
        if len(f) < 6 or f[4] != str(ROOT):
            continue
        jid, name, state, elapsed = f[0], f[1], f[2].split()[0], f[3]
        ours.append((jid, name, state))
        gpu_h += hours(elapsed) if elapsed else 0
        if jid in handled or state in ("PENDING", "RUNNING", "REQUEUED", "COMPLETED", "CANCELLED"):
            if state in ("COMPLETED", "CANCELLED") and jid not in handled:
                handled.add(jid)
                if state == "COMPLETED":
                    event(f"done  {name} ({jid})")
            continue
        if state in ENDED_RETRY:
            handle(st, jid, name, state, subs)
        handled.add(jid)

    # A job whose dependency failed never runs and holds a queue slot forever.
    for row in sh("squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%r|%Z").splitlines():
        jid, name, reason, wd = (row.split("|") + ["", "", "", ""])[:4]
        if reason == "DependencyNeverSatisfied" and wd == str(ROOT):
            sh("scancel", jid)
            event(f"cancelled {name} ({jid}): its dependency failed")

    release_held()
    resolve_runner_failures(st, ours)
    st["handled"] = sorted(handled)
    from bidir.pipeline_state import atomic_json
    atomic_json(STATE, st)
    write_status(st, ours, gpu_h)
    return 0


def resolve_runner_failures(st: dict, ours: list) -> None:
    """Retain failure history while clearing alarms proven resolved by a later job."""
    quarantined = {line.split()[0] for line in QUAR.read_text().splitlines() if line.strip()} if QUAR.exists() else set()
    for failure in st['code_failures']:
        name = failure['name']
        if failure.get('resolved_by') or not name.startswith(('tr_', 'ev_')):
            continue
        if name.split('_', 1)[1] in quarantined:
            continue
        for jid, completed_name, state in ours:
            if (completed_name != name or state != 'COMPLETED' or not jid.isdigit()
                    or int(jid) <= int(failure['id'])):
                continue
            proof = ROOT / 'runs/status' / f'{name}.{jid}.json'
            try:
                status = json.loads(proof.read_text())
            except (OSError, ValueError):
                continue
            if status.get('exit_code') != 0 or status.get('job_id') != jid:
                continue
            failure['resolved_by'] = dict(job_id=jid, status_file=str(proof), at=now())
            event(f"resolved historical failure {failure['id']} for {name}: completed {jid}")
            with FAILURES.open('a') as log:
                log.write(f"\n**Resolved {now()}:** {name} failure {failure['id']} followed by "
                          f"successful job {jid}; proof `{proof}`. Failure evidence retained.\n")
            break


def release_held() -> None:
    """Held lines wait on a smoke job (held_attrib.gate holds its id): released into the queue
    once it has trained for 15 min or completed, kept back if it failed. Added 2026-09-26 so
    seven attribution pairs do not re-fail on a fix that has not been proven on a GPU."""
    held, gate = F / "held_attrib.txt", F / "held_attrib.gate"
    if not (held.exists() and gate.exists()):
        return
    jid = gate.read_text().strip()
    row = sh("sacct", "-j", jid, "-X", "-n", "-P", "--format=State,Elapsed").strip().split("|")
    if len(row) < 2:
        return
    state, elapsed = row[0].split()[0], row[1]
    if state == "COMPLETED" or (state == "RUNNING" and hours(elapsed) >= 0.25):
        requeue(held.read_text().rstrip("\n"))
        held.rename(F / "held_attrib.released")
        gate.unlink()
        event(f"released held attribution pairs: smoke job {jid} is {state} after {elapsed}")
    elif state in ENDED_RETRY or state == "CANCELLED":
        event(f"held attribution pairs stay held: smoke job {jid} ended {state}")
        gate.unlink()


def write_status(st: dict, ours: list, gpu_h: float) -> None:
    counts: dict[str, int] = {}
    for _, _, s in ours:
        counts[s] = counts.get(s, 0) + 1
    hold = sh("squeue", "-h", "-u", os.environ["USER"], "-n", "hold-node", "-t", "RUNNING", "-o", "%e").strip() or "NOT RUNNING"
    qleft = len(QUEUE.read_text().splitlines()) if QUEUE.exists() else 0
    quar = QUAR.read_text().strip() if QUAR.exists() else "(none)"
    ev = EVENTS.read_text().splitlines()[-25:] if EVENTS.exists() else []
    cf = [c for c in st["code_failures"] if not c.get('resolved_by')]
    STATUS.write_text(
        f"# bidirectional: unattended run status\n\nUpdated {now()} by runs/feeder/watchdog.py.\n\n"
        f"- hold-node (hosts the feeder) ends: **{hold}**\n"
        f"- this project's jobs since {SINCE}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n"
        f"- elapsed job-hours since then: {gpu_h:.1f}\n"
        f"- lines still waiting in queue.txt: {qleft}\n"
        f"- failures needing a human: **{len(cf)}** (details in FAILURES.md)\n\n"
        f"## Quarantined runner cells\n\n```\n{quar}\n```\n\n"
        f"## Needs a human\n\n" + ("\n".join(f"- {c['at']} `{c['name']}` ({c['id']}, {c['state']}): {c['why']}" for c in cf[-20:]) or "(nothing)") +
        "\n\n## Recent events\n\n```\n" + "\n".join(ev) + "\n```\n")


if __name__ == "__main__":
    raise SystemExit(main())
