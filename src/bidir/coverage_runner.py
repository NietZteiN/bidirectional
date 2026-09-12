"""Measure which lines a Python program executes, in a subprocess.

Both directions of the `coverage` domain are verified by running code, so this is the domain's
entire criterion and it has to be exact. `sys.settrace` rather than the `coverage` package: the
programs are single files run once, the trace callback is a dozen lines, and it avoids a
dependency whose own import lines would show up in the trace.

A subprocess per program, because the programs come from a benchmark and may loop forever, raise,
or call `exit()`. The timeout is the only thing standing between a generated input and a stuck
job.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

_HARNESS = r'''
import json, sys, runpy

TARGET = {target!r}
executed = set()

def tracer(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == TARGET:
        executed.add(frame.f_lineno)
    return tracer

status, err = "ok", None
sys.settrace(tracer)
try:
    runpy.run_path(TARGET, run_name="__main__")
except SystemExit:
    pass
except BaseException as e:
    status, err = "raised", type(e).__name__
finally:
    sys.settrace(None)

json.dump({{"status": status, "error": err, "lines": sorted(executed)}},
          open({out!r}, "w"))
'''


@functools.lru_cache(maxsize=1)
def interpreter_overhead_s() -> float:
    """Wall-clock cost of starting this interpreter and importing what the harness imports.

    MEASURED, BECAUSE IT IS NOT A CONSTANT. `subprocess.run(..., timeout=timeout_s)` bounds
    interpreter startup, imports AND the program together, so a budget chosen for the program
    is really a budget for whatever the filesystem is doing that minute. On 2026-09-12 the
    same 40 known-correct `coverage` reverse answers scored 1.000 on the login node, where the
    interpreter is warm in page cache, and 0.000 on a compute node -- status "timeout" on all
    40, at every worker count INCLUDING ONE, so not contention. This tree lives on MooseFS at
    ~1,289 MB/s and a cold `python -c pass` there costs seconds.

    Measuring the overhead once and adding it to the budget makes `timeout_s` mean what its
    name says -- time allowed for the program -- and makes it transfer between filesystems.
    """
    t = time.monotonic()
    subprocess.run([sys.executable, "-c", "import json, sys, pathlib"], capture_output=True)
    return time.monotonic() - t


def run_coverage(source: str, timeout_s: float = 5.0,
                 tmpdir: Optional[str] = None, _retry: bool = True) -> dict[str, Any]:
    """Execute `source` and return {status, error, lines}.

    `status` is "ok" if it ran to completion, "raised" if it threw (the lines executed before the
    exception are still returned, which is what a coverage question asks), "timeout", or
    "harness_error".

    `timeout_s` bounds THE PROGRAM. Interpreter startup is measured separately and added, with
    3x headroom, because that cost depends on the filesystem rather than on the program.

    A timeout is retried once, serially and with a doubled budget, because a timeout is a
    measurement failure rather than a wrong answer and the caller cannot tell the difference
    once it is folded into a score.
    """
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        prog = Path(td) / "prog.py"
        out = Path(td) / "cov.json"
        prog.write_text(source)
        harness = Path(td) / "harness.py"
        harness.write_text(_HARNESS.format(target=str(prog), out=str(out)))
        budget = timeout_s + 3.0 * interpreter_overhead_s() + 1.0
        try:
            subprocess.run([sys.executable, str(harness)], timeout=budget,
                           capture_output=True, cwd=td)
        except subprocess.TimeoutExpired:
            if _retry:
                # Serial retry with a doubled program budget. If this program really does not
                # terminate, the retry costs one timeout; if the first attempt lost a race with
                # the filesystem or a neighbour, the retry recovers a real measurement.
                return run_coverage(source, timeout_s * 2, tmpdir, _retry=False)
            return {"status": "timeout", "error": None, "lines": [],
                    "budget_s": budget, "overhead_s": interpreter_overhead_s()}
        if not out.exists():
            return {"status": "harness_error", "error": None, "lines": []}
        try:
            return json.loads(out.read_text())
        except Exception:
            return {"status": "harness_error", "error": None, "lines": []}


def run_many(sources: list[str], timeout_s: float = 5.0, workers: int = 16,
             tmpdir: Optional[str] = None) -> list[dict[str, Any]]:
    """The scorer runs one of these per trial, so it is worth parallelising."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda s: run_coverage(s, timeout_s, tmpdir), sources))
