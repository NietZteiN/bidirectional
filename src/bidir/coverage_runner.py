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

import json
import subprocess
import sys
import tempfile
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


def run_coverage(source: str, timeout_s: float = 5.0,
                 tmpdir: Optional[str] = None) -> dict[str, Any]:
    """Execute `source` and return {status, error, lines}.

    `status` is "ok" if it ran to completion, "raised" if it threw (the lines executed before the
    exception are still returned, which is what a coverage question asks), "timeout", or
    "harness_error".
    """
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        prog = Path(td) / "prog.py"
        out = Path(td) / "cov.json"
        prog.write_text(source)
        harness = Path(td) / "harness.py"
        harness.write_text(_HARNESS.format(target=str(prog), out=str(out)))
        try:
            subprocess.run([sys.executable, str(harness)], timeout=timeout_s,
                           capture_output=True, cwd=td)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": None, "lines": []}
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
