"""Code coverage: predict which lines run, or find an input that reaches a line.

The paired task from DexBench (Hasanov et al., ACL 2026, `papers/hasanov2026pathnottaken.pdf`):

  forward   program + input  ->  the set of executed line numbers
  reverse   program + target line -> an input whose execution reaches that line

Both directions are verified by RUNNING the program (`bidir.coverage_runner`), so neither needs
a threshold, and the reverse direction is genuinely set-valued: any input that reaches the target
line is correct, not only the one the instance was built from.

WHY THIS DOMAIN EXISTS. Every other cell here is our own construction, so a reader may reasonably
ask whether the phenomenon is a property of our corpora. This one's evaluation set is somebody
else's published benchmark, on a task they defined, which makes the result checkable against
their own table of 13 models.

GROUND TRUTH IS MEASURED, NOT INHERITED. DexBench ships candidate coverage sets ("FOCCs") from
CFG path enumeration, and its pipeline validates them against real coverage in a later stage.
Those candidates are not reliable on their own: for `CRUXEval/97`, whose `lst.clear()` empties
the list before a `for ... else`, the true coverage is [1,3,4,5,9,12] and none of the three
generated FOCCs contains it -- they miss the `else` clause entirely. So every instance here is
labelled by execution, and the FOCC lists are not used.

CONTAMINATION. The eval set is DexBench's 298 CRUXEval-derived programs. Training instances are
drawn from the CRUXEval programs it did NOT select, so the split is disjoint by program, and
`build_report.json` records the check.
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from bidir.config import THIRD_PARTY
from bidir.coverage_runner import run_many
from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "coverage"
CELL = "coverage"
CELL_OPTS: dict[str, Any] = {}

DEXBENCH = THIRD_PARTY / "dexbench"
#: A line worth asking the reverse question about: a branch, a loop header, or a return.
#: re.M matters: without it `^` anchors to the start of the whole program, and since CRUXEval
#: programs all begin with `def f(...)` the whole-program filter rejected every candidate and the
#: training pool came out empty. The per-line use below is unaffected either way.
_BRANCH = re.compile(r"^\s*(if|elif|else|for|while|return)\b", re.M)


def _split_harness(source: str) -> Optional[tuple[str, str]]:
    """(program without its final call, the call's argument text).

    DexBench's formatted programs end in `unittest.TestCase().assertEqual(f(ARGS), EXPECTED)`.
    Splitting the arguments out is what turns a fixed script into a paired instance: the program
    is the context and the arguments are the side that varies.
    """
    lines = source.rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.search(r"assertEqual\(\s*(\w+)\((.*)\)\s*,\s*(.*)\)\s*$", lines[i])
        if m:
            args = m.group(2)
            try:
                ast.parse(f"f({args})")
            except SyntaxError:
                return None
            return "\n".join(lines[:i]).rstrip(), args
    return None


def render_program(body: str, args: str, entry: str = "f") -> str:
    """The runnable file for a given argument text: the program plus a bare call."""
    return f"{body}\n\n{entry}({args})\n"


def numbered(source: str) -> str:
    return "\n".join(f"Line {i}: {ln}" for i, ln in enumerate(source.splitlines(), 1))


# --------------------------------------------------------------------------- build

def _dexbench_programs() -> list[tuple[str, str]]:
    d = DEXBENCH / "data" / "CRUXEval" / "formatted_cruxeval_programs"
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} missing -- clone https://github.com/sail-ucf/dexbench into third_party/")
    return [(p.stem, p.read_text()) for p in sorted(d.glob("sample_*.py"))]


def _cruxeval_rest(exclude: set[str], cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    """CRUXEval programs DexBench did not select, formatted the same way."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "cruxeval-org/cruxeval"), split="test")
    out = []
    for r in ds:
        if r["id"] in exclude:
            continue
        code, inp, outp = r["code"].strip(), r["input"].strip(), r["output"].strip()
        if not _BRANCH.search(code):        # DexBench's own filter: needs control flow to be a task
            continue
        src = (f"import unittest\n\n{code}\n\n\n"
               f"unittest.TestCase().assertEqual(f({inp}), {outp})\n")
        out.append((r["id"], src))
    return out


def _instances(programs: Sequence[tuple[str, str]], split: str, rng: random.Random,
               cfg: Mapping[str, Any], limit: int) -> list[PairInstance]:
    prepared = []
    for pid, source in programs:
        sp = _split_harness(source)
        if sp is None:
            continue
        body, args = sp
        prepared.append((pid, body, args))

    # One execution per program to label it. This is the whole cost of building the domain.
    verdicts = run_many([render_program(b, a) for _, b, a in prepared],
                        timeout_s=float(cfg.get("exec_timeout_s", 5.0)),
                        workers=int(cfg.get("exec_workers", 16)))

    rows: list[PairInstance] = []
    for (pid, body, args), v in zip(prepared, verdicts):
        if v["status"] not in ("ok", "raised") or len(v["lines"]) < 3:
            continue
        src_lines = body.splitlines()
        # The reverse question needs a target that is actually reachable and actually a branch.
        targets = [n for n in v["lines"]
                   if n <= len(src_lines) and _BRANCH.search(src_lines[n - 1] or "")]
        if not targets:
            continue
        # Prefer the DEEPEST branch: a line at the top level of the function is reached by almost
        # any argument, which would hand the reverse direction a large free floor. Measured before
        # this change, a constant `None` answer scored 0.17 on reverse purely by reaching shallow
        # targets. Indentation is the cheap proxy for conditional depth; later lines break ties.
        targets.sort(key=lambda n: (len(src_lines[n - 1]) - len(src_lines[n - 1].lstrip()), n))
        target_line = targets[-1]
        rows.append(PairInstance(
            pair_id=f"cov::{pid}", domain=NAME, subtask="cruxeval", split=split,
            side_a=args, side_b=json.dumps(v["lines"]),
            meta={"program": body, "numbered": numbered(body), "entry_point": "f",
                  "target_line": target_line, "exec_status": v["status"],
                  "source": "dexbench" if split == "test" else "cruxeval_remainder",
                  "determinable": True}))
        if len(rows) >= limit:
            break
    return rows


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    rng = random.Random(int(cfg.get("seed", 17)))
    dex = _dexbench_programs()
    dex_ids = {pid for pid, _ in dex}
    test = _instances(dex, "test", rng, cfg, int(cfg["n_test"]))

    rest = _cruxeval_rest(dex_ids, cfg)
    rng.shuffle(rest)
    need = int(cfg["n_train"]) + int(cfg["n_val"])
    pool = _instances(rest, "train", rng, cfg, need)
    n_train = min(int(cfg["n_train"]), max(0, len(pool) - int(cfg["n_val"])))
    train, val = pool[:n_train], pool[n_train:n_train + int(cfg["n_val"])]
    for r in val:
        r.split = "val"
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    meta = inst.get("meta") or {}
    prog = meta.get("numbered") or numbered(meta.get("program", ""))
    if direction == "forward":
        return ("Analyze this Python program to determine its code coverage when the function is "
                "called with the argument shown. List every executable line number that runs at "
                f"least once.\n\nProgram:\n{prog}\n\nCall:\n{meta.get('entry_point', 'f')}"
                f"({inst['side_a']})\n\nAnswer with a JSON list of integers, sorted ascending.")
    return ("Find an argument for this Python function that makes execution reach line "
            f"{meta.get('target_line')}.\n\nProgram:\n{prog}\n\nAnswer with only the argument "
            "text, in valid Python literal syntax, as it would appear inside the call "
            f"{meta.get('entry_point', 'f')}(...).")


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return ("Output only the JSON list, for example [1, 3, 4]. Count a line as executed if it "
                "runs at least once, including the def line.")
    return ("Output only the argument text. Any argument that reaches the line is correct -- it "
            "does not have to be the one the program was written with.")


# --------------------------------------------------------------------------- scoring

_INTS = re.compile(r"-?\d+")


def parse_lines(text: str) -> Optional[list[int]]:
    t = (text or "").strip()
    m = re.search(r"\[[^\]]*\]", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list) and all(isinstance(x, int) for x in v):
                return sorted(set(v))
        except Exception:
            pass
    nums = _INTS.findall(t)
    return sorted({int(n) for n in nums}) if nums else None


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]
    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]

    if direction == "forward":
        for row, out, inst in zip(rows, outputs, insts):
            got = parse_lines(out or "")
            want = sorted(json.loads(inst["side_b"]))
            row["parse_ok"] = int(got is not None)
            row["off_target"] = int(got is None)
            if got is None:
                row["strict"] = 0
                row["line_f1"] = 0.0
            else:
                inter = len(set(got) & set(want))
                row["line_f1"] = (2 * inter / (len(got) + len(want))) if (got or want) else 0.0
                row["strict"] = int(got == want)
            row["criterion"] = "exact executed-line set, measured by execution"
        return rows

    # Reverse: SET-VALUED. Any argument that reaches the target line is correct; scoring against
    # the stored one would punish the behaviour the task asks for.
    cleaned = [re.sub(r"^```\w*\n?|```$", "", (o or "").strip(), flags=re.M).strip() for o in outputs]
    progs, slots = [], []
    for i, (out, inst) in enumerate(zip(cleaned, insts)):
        if not out:
            continue
        meta = inst.get("meta") or {}
        progs.append(render_program(meta["program"], out, meta.get("entry_point", "f")))
        slots.append(i)
    verdicts = run_many(progs, timeout_s=float(cfg.get("exec_timeout_s", 5.0)),
                        workers=int(cfg.get("exec_workers", 16))) if progs else []

    for row in rows:
        row["parse_ok"] = 0
        row["strict"] = 0
        row["reached_target"] = 0
        row["criterion"] = "predicted argument reaches the target line (any input that does)"
    for i, v in zip(slots, verdicts):
        row, inst = rows[i], insts[i]
        target = (inst.get("meta") or {}).get("target_line")
        row["exec_status"] = v["status"]
        row["parse_ok"] = int(v["status"] in ("ok", "raised"))
        row["reached_target"] = int(target in set(v["lines"]))
        row["is_stored_input"] = int(normalize(cleaned[i]) == normalize(inst["side_a"]))
        row["strict"] = int(row["reached_target"] and not row["echo"])
    for row in rows:
        row["off_target"] = int(not row["parse_ok"])
    return rows
