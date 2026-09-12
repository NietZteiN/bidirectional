"""Execution prediction: output prediction <-> input prediction, on CRUXEval.

The paired directions are `f(input) -> ?` and `f(?) -> output` over the same program. Both are
execution-verified, both are tasks the base model can already do, and the pair is free.

This domain ties the series together. obtune's stacking work observed that breadth SFT forgets
input prediction while output prediction holds — which, if it is the same phenomenon, is
directional collapse measured on a third axis. Confirming it here is what lets the paper say
so instead of gesturing at it.

The reverse direction is SET-VALUED and that is handled explicitly: many inputs can produce
one output, so the criterion is not "matches the stored input" but "the predicted input
actually executes to the required output". A model that finds a different valid preimage is
correct, and scoring it against the stored input would punish exactly the behaviour the task
asks for.
"""
from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from bidir.config import ensure_obtune
from bidir.domains._common import base_row, exec_workers, normalize
from bidir.schema import PairInstance

NAME = "exec"
CELL = "exec"
CELL_OPTS: dict[str, Any] = {}

_FENCE = re.compile(r"^```\w*\n?|```$", re.M)


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    """CRUXEval is 800 problems and ships one split, so train/val/test are carved from it by
    PROGRAM with a fixed seed. 800 is well under the 6,500 every other domain gets; the
    domain config records that, and the budget note in RUN_PLAN.md §4 counts it as the small
    cell it is rather than pretending otherwise."""
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_id"], split=cfg.get("hf_split", "test"))
    rows = []
    for r in ds:
        code, inp, out = (r["code"] or "").strip(), (r["input"] or "").strip(), (r["output"] or "").strip()
        if not code or not inp or not out:
            continue
        rows.append(PairInstance(
            pair_id=f"crux::{r['id']}", domain=NAME, subtask="cruxeval",
            side_a=inp, side_b=out, split="train",
            meta={"code": code, "entry_point": _entry_point(code)}))
    rng = random.Random(int(cfg.get("seed", 17)))
    rng.shuffle(rows)
    n_test, n_val = int(cfg["n_test"]), int(cfg["n_val"])
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
    for r in test:
        r.split = "test"
    for r in val:
        r.split = "val"
    n_train = int(cfg.get("n_train", 0)) or len(train)
    return {"train": train[:n_train], "val": val, "test": test}


def content_key(pair) -> str:
    """Instance identity here is (program, input, output), not (input, output): two different
    CRUXEval programs can share an input/output pair without either being a leak."""
    d = pair if isinstance(pair, dict) else pair.model_dump()
    return normalize((d.get("meta") or {}).get("code", "")) + "\u241f" + \
        normalize(d["side_a"]) + "\u241f" + normalize(d["side_b"])


def _entry_point(code: str) -> str:
    m = re.search(r"^\s*def\s+(\w+)", code, flags=re.M)
    return m.group(1) if m else "f"


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    code = (inst.get("meta") or {}).get("code", "")
    ep = (inst.get("meta") or {}).get("entry_point", "f")
    if direction == "forward":
        return (f"Given this Python function, what does it return for the input shown?\n\n"
                f"{code}\n\nCall:\n{ep}({inst['side_a']})\n\n"
                f"Write only the returned value, as a Python literal.")
    return (f"Given this Python function, find an argument that makes it return the value "
            f"shown.\n\n{code}\n\nRequired return value:\n{inst['side_b']}\n\n"
            f"Write only the argument, as a Python literal.")


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output only the value. Do not restate the call or explain."
    return ("Output only the argument. Any argument that produces the required value is "
            "correct — it does not have to be the one the function was written for.")


# --------------------------------------------------------------------------- scoring

def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Both directions execute. Forward compares the canonicalized return value; reverse RUNS
    the predicted argument and checks the function returns what was asked for."""
    ensure_obtune()
    from obtune.exec.canon import canon_or_none
    from obtune.exec.pool import BatchItem, run_batch

    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]
    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]
    cleaned = [_FENCE.sub("", (o or "")).strip() for o in outputs]

    items, slots = [], []
    for i, (out, inst) in enumerate(zip(cleaned, insts)):
        meta = inst.get("meta") or {}
        if not out:
            continue
        # Forward: run the function on the STORED input and compare to the model's answer.
        # Reverse: run it on the MODEL's argument and compare to the required output.
        args_repr = inst["side_a"] if direction == "forward" else out
        # `BatchItem` takes `program_id` and `args_reprs` — NOT a `cases` list. Passing `cases`
        # raised TypeError on every single trial, so this scorer could never have run; caught
        # 2026-09-10 by feeding it oracle inputs.
        items.append(BatchItem(program_id=str(inst["pair_id"]), language="python",
                               code=meta["code"], entry_point=meta.get("entry_point", "f"),
                               args_reprs=[args_repr]))
        slots.append(i)

    verdicts = run_batch(items, timeout_s=float(cfg.get("exec_timeout_s", 2.0)),
                         workers=exec_workers(cfg)) if items else []

    for i, verdict in zip(slots, verdicts):
        row, out, inst = rows[i], cleaned[i], insts[i]
        case = (verdict.cases or [None])[0]
        got = case.output if (case is not None and case.ok) else None
        row["exec_status"] = getattr(verdict, "status", "ok")
        if direction == "forward":
            row["strict"] = int(got is not None and canon_or_none(_literal(out)) == got)
            row["criterion"] = "canonicalized return value matches execution"
        else:
            want = canon_or_none(_literal(inst["side_b"]))
            row["found_valid_preimage"] = int(got is not None and got == want)
            row["is_stored_input"] = int(normalize(out) == normalize(inst["side_a"]))
            row["strict"] = int(row["found_valid_preimage"])
            row["criterion"] = "predicted argument executes to the required output (any valid preimage)"

    for row in rows:
        row.setdefault("strict", 0)
        row.setdefault("criterion", "execution")
        row["off_target"] = int(row.get("exec_status", "no_run") in ("parse_fail", "no_run", "error"))
    return rows


def _literal(text: str) -> Any:
    import ast
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        return text.strip()
