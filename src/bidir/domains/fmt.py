"""Format conversion, synthetic: JSON<->YAML, JSON<->XML, Markdown table<->CSV.

This domain is cheap to generate and is the only place where INVERTIBILITY IS SET, not
inferred. Natural domains confound invertibility with difficulty — a renaming transform is
both lossy and hard — so RQ3's clean test lives here: the `lossy` variants drop a KNOWN
fraction of leaf values, and the recoverable ceiling under mix50 should track the fraction of
instances whose inverse is still determinable from the input. A ladder that costs almost
nothing to run and that no natural domain can provide.

Both directions are parse-and-compare, so the criterion is exact and needs no threshold:
the output must parse AND be structurally equal to the target. `strict` therefore carries no
tau, which is why this domain is also the sanity check on the harness itself.
"""
from __future__ import annotations

import csv
import io
import json
import random
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import yaml

from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "fmt"
CELL = "fmt"
CELL_OPTS: dict[str, Any] = {}

SUBTASKS = ("json-yaml", "json-xml", "mdtable-csv")

_KEYS = ["id", "name", "status", "owner", "region", "count", "score", "active", "tag", "note",
         "created", "updated", "priority", "channel", "source", "target", "version", "label"]
_WORDS = ["alpha", "bravo", "delta", "echo", "kilo", "lima", "north", "south", "orbit", "pixel",
          "quartz", "ridge", "signal", "timber", "umber", "vector", "willow", "zephyr"]


def _rand_value(rng: random.Random, depth: int) -> Any:
    r = rng.random()
    if depth < 2 and r < 0.25:
        return {rng.choice(_KEYS): _rand_value(rng, depth + 1) for _ in range(rng.randint(2, 4))}
    if depth < 2 and r < 0.35:
        return [_rand_value(rng, depth + 1) for _ in range(rng.randint(2, 4))]
    if r < 0.55:
        return rng.choice(_WORDS)
    if r < 0.75:
        return rng.randint(0, 9999)
    if r < 0.85:
        return round(rng.uniform(0, 100), 2)
    return rng.choice([True, False])


def _rand_doc(rng: random.Random) -> dict[str, Any]:
    return {rng.choice(_KEYS) + str(i): _rand_value(rng, 0) for i in range(rng.randint(3, 7))}


def _rand_table(rng: random.Random) -> list[list[str]]:
    ncol = rng.randint(2, 5)
    header = rng.sample(_KEYS, ncol)
    rows = [[str(rng.choice(_WORDS)) if rng.random() < 0.6 else str(rng.randint(0, 999))
             for _ in range(ncol)] for _ in range(rng.randint(2, 6))]
    return [header] + rows


def to_md_table(t: Sequence[Sequence[str]]) -> str:
    head, body = t[0], t[1:]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def to_csv(t: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(t)
    return buf.getvalue().strip()


def to_xml(d: Mapping[str, Any]) -> str:
    def build(parent: ET.Element, key: str, val: Any) -> None:
        el = ET.SubElement(parent, key)
        if isinstance(val, dict):
            for k, v in val.items():
                build(el, k, v)
        elif isinstance(val, list):
            for v in val:
                build(el, "item", v)
        else:
            el.text = "true" if val is True else "false" if val is False else str(val)
    root = ET.Element("root")
    for k, v in d.items():
        build(root, k, v)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def from_xml(text: str) -> Any:
    def parse(el: ET.Element) -> Any:
        kids = list(el)
        if not kids:
            t = (el.text or "").strip()
            if t in ("true", "false"):
                return t == "true"
            for cast in (int, float):
                try:
                    return cast(t)
                except ValueError:
                    pass
            return t
        if all(k.tag == "item" for k in kids):
            return [parse(k) for k in kids]
        return {k.tag: parse(k) for k in kids}
    return parse(ET.fromstring(text))


def count_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_leaves(v) for v in value)
    return 1


def drop_leaves_counted(value: Any, share: float, rng: random.Random) -> tuple[Any, int]:
    """`drop_leaves`, also returning how many leaves it actually removed.

    The count is what makes the ladder measurable. Under an EXACT criterion an instance is
    recoverable only if NO leaf was dropped, so the instance-level ceiling is not the drop
    share — it is (1 - share)^leaves, which with ~11.7 leaves per document collapses far
    faster than the share suggests. Recording the count per instance lets the ceiling be
    scored against what is actually recoverable instead of against a nominal parameter.
    """
    dropped = 0

    def walk(v: Any) -> Any:
        nonlocal dropped
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if rng.random() < share:
            dropped += 1
            return None
        return v

    return walk(value), dropped


def drop_leaves(value: Any, share: float, rng: random.Random) -> Any:
    """The invertibility ladder: remove `share` of leaf values, keeping the key paths. At
    share=0 the transform is lossless and its inverse is determined; at share=1 only the
    shape survives and no amount of reverse data can recover the values. The recoverable
    ceiling should track this number and nothing else."""
    if isinstance(value, dict):
        return {k: drop_leaves(v, share, rng) for k, v in value.items()}
    if isinstance(value, list):
        return [drop_leaves(v, share, rng) for v in value]
    return None if rng.random() < share else value


# --------------------------------------------------------------------------- build

def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    rng = random.Random(int(cfg.get("seed", 17)))
    lossy = float(cfg.get("lossy_share", 0.0))
    subtasks = list(cfg.get("subtasks", SUBTASKS))
    sizes = {"train": int(cfg["n_train"]), "val": int(cfg["n_val"]), "test": int(cfg["n_test"])}
    out: dict[str, list[PairInstance]] = {}
    n = 0
    for split, count in sizes.items():
        rows: list[PairInstance] = []
        while len(rows) < count:
            st = subtasks[len(rows) % len(subtasks)]
            n += 1
            if st == "mdtable-csv":
                t = _rand_table(rng)
                a, b = to_md_table(t), to_csv(t)
            else:
                d = _rand_doc(rng)
                n_leaf = count_leaves(d)
                if lossy:
                    tgt, n_dropped = drop_leaves_counted(d, lossy, rng)
                else:
                    tgt, n_dropped = d, 0
                a = json.dumps(d, indent=2, sort_keys=True)
                b = (yaml.safe_dump(tgt, sort_keys=True, allow_unicode=True).strip()
                     if st == "json-yaml" else to_xml(tgt))
                meta = {"lossy_share": lossy, "n_leaves": n_leaf, "n_dropped": n_dropped,
                        # Under the exact criterion the REVERSE direction can only succeed when
                        # nothing was dropped. This flag is the ladder's independent variable.
                        "determinable": n_dropped == 0,
                        "information_kept": (n_leaf - n_dropped) / n_leaf if n_leaf else 1.0}
            if st == "mdtable-csv":
                meta = {"lossy_share": lossy, "n_leaves": None, "n_dropped": 0,
                        "determinable": True, "information_kept": 1.0}
            rows.append(PairInstance(pair_id=f"fmt::{st}::{lossy:g}::{n}", domain=NAME, subtask=st,
                                     side_a=a, side_b=b, split=split, meta=meta))
        out[split] = rows
    return out


# --------------------------------------------------------------------------- prompts

_FMT_NAME = {"json": "JSON", "yaml": "YAML", "xml": "XML", "mdtable": "Markdown table", "csv": "CSV"}


def _formats(subtask: str, direction: str) -> tuple[str, str]:
    a, b = subtask.split("-")
    return (a, b) if direction == "forward" else (b, a)


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    src, tgt = _formats(inst["subtask"], direction)
    text = inst["side_a"] if direction == "forward" else inst["side_b"]
    return (f"Convert the following {_FMT_NAME[src]} document into {_FMT_NAME[tgt]}, "
            f"preserving every key and value.\n\n{_FMT_NAME[src]}:\n{text}")


def augmented_hint(direction: str) -> str:
    return "Output only the converted document. Do not add fields, comments, or explanation."


# --------------------------------------------------------------------------- scoring

def _parse(text: str, fmt: str) -> Any:
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        return yaml.safe_load(text)
    if fmt == "xml":
        return from_xml(text)
    if fmt == "csv":
        return [r for r in csv.reader(io.StringIO(text)) if r]
    if fmt == "mdtable":
        rows = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            rows.append(cells)
        return rows
    raise ValueError(fmt)


def _equal(a: Any, b: Any) -> bool:
    """Structural equality across formats: XML and CSV lose Python types, so scalars are
    compared as strings while containers must match in shape and keys."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    if a is None or b is None:
        return a is None and b is None
    return str(a).strip().lower() == str(b).strip().lower()


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for out, inst in zip(outputs, insts):
        src_fmt, tgt_fmt = _formats(inst["subtask"], direction)
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        row = base_row(out, source, target)
        try:
            got = _parse(out or "", tgt_fmt)
            row["parse_ok"] = 1
        except Exception:
            got, row["parse_ok"] = None, 0
        row["off_target"] = int(not row["parse_ok"])
        try:
            want = _parse(target, tgt_fmt)
            row["structural_equal"] = int(row["parse_ok"] and _equal(got, want))
        except Exception:
            row["structural_equal"] = 0
        row["strict"] = int(bool(row["structural_equal"]) and not row["echo"])
        row["criterion"] = "parse & structural equality & not-echo"

        # The ladder needs a CONTINUOUS reading as well as the exact one. Exact match is
        # all-or-nothing per instance, so at any real drop rate almost every instance fails
        # and the rungs stop being distinguishable — leaf recall keeps measuring after that.
        meta = inst.get("meta") or {}
        row["information_kept"] = meta.get("information_kept", 1.0)
        row["determinable"] = int(meta.get("determinable", True))
        if got is not None:
            try:
                want_full = _parse(inst["side_a"] if direction == "reverse" else target, "json")
                row["leaf_recall"] = _leaf_recall(got, want_full)
            except Exception:
                row["leaf_recall"] = 0.0
        else:
            row["leaf_recall"] = 0.0
        # Scored only where the inverse is determined at all: a floor that no reverse dose can
        # move is the RQ3 prediction, and mixing determined and undetermined instances into one
        # rate hides exactly that.
        row["strict_determinable_only"] = row["strict"] if row["determinable"] else None
        rows.append(row)
    return rows


def _leaf_recall(got: Any, want: Any) -> float:
    """Share of the ORIGINAL document's leaves the output reproduces, by key path."""
    def flat(v: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(v, dict):
            out: dict[str, Any] = {}
            for k, x in v.items():
                out.update(flat(x, f"{prefix}/{k}"))
            return out
        if isinstance(v, list):
            out = {}
            for i, x in enumerate(v):
                out.update(flat(x, f"{prefix}/{i}"))
            return out
        return {prefix: v}

    w, g = flat(want), flat(got)
    if not w:
        return 1.0
    hit = sum(1 for k, v in w.items()
              if k in g and str(g[k]).strip().lower() == str(v).strip().lower())
    return hit / len(w)
