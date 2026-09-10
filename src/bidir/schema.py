"""The one instance type every domain produces and every arm consumes.

A `PairInstance` is a pair of texts read in two directions. `side_a -> side_b` is the
forward direction of its cell and `side_b -> side_a` the reverse. The fields are NEVER
swapped to express direction (obtune's invariant): `TrainRow.task` says which way a row is
read, and the prompt builder reads the sides accordingly. That is what keeps a mix arm's
reverse rows byte-identical in content to the forward rows they replaced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Direction = Literal["forward", "reverse"]
Task = Literal["fwd", "rev", "replay"]
DIRECTION_OF_TASK: dict[str, Optional[str]] = {"fwd": "forward", "rev": "reverse", "replay": None}


class PairInstance(BaseModel):
    pair_id: str  # unique within a domain; the split unit and the bootstrap cluster
    domain: str  # code | mt | sql | d2t | fmt | exec | replay
    subtask: str  # code: condition; mt: "en-de"; sql: db_id; fmt: "json-yaml"; ...
    side_a: str
    side_b: str
    split: str  # train | val | test
    meta: dict[str, Any] = Field(default_factory=dict)  # scorer inputs (db path, cases, refs)

    @field_validator("split")
    @classmethod
    def _split_ok(cls, v: str) -> str:
        if v not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {v!r}")
        return v


class TrainRow(PairInstance):
    task: str  # fwd | rev | replay

    @field_validator("task")
    @classmethod
    def _task_ok(cls, v: str) -> str:
        if v not in DIRECTION_OF_TASK:
            raise ValueError(f"task must be one of {sorted(DIRECTION_OF_TASK)}, got {v!r}")
        return v

    @property
    def direction(self) -> Optional[str]:
        return DIRECTION_OF_TASK[self.task]

    @classmethod
    def from_pair(cls, p: PairInstance, task: str) -> "TrainRow":
        return cls(**p.model_dump(), task=task)


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            d = r.model_dump() if isinstance(r, BaseModel) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_pairs(path: str | Path) -> list[PairInstance]:
    return [PairInstance(**d) for d in iter_jsonl(path)]
