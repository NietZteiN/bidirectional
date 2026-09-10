"""Game of Life: a determined inverse that is computationally hard (catalogue #30).

This domain separates an axis the plan's RQ3 currently runs together. "Invertibility" is two
different properties:

  * INFORMATION-THEORETIC — is the inverse determined by the input at all? The `fmt_det*` ladder
    varies exactly this, and where the answer is no, no amount of reverse data can help.
  * COMPUTATIONAL — is the inverse determined but hard to find? Nothing in the project tests
    this, and it is the more interesting case for a claim about what fine-tuning destroys.

Here every instance HAS a predecessor by construction — the corpus is built by drawing a board
and stepping it forward — so the inverse is always determined. Finding one is nonetheless
NP-hard in general. That gives RQ3 a ceiling derived from the problem rather than from the
corpus, and lets the paper say which kind of hardness a reverse dose can buy its way out of.

The forward direction is a deterministic local rule, so like `diacritics` it is a domain where
forward performance cannot be the thing consuming the model.

Boards are small (default 6x6, wrapped) because the point is to measure direction, not to pose
a search problem no model of this scale can attempt. Base rates are checked by the gate protocol
before any grid is committed, exactly as for every other cell.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from bidir.domains._common import base_row
from bidir.schema import PairInstance

NAME = "automata"
CELL = "automata"
CELL_OPTS: dict[str, Any] = {}

ALIVE, DEAD = "#", "."


def parse_board(text: str) -> list[list[int]] | None:
    rows = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not rows:
        return None
    w = len(rows[0])
    board = []
    for r in rows:
        if len(r) != w or any(c not in (ALIVE, DEAD) for c in r):
            return None
        board.append([1 if c == ALIVE else 0 for c in r])
    return board


def render(board: Sequence[Sequence[int]]) -> str:
    return "\n".join("".join(ALIVE if c else DEAD for c in row) for row in board)


def step(board: Sequence[Sequence[int]]) -> list[list[int]]:
    """One Life generation on a torus. B3/S23."""
    h, w = len(board), len(board[0])
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            n = sum(board[(y + dy) % h][(x + dx) % w]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx))
            out[y][x] = 1 if (n == 3 or (board[y][x] and n == 2)) else 0
    return out


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    rng = random.Random(int(cfg.get("seed", 17)))
    h, w = int(cfg.get("height", 6)), int(cfg.get("width", 6))
    density = float(cfg.get("density", 0.35))
    sizes = {"train": int(cfg["n_train"]), "val": int(cfg["n_val"]), "test": int(cfg["n_test"])}
    out: dict[str, list[PairInstance]] = {}
    seen: set[str] = set()
    n = 0
    for split, count in sizes.items():
        rows: list[PairInstance] = []
        guard = 0
        while len(rows) < count and guard < count * 50:
            guard += 1
            board = [[1 if rng.random() < density else 0 for _ in range(w)] for _ in range(h)]
            nxt = step(board)
            live = sum(sum(r) for r in nxt)
            # All-dead successors are uninformative in both directions and would be a large,
            # trivially-solved mode in the corpus.
            if live == 0 or live == h * w:
                continue
            a, b = render(board), render(nxt)
            if b in seen:
                # Distinct predecessors mapping to one successor are exactly what makes the
                # inverse set-valued; keeping both would put the same reverse question in the
                # corpus twice with different "answers".
                continue
            seen.add(b)
            n += 1
            rows.append(PairInstance(
                pair_id=f"life::{h}x{w}::{n}", domain=NAME, subtask=f"{h}x{w}",
                side_a=a, side_b=b, split=split,
                meta={"height": h, "width": w, "live_next": live,
                      # A predecessor exists by construction — the inverse is DETERMINED here,
                      # and only the search is hard. That is the distinction this cell adds.
                      "determinable": True}))
        if len(rows) < count:
            raise ValueError(f"automata generator produced {len(rows)}/{count} for {split}")
        out[split] = rows
    return out


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    meta = inst.get("meta") or {}
    h, w = meta.get("height", 6), meta.get("width", 6)
    rules = (f"Conway's Game of Life on a {h}x{w} grid whose edges wrap around. A live cell "
             f"('{ALIVE}') with 2 or 3 live neighbours stays alive; a dead cell ('{DEAD}') with "
             f"exactly 3 live neighbours becomes alive; every other cell is dead next step.")
    if direction == "forward":
        return f"{rules}\n\nGive the NEXT state of this grid.\n\nGrid:\n{inst['side_a']}"
    return (f"{rules}\n\nGive any grid whose next state is the one below. There is at least "
            f"one.\n\nNext state:\n{inst['side_b']}")


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return f"Output only the grid, {ALIVE} and {DEAD} characters, one row per line."
    return (f"Output only the predecessor grid, {ALIVE} and {DEAD} characters, one row per line. "
            f"It does not have to be the one the puzzle came from — any grid that steps to the "
            f"target is correct.")


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for out, inst in zip(outputs, insts):
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        row = base_row(out, source, target)
        board = parse_board(out or "")
        meta = inst.get("meta") or {}
        h, w = meta.get("height", 6), meta.get("width", 6)
        ok_shape = board is not None and len(board) == h and len(board[0]) == w
        row["parse_ok"] = int(ok_shape)
        row["off_target"] = int(not ok_shape)
        if not ok_shape:
            row["strict"] = 0
            row["criterion"] = "simulation"
            rows.append(row)
            continue
        if direction == "forward":
            row["strict"] = int(render(step(parse_board(inst["side_a"]))) == render(board))
            row["criterion"] = "simulated next state matches exactly"
        else:
            # SET-VALUED: any grid that steps to the target is correct, not only the one the
            # instance was generated from. Scoring against the stored predecessor would punish
            # exactly the behaviour the task asks for.
            row["found_valid_predecessor"] = int(render(step(board)) == inst["side_b"].strip())
            row["is_stored_predecessor"] = int(render(board) == inst["side_a"].strip())
            row["strict"] = int(row["found_valid_predecessor"])
            row["criterion"] = "any grid that steps to the target (set-valued inverse)"
        rows.append(row)
    return rows
