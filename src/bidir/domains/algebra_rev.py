"""`algebra` with the training direction mirrored: FACTORING is forward, expansion reverse.

    algebra      forward = factored -> expanded   (expansion: mechanical)
    algebra_rev  forward = expanded -> factored   (factoring: search)

WHY THIS CELL EXISTS. Every collapse measured so far trains a direction the model is already
comparatively good at and loses the one it is worse at. In MT that is unmistakable: `mt_de-en`
trains de->en, where this model scores COMET 0.883, and en->de (0.819) goes to ZERO; `mt_en-de`
trains the weaker direction and the stronger one survives at -12 %. So two explanations fit the
same data exactly:

    (1) DIRECTIONAL  -- training one direction destroys its inverse, whichever that is;
    (2) COMPETENCE   -- training the direction a model is better at destroys the weaker one.

They are not separable in MT, because "toward English" and "the stronger direction" coincide for
every European pair this model knows. They ARE separable here. Expansion is mechanical and
factoring is search: the asymmetry is mathematical, has nothing to do with surface form or
language, and holds for any model that can do algebra at all.

    train EXPANSION (`algebra`, the easy/strong direction)  -> does factoring collapse?
    train FACTORING (`algebra_rev`, the hard/weak direction) -> does expansion collapse?

Under (1) both collapse. Under (2) only the first does. The pair is the experiment; neither cell
answers it alone, which is why this is a cell and not a variant arm.

IMPLEMENTATION. The sides are swapped at build time so that `side_a` is this cell's forward
SOURCE, exactly as `mt_de-en` mirrors `mt_en-de` -- direction stays a property of how a row is
READ (CLAUDE.md §3.1). Everything else DELEGATES to `algebra` with the direction flipped and the
sides flipped back, so the two cells share one scorer and one instruction set and cannot drift
apart. A second copy of the "genuinely factored" criterion is the last thing this domain needs:
it already cost 19 of 40 gold answers once.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from bidir.domains import algebra
from bidir.schema import PairInstance

NAME = "algebra_rev"

#: forward <-> reverse. Applied to every delegated call.
_FLIP = {"forward": "reverse", "reverse": "forward"}


def _as_algebra(inst: Mapping[str, Any]) -> dict[str, Any]:
    """This cell's instance, in `algebra`'s orientation (side_a factored, side_b expanded)."""
    d = dict(inst)
    d["side_a"], d["side_b"] = inst["side_b"], inst["side_a"]
    return d


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    """`algebra`'s pairs with the sides swapped, and the SAME split assignment.

    Built from the identical generator and seed, so an instance that is `algebra`'s test item is
    this cell's test item too. That matters: the two cells are meant to differ in one thing --
    which direction training sees -- and a different split would add a second difference.
    """
    base = algebra.build_pairs(cfg)
    out: dict[str, list[PairInstance]] = {}
    for split, rows in base.items():
        out[split] = [
            PairInstance(
                pair_id=r.pair_id.replace("alg::", "algrev::", 1),
                domain=NAME, subtask=r.subtask,
                side_a=r.side_b,          # expanded: this cell's forward SOURCE
                side_b=r.side_a,          # factored: this cell's forward TARGET
                split=r.split, meta=dict(r.meta or {}),
            )
            for r in rows
        ]
    return out


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    return algebra.instruction(_FLIP[direction], _as_algebra(inst))


def augmented_hint(direction: str) -> str:
    return algebra.augmented_hint(_FLIP[direction])


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Delegated, so `algebra_rev` forward is scored by exactly the criterion `algebra` reverse
    uses (symbolic equivalence AND genuinely factored AND not-echo), and vice versa."""
    return algebra.score_batch(_FLIP[direction], outputs,
                               [_as_algebra(i) for i in insts], cfg)
