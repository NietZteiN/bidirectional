"""Expansion <-> factorization: the formal-reasoning domain (catalogue #56).

Forward (factored -> expanded) is mechanical: distribute and collect. Reverse (expanded ->
factored) is search. The asymmetry is real, well known, and has nothing to do with surface
form — which is what makes it a different kind of evidence from the surface transforms the rest
of the paper runs.

Chosen over #55 (differentiation <-> integration), which is the better-known task, because
integration's failure mode is contaminated: a great many elementary functions have no elementary
antiderivative, so "the model failed" and "no answer exists" are entangled unless the generator
is carefully restricted. Every instance here has an answer, the verifier is one SymPy call, and
the criterion cannot be argued with.

Criterion. Forward: the expansion is correct, checked by `expand(pred) - expand(gold) == 0`
rather than by string match, because `x**2 - 2*x - 15` and `-15 - 2*x + x**2` are the same
polynomial. Reverse: the factorization must both *be* a factorization (SymPy's `factor` leaves
it unchanged) and multiply back to the target. Checking only the second would award the identity
"factorization" — the expanded polynomial itself — which is the algebraic form of an echo.
"""
from __future__ import annotations

import random
from math import gcd
from typing import Any, Mapping, Sequence

from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "algebra"
CELL = "algebra"
CELL_OPTS: dict[str, Any] = {}

VARS = ("x", "y", "z")


def _sympy():
    import sympy
    return sympy


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    sp = _sympy()
    rng = random.Random(int(cfg.get("seed", 17)))
    max_root = int(cfg.get("max_root", 12))
    sizes = {"train": int(cfg["n_train"]), "val": int(cfg["n_val"]), "test": int(cfg["n_test"])}
    degrees = list(cfg.get("degrees", [2, 2, 3, 3, 4]))

    out: dict[str, list[PairInstance]] = {}
    seen: set[str] = set()
    n = 0
    for split, count in sizes.items():
        rows: list[PairInstance] = []
        guard = 0
        while len(rows) < count and guard < count * 60:
            guard += 1
            deg = degrees[len(rows) % len(degrees)]
            # Two variables allowed at ANY degree. Restricting low degrees to one variable
            # capped the distinct primitive quadratics at roughly 2,145, which is fewer than the
            # corpus needs — the generator starved at 5,840 of 6,500 rather than producing a
            # biased sample, which is the failure mode to prefer but still a failure.
            nvars = 1 if rng.random() < 0.6 else 2
            syms = [sp.Symbol(v) for v in VARS[:nvars]]
            factors = []
            for _ in range(deg):
                v = syms[rng.randrange(len(syms))]
                # PRIMITIVE factors only: gcd(a, b) == 1. Otherwise the constructed form is not
                # the complete factorization — (3x + 12) is 3(x + 4) — and the reverse target
                # would be an answer the task's own wording ("factor completely") calls wrong.
                for _ in range(40):
                    a = rng.choice([1, 1, 1, 2, 3, 4])   # mostly monic: keeps roots readable
                    b = rng.randint(-max_root, max_root)
                    if b == 0 or gcd(a, abs(b)) == 1:
                        break
                else:
                    a, b = 1, rng.randint(1, max_root)
                factors.append(a * v + b)
            expanded = sp.expand(sp.Mul(*factors))
            if expanded.is_number or not expanded.is_Add:
                continue
            # The factored form is the one we BUILT, not SymPy's canonicalization of it.
            # Calling `factor()` per candidate was both the generator's bottleneck and the
            # reason it starved at 3,921 of 6,500: it is slow, and its output sometimes differs
            # from the constructed form in ways that tripped the round-trip guard. The scorer
            # accepts any expression that expands correctly AND is genuinely factored, so the
            # canonical form is not required here — only a correct one.
            factored = sp.Mul(*factors, evaluate=False)
            key = sp.srepr(expanded)
            if key in seen:
                continue
            seen.add(key)
            n += 1
            rows.append(PairInstance(
                pair_id=f"alg::{deg}::{n}", domain=NAME, subtask=f"deg{deg}_{nvars}var",
                side_a=str(factored), side_b=str(expanded), split=split,
                meta={"degree": deg, "n_vars": nvars, "determinable": True}))
        if len(rows) < count:
            raise ValueError(f"algebra generator produced {len(rows)}/{count} for {split}")
        out[split] = rows
    return out


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    if direction == "forward":
        return ("Expand this expression and collect like terms. Write the result as a "
                "polynomial.\n\nExpression:\n" + inst["side_a"])
    return ("Factor this polynomial completely over the integers. Write it as a product of "
            "factors.\n\nPolynomial:\n" + inst["side_b"])


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output only the expanded polynomial. Do not show working."
    return ("Output only the factored product, for example (x + 3)*(x - 5). Do not restate the "
            "polynomial you were given.")


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    sp = _sympy()
    rows = []
    for out, inst in zip(outputs, insts):
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        row = base_row(out, source, target)
        expr = None
        try:
            expr = sp.sympify((out or "").strip().replace("^", "**"))
            row["parse_ok"] = 1
        except Exception:
            row["parse_ok"] = 0
        row["off_target"] = int(not row["parse_ok"])
        row["equivalent"] = 0
        row["is_factored"] = 0
        if expr is not None:
            try:
                row["equivalent"] = int(sp.simplify(sp.expand(expr) - sp.expand(sp.sympify(target))) == 0)
            except Exception:
                row["equivalent"] = 0
            if direction == "reverse":
                try:
                    # "Genuinely factored" is: a product at the top level, whose expansion is
                    # not itself. That rejects the expanded polynomial — the algebraic form of
                    # an echo — while accepting any correct factorization.
                    #
                    # It deliberately does NOT demand SymPy's canonical form. Requiring
                    # `factor(expr) == expr` rejected 19 of 40 GOLD factorizations, because
                    # SymPy pulls constants out front and the orderings differ. A criterion
                    # that fails half of all correct answers measures the criterion.
                    row["is_factored"] = int((not expr.is_Add) and sp.expand(expr) != expr)
                except Exception:
                    row["is_factored"] = 0
        if direction == "forward":
            row["strict"] = int(row["equivalent"] and not row["echo"])
            row["criterion"] = "symbolic equivalence after expansion & not-echo"
        else:
            row["strict"] = int(row["equivalent"] and row["is_factored"] and not row["echo"])
            row["criterion"] = "symbolic equivalence & genuinely factored & not-echo"
        rows.append(row)
    return rows
