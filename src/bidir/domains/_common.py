"""Helpers shared by domain modules: echo detection, thresholds, and the score contract."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional

_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Whitespace- and unicode-normalized, for echo and exact-match tests only."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s or "")).strip()


def is_echo(output: str, source: str, threshold: float = 0.95) -> bool:
    """Output reproduces its own input. Exact after normalization, or near-exact by
    character trigram overlap — a model that copies the input and changes one token is
    echoing, and the strict criterion must not award it."""
    o, s = normalize(output), normalize(source)
    if not o:
        return False
    if o == s:
        return True
    if len(o) < 12 or len(s) < 12:
        return False
    def tri(x: str) -> set[str]:
        return {x[i:i + 3] for i in range(len(x) - 2)}
    a, b = tri(o), tri(s)
    return len(a & b) / max(1, len(a | b)) >= threshold


def base_row(output: str, source: str, target: str) -> dict[str, Any]:
    """The fields every domain's `score` starts from."""
    return {
        "empty_output": int(not normalize(output)),
        "echo": int(is_echo(output, source)),
        "identity_output": int(normalize(output) == normalize(source)),
        "exact_match": int(normalize(output) == normalize(target)),
        "n_chars": len(output or ""),
    }


def threshold_for(cfg: Mapping[str, Any], direction: str, name: str) -> Optional[float]:
    """τ, frozen from the BASE model's score distribution before any tuned model is scored.
    `scripts/15_base_gate.py` writes these into the domain config with the date; a missing
    threshold is an error rather than a default, so a run cannot silently score against a
    number nobody chose."""
    t = (cfg.get("thresholds") or {}).get(direction, {})
    if name not in t:
        raise KeyError(
            f"threshold {name!r} for direction {direction!r} is not frozen in this domain config; "
            "run scripts/15_base_gate.py first (plan §7: thresholds come from the base "
            "distribution before any fine-tuned model is scored)"
        )
    return float(t[name])


def content_key(pair) -> str:
    """What "the same instance" means for the leakage check.

    The default is the pair's own two sides. A domain whose instance identity includes more
    than that overrides it — `exec_pred` does, because two different CRUXEval programs can
    legitimately share an (input, output) pair, and treating that as leakage would be a false
    positive that hides real ones behind noise.
    """
    d = pair if isinstance(pair, dict) else pair.model_dump()
    return normalize(d["side_a"]) + "\u241f" + normalize(d["side_b"])


def exec_workers(cfg: Mapping[str, Any], reserve: int = 1) -> int:
    """How many programs may execute at once. **Derived from the CPUs we actually hold.**

    A HARDCODED WORKER COUNT SILENTLY TURNS CORRECT ANSWERS INTO FAILURES. Measured on
    2026-09-12, scoring 40 known-correct `code` answers on a node where SLURM had granted 8
    CPUs of 64:

        exec_workers=32  ->  strict 0.375,  25/40 exec_status="error"
        exec_workers=4   ->  strict 1.000,  40/40 exec_status="match"

    The configs said 32 (and 16 for `coverage`) regardless of the allocation. obtune's executor
    gives each child a wall-clock budget of `timeout_s * n_cases + 10`; at 4x oversubscription a
    child spends most of that descheduled, gets killed, and `exec_equivalence` folds the timeout
    into `status="error"` — which `forward_success_exec` reads as "not a match", i.e. as a WRONG
    ANSWER. So the measured accuracy of every code-executing domain was a function of how many
    CPUs the job happened to be given, and the error is one-directional: it can only push rates
    down. `code` is the known-positive control, so a capped control would have read as "the
    phenomenon is weaker on this panel" — support for the null, manufactured by a scheduler flag.

    `len(os.sched_getaffinity(0))` is the right quantity because it reflects the cgroup SLURM
    put us in, not the machine's core count (64 on that node, 8 of them ours). One CPU is left
    for the parent, which is doing the scoring while the children run.

    A config may still pin `exec_workers` explicitly — the mech experiments need to — but a
    pinned value that exceeds the allocation RAISES, because oversubscription here is never a
    legitimate choice and its failure mode is silent.
    """

    import os

    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux; os.cpu_count is the best we have
        available = os.cpu_count() or 2

    pinned = cfg.get("exec_workers")
    if pinned is None:
        return max(1, available - reserve)

    pinned = max(1, int(pinned))
    # Oversubscription is never a legitimate choice here, so it raises rather than degrading
    # the numbers. This is the guard that the 2026-09-12 measurement above says is needed: the
    # failure mode is silent, one-directional, and indistinguishable from a model being bad.
    if pinned > available and not cfg.get("allow_exec_oversubscribe"):
        raise RuntimeError(
            f"exec_workers={pinned} exceeds the {available} CPU(s) this process may use. "
            f"Oversubscribing makes obtune's executor kill children on wall clock and report "
            f"the timeout as exec_status='error', which the criterion scores as a wrong answer "
            f"— so this would quietly lower every rate in a code-executing domain.\n"
            f"  Either raise --cpus for the job, or drop exec_workers from the domain config "
            f"and let it be derived from the allocation."
        )
    return pinned


def cluster_key(pair) -> str:
    """The INDEPENDENT unit, for the direction partition and for the bootstrap.

    Usually a pair is its own unit and this is just `pair_id`. `code` is not: its rows are
    (program, obfuscation condition) and five conditions share one source program, so the
    program is the unit and the row is not. Getting this wrong costs two different things, and
    `code` is the known-positive control, so it costs them where it matters most:

      * THE DIRECTION PARTITION. `split_directions` keeps a unit wholly forward or wholly
        reversed, because a unit seen both ways makes a low dose into a small `flip`
        (CLAUDE.md §3.2). Partitioned by row, program P's condition L1b goes forward while its
        L2 goes reverse -- and both sides are derived from P's own code, so the model sees that
        code as input AND as output. At mix50, with ~4.2 conditions per program in the training
        split, that happens for essentially every program.
      * THE CLUSTER BOOTSTRAP. Five correlated rows resampled as five independent observations
        DEFLATES the interval. `code`'s 2,060 test rows are 412 programs, so its effective n is
        a fifth of what the row count suggests, and an equivalence claim read off the row count
        would be roughly twice as confident as the data supports.

    A domain overrides this when its rows are not independent. `code` does.
    """
    d = pair if isinstance(pair, dict) else pair.model_dump()
    return str(d["pair_id"])
