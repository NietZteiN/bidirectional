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
