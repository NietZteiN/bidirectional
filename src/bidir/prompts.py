"""Prompt construction — one system prompt for BOTH directions, direction in the user turn.

Why one system prompt. obtune's `flipsym` control showed no difference between per-direction
personas and a shared one at 1.5B, and a shared prompt removes the confound the SRH phase
would otherwise have to argue away: a model that holds two disjoint circuits can look
bidirectional if the directions arrive under different system prompts.

The invariant that matters. The reverse TRAINING prompt is byte-identical to the reverse
EVALUATION prompt under the `simple` strategy, by construction: both call `build_messages`
with the same arguments. `tests/test_prompts.py` asserts it on real domain rows.

Domain modules supply only `instruction(direction, inst) -> str` (the user-turn text, input
included) and optionally `augmented_hint(direction) -> str`. Everything else is here.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence

PROMPT_VERSION = "bidir_v1"

SYSTEM = (
    "You are a precise transformation tool. You convert the given input into the requested "
    "form exactly as instructed, preserving its meaning, and you never change what it says or "
    "computes. Reply with the result only: no explanation, no commentary, and no markdown "
    "code fences."
)

STRATEGIES = ("simple", "few_shot", "cot", "augmented")

COT_SUFFIX = (
    "\n\nFirst think through the conversion step by step inside <thinking> and </thinking> "
    "tags. Then, on a new line starting with `ANSWER:`, give the final result and nothing else."
)

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER:\s*", re.IGNORECASE)


def build_messages(
    inst: Mapping[str, Any],
    direction: str,
    domain: Any,
    strategy: str = "simple",
    shots: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[dict[str, str]]:
    """[system, (demo user/assistant)*, user]. `domain` is a domain module."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known {STRATEGIES}")
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    if strategy == "few_shot":
        if not shots:
            raise ValueError("few_shot needs demonstration pairs")
        for s in shots:
            msgs.append({"role": "user", "content": domain.instruction(direction, s)})
            msgs.append({"role": "assistant", "content": completion_for(s, direction)})
    user = domain.instruction(direction, inst)
    if strategy == "augmented":
        hint = getattr(domain, "augmented_hint", lambda d: "")(direction)
        if hint:
            user = f"{user}\n\n{hint}"
    elif strategy == "cot":
        user = user + COT_SUFFIX
    msgs.append({"role": "user", "content": user})
    return msgs


def completion_for(inst: Mapping[str, Any], direction: str) -> str:
    if direction == "forward":
        return inst["side_b"]
    if direction == "reverse":
        return inst["side_a"]
    raise ValueError(f"direction must be forward|reverse, got {direction!r}")


def input_for(inst: Mapping[str, Any], direction: str) -> str:
    return inst["side_a"] if direction == "forward" else inst["side_b"]


def build_example(row: Mapping[str, Any], domain: Any) -> dict[str, Any]:
    """A training example in TRL's conversational prompt-completion form.

    `replay` rows carry their own user/assistant text in side_a/side_b under the same
    system prompt, so the replay arm differs from `sft` only in what the replaced share of
    rows contains — not in format.
    """
    task = row["task"]
    if task in ("pos", "neg"):
        # Auxiliary judgement tasks (CFT). The domain owns their format because only it knows
        # what "these two programs are equivalent" looks like.
        return domain.aux_example(task, row)
    if task == "replay":
        prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["side_a"]}]
        return {"prompt": prompt, "completion": [{"role": "assistant", "content": row["side_b"]}]}
    direction = "forward" if task == "fwd" else "reverse"
    prompt = build_messages(row, direction, domain, strategy="simple")
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": completion_for(row, direction)}]}


def extract_answer(raw: str, strategy: str = "simple") -> str:
    """Strip fences; for `cot`, keep only what follows ANSWER:."""
    text = raw
    if strategy == "cot":
        m = list(_ANSWER_RE.finditer(text))
        text = text[m[-1].end():] if m else text.split("</thinking>")[-1]
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    # A fence the model opened but never closed still hides the answer behind a marker line.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()


def template_sha256() -> str:
    """Hash of every prompt constant, recorded in each run manifest."""
    blob = json.dumps({"version": PROMPT_VERSION, "system": SYSTEM, "cot": COT_SUFFIX,
                       "strategies": STRATEGIES}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def provenance_block() -> dict[str, str]:
    return {"prompt_version": PROMPT_VERSION, "prompt_template_sha256": template_sha256()}
