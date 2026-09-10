"""vLLM wrapper: one engine per process, many adapters per pass.

Wraps obtune's `eval_vllm.Engine`, which already carries the LoRARequest registry and the
flashinfer/`__main__`-guard lessons from log/setup/2026-08-30_vllm-unblocked.md. Multi-LoRA
is why an eval pass is one job: every arm in a cell is scored against the same engine, in the
same batch, so a within-pass contrast is not contaminated by engine restarts.

Cross-pass determinism. Greedy decoding is NOT bitwise reproducible across passes — vLLM
batches continuously, so which arms share a pass changes reduction order and occasionally an
argmax. Measured in the workshop paper at 6-8 % of generations differing and 4-8 graded
trials flipping per 1,500. The rule that follows: one pass per table, contrasts only within a
pass, never quote a cross-pass difference finer than 0.5 pp.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from bidir.config import GLOBAL_SEED, ensure_obtune, resolve_model

_ENGINES: dict[tuple, Any] = {}

DEFAULT_ENGINE: dict[str, Any] = {
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "max_lora_rank": 64,
    "max_loras": 8,
    "max_cpu_loras": 32,
    "gpu_memory_utilization": 0.85,
    "seed": GLOBAL_SEED,
}


def get_engine(model_key_or_hf_id: str, ecfg: Optional[Mapping[str, Any]] = None):
    """One engine per (model, config) in a process. Two engines on one GPU would each claim
    `gpu_memory_utilization` of the card and the second would OOM."""
    ensure_obtune()
    from obtune.eval_vllm import Engine

    try:
        hf_id = resolve_model(model_key_or_hf_id)["hf_id"]
    except KeyError:
        hf_id = model_key_or_hf_id
    cfg = {**DEFAULT_ENGINE, **dict(ecfg or {})}
    key = (hf_id, tuple(sorted(cfg.items())))
    if key not in _ENGINES:
        _ENGINES[key] = Engine(hf_id, cfg)
    return _ENGINES[key]


def render(engine, messages_list: Sequence[Sequence[Mapping[str, str]]]) -> list[str]:
    """Prompt text exactly as training rendered it — same `template_mode` path, so the eval
    prompt is a byte-identical prefix of what the trainer saw."""
    ensure_obtune()
    from obtune import prompts as oprompts
    return [oprompts.render_chat(list(m), engine.tokenizer) for m in messages_list]


def generate(engine, messages_list, adapters: Sequence[Optional[str]],
             sampling: Optional[Mapping[str, Any]] = None) -> tuple[list[str], list[int]]:
    s = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": GLOBAL_SEED, **dict(sampling or {})}
    return engine.generate(render(engine, messages_list), s, list(adapters))


def generate_with(model_key_or_hf_id: str, prompts_: Sequence[str], max_tokens: int = 256,
                  system: Optional[str] = None) -> list[str]:
    """Plain single-turn generation from a model that is not the one under test — the frozen
    round-trip parsers and extractors. Loads its own engine; call it once per batch."""
    from bidir.prompts import SYSTEM

    engine = get_engine(model_key_or_hf_id, {"max_loras": 1, "gpu_memory_utilization": 0.45})
    msgs = [[{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": p}]
            for p in prompts_]
    out, _ = generate(engine, msgs, [None] * len(msgs), {"max_tokens": max_tokens})
    return out
