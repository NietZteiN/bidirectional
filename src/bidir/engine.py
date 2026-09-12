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

import sys
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


#: Fraction of the card the resident engines may claim between them. vLLM's
#: `gpu_memory_utilization` is a share of TOTAL device memory that each engine reserves for
#: itself, so two engines at 0.85 and 0.45 do not negotiate -- the second simply cannot get
#: what it asked for. 0.95 rather than 1.0 leaves room for the CUDA context and cuBLAS
#: workspaces, which are outside every engine's budget.
_UTILISATION_CEILING = 0.95


def _claimed_utilisation() -> float:
    return sum(e_cfg for _, e_cfg in _ENGINE_UTIL.items())


#: (hf_id, key) -> the utilisation that engine claimed. Read by the budget check below.
_ENGINE_UTIL: dict[tuple, float] = {}


def get_engine(model_key_or_hf_id: str, ecfg: Optional[Mapping[str, Any]] = None):
    """One engine per (model, config) in a process.

    THE BUDGET IS CHECKED, NOT ASSUMED. `gpu_memory_utilization` is a share of the whole
    device that each engine reserves outright, so engines do not share gracefully: a model
    under test at 0.85 plus a frozen round-trip parser at 0.45 asks for 1.30 of the card and
    the second load dies inside vLLM's memory profiler, several frames from anything naming
    the first model. SQL and D2T both need a second resident model to score at all, so this
    is a normal path rather than an exotic one. Raising here names both engines and the sum.
    """
    ensure_obtune()
    from obtune.eval_vllm import Engine

    try:
        hf_id = resolve_model(model_key_or_hf_id)["hf_id"]
    except KeyError:
        hf_id = model_key_or_hf_id
    cfg = {**DEFAULT_ENGINE, **dict(ecfg or {})}
    key = (hf_id, tuple(sorted(cfg.items())))
    if key not in _ENGINES:
        want = float(cfg.get("gpu_memory_utilization", 0.85))
        held = _claimed_utilisation()
        if held + want > _UTILISATION_CEILING:
            resident = ", ".join(f"{k[0]}@{v:.2f}" for k, v in _ENGINE_UTIL.items())
            raise RuntimeError(
                f"GPU utilisation budget exceeded: {resident} already claim {held:.2f} of the "
                f"card and {hf_id} wants {want:.2f} (ceiling {_UTILISATION_CEILING}).\n"
                f"  Two engines each reserve their share of TOTAL device memory; they do not "
                f"negotiate.\n"
                f"  Fix the domain config, not this ceiling: a domain whose criterion needs a "
                f"second resident model must budget for both (see configs/domains/sql.yaml)."
            )
        _ENGINES[key] = Engine(hf_id, cfg)
        _ENGINE_UTIL[key] = want
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
                  system: Optional[str] = None, util_override: Optional[float] = None) -> list[str]:
    """Plain single-turn generation from a model that is not the one under test — the frozen
    round-trip parsers and extractors. Loads its own engine; call it once per batch."""
    from bidir.prompts import SYSTEM

    # The utilisation comes from the caller's config, because what is left of the card depends
    # on what the model under test claimed. Hardcoding 0.45 here asked for 1.30 of the device
    # whenever the domain engine was at its 0.85 default.
    util = float(util_override) if util_override is not None else 0.45
    engine = get_engine(model_key_or_hf_id, {"max_loras": 1, "gpu_memory_utilization": util})
    msgs = [[{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": p}]
            for p in prompts_]
    out, _ = generate(engine, msgs, [None] * len(msgs), {"max_tokens": max_tokens})
    return out


def shutdown_and_exit(rc: int = 0) -> None:
    """Flush, release every engine, and leave -- without waiting on vLLM's teardown.

    MEASURED, NOT PRECAUTIONARY. Job 391263 printed its final gate summary and then sat
    RUNNING for another 6m14s on an H200 without writing its status file, i.e. the python
    process had not exited. With VLLM_WORKER_MULTIPROC_METHOD=spawn the engine core is a
    child process, and interpreter shutdown waits on it; it does not always come back.

    Two reasons that is worth a hard exit rather than a longer walltime:
      * this project's share of the juno QoS pool is ONE running job, so a hung teardown
        blocks the next cell for as long as it lasts;
      * a job killed at the walltime is recorded as a failure, so a teardown hang can turn a
        completed eval into a failed one and lose the run's provenance.

    Every script that calls this has already written its results to disk. The exit code is what
    the caller would have returned, so the `finish` trap in the sbatch template still records
    the outcome -- os._exit ends the process, it does not skip the shell's trap.

    Best effort is attempted first: dropping our references and collecting gives vLLM's own
    atexit path a chance to run cleanly on the engines it can close.
    """
    import gc
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    _ENGINES.clear()
    _ENGINE_UTIL.clear()
    gc.collect()
    os._exit(rc)
