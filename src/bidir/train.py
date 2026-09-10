"""LoRA (or full) fine-tuning of one arm on one domain cell.

A port of obtune/src/obtune/cft/train.py to the direction-agnostic schema. What is kept on
purpose: TRL's conversational prompt-completion form with `completion_only_loss=True` and
`packing=False`; overlong examples DROPPED, never truncated (right-truncation eats the
target, which is the whole supervision signal), with the drop rate gated; every resolved
knob written to the run manifest; template adaptation through obtune.prompts so the
training prefix is byte-identical to what eval renders.

    python -m bidir.train --domain mt_en-de --arm mix5 --model llama32-3b --seed 17
    python -m bidir.train ... --dry-run          # build, check, write manifest, no step
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from bidir import arms as arm_registry
from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, PROJECT_ROOT, RUNS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import build_mixture, direction_balance
from bidir.schema import TrainRow

MAX_DROP_FRACTION = 0.25
SCRIPTS_FOR_PROVENANCE = ["src/bidir/train.py", "src/bidir/mixture.py", "src/bidir/prompts.py",
                          "src/bidir/arms.py", "src/bidir/schema.py"]


def adapter_dir(domain: str, model: str, arm: str, rank: int, seed: int, root: str = "adapters") -> Path:
    return RUNS_DIR / root / domain / model / f"{arm}_r{rank}_s{seed}"


def run_id_for(domain: str, model: str, arm: str, seed: int) -> str:
    return f"bidir__{domain}__{model}__{arm}__s{seed}"


def _effective_train_knobs(cfg: Mapping[str, Any], mcfg: Mapping[str, Any]) -> dict[str, Any]:
    t = dict(cfg.get("train", {}))
    for key in ("per_device_batch", "grad_accum", "max_seq_len"):
        if t.get(key) is None:
            t[key] = mcfg[key]
    return t


def _bidir_git() -> dict[str, str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True).stdout.strip() != ""
        return {"bidir_git_sha": sha, "bidir_git_dirty": str(dirty)}
    except Exception:
        return {}


def measure_lengths(examples: Sequence[dict[str, Any]], tasks: Sequence[str], tokenizer: Any,
                    max_seq_len: int) -> tuple[list[int], dict[str, Any]]:
    """Tokenize as TRL will (through obtune's template adaptation) and gate on length."""
    from obtune import prompts as oprompts

    full_texts, prompt_texts = [], []
    for e in examples:
        if isinstance(e["prompt"], str):  # plain-mode text form
            prompt_texts.append(e["prompt"])
            full_texts.append(e["prompt"] + e["completion"])
        else:
            prompt_texts.append(oprompts.render_chat(e["prompt"], tokenizer))
            full_texts.append(oprompts.render_full(list(e["prompt"]) + list(e["completion"]), tokenizer))
    full_lens = [len(x) for x in tokenizer(full_texts, add_special_tokens=False)["input_ids"]]
    prompt_lens = [len(x) for x in tokenizer(prompt_texts, add_special_tokens=False)["input_ids"]]
    keep = [i for i, n in enumerate(full_lens) if n <= max_seq_len]
    keep_set = set(keep)
    sup_by_task: dict[str, int] = defaultdict(int)
    n_by_task: dict[str, int] = defaultdict(int)
    for i in keep:
        sup_by_task[tasks[i]] += max(0, full_lens[i] - prompt_lens[i])
        n_by_task[tasks[i]] += 1
    n = len(full_lens) or 1
    ls = sorted(full_lens)
    stats = {
        "n_examples": len(examples), "n_kept": len(keep), "n_dropped": len(examples) - len(keep),
        "drop_rate": (len(examples) - len(keep)) / n,
        "dropped_by_task": dict(sorted(Counter(tasks[i] for i in range(len(tasks)) if i not in keep_set).items())),
        "max_seq_len": max_seq_len, "len_mean": sum(full_lens) / n, "len_p50": ls[n // 2],
        "len_p95": ls[min(n - 1, int(0.95 * n))], "len_max": max(full_lens) if full_lens else 0,
        "supervised_tokens_by_task": dict(sorted(sup_by_task.items())),
        "sequence_tokens_total": sum(full_lens[i] for i in keep),
        "supervised_tokens_total": sum(sup_by_task.values()),
        "mean_supervised_tokens_by_task": {k: sup_by_task[k] / n_by_task[k] for k in sorted(sup_by_task) if n_by_task[k]},
    }
    return keep, stats


def _attach_auxiliary(examples, rows, spec, domain_mod, tokenizer, oprompts):
    """Render the auxiliary sequences the attribution objectives consume, as TEXT.

    `unlikelihood` (Zan et al.): the SAME prompt with the OTHER direction's text as the
    completion — an instruction-conflicting sample whose likelihood the loss pushes down. Note
    it never supervises the reverse mapping; it only penalizes applying the forward one when
    the reverse was asked for, which is why its direction exposure is negative-signed.

    `roundtrip`: the reverse-direction prompt, and the original input as the reconstruction
    target. The model's own forward output is spliced in at training time (see
    `losses.RoundTripTrainer`) — using the gold target here instead would make this arm
    literally a reverse example, which is the hypothesis under test.
    """
    out = []
    for ex, row in zip(examples, rows):
        e = dict(ex)
        inst = row.model_dump()
        if spec.loss == "unlikelihood":
            fwd_prompt = oprompts.render_chat(prompts.build_messages(inst, "forward", domain_mod), tokenizer)
            e["conflict_prompt"] = oprompts.render_chat(
                prompts.build_messages(inst, "reverse", domain_mod), tokenizer)
            # The wrong answer under a reverse instruction is the forward output.
            e["conflict_completion"] = prompts.completion_for(inst, "forward")
        elif spec.loss == "roundtrip":
            e["roundtrip_prompt"] = oprompts.render_chat(
                prompts.build_messages(inst, "forward", domain_mod), tokenizer)
            e["roundtrip_target"] = prompts.completion_for(inst, "reverse")
        out.append(e)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, help="cell name, e.g. mt_en-de, sql, code")
    ap.add_argument("--arm", required=True, help=f"one of {sorted(arm_registry.ARMS)}")
    ap.add_argument("--model", required=True, help="key in configs/models.yaml")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--train-config", default="train/_base_lora.yaml")
    ap.add_argument("--out", default=None, help="override the adapter output directory")
    ap.add_argument("--init-adapter", default=None, help="override the adapter to continue from (relearn-k)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="build everything, run every check, no optimizer step")
    ap.add_argument("--cpu", action="store_true")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_obtune()
    spec = arm_registry.resolve(args.arm)
    if not spec.trains:
        raise SystemExit("`base` is never trained")
    if spec.aux_tasks and args.domain != "code":
        raise SystemExit(f"arm {spec.name!r} needs the CFT auxiliary pools, which exist only for `code`")

    cfg = load_config(args.train_config)
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = int(args.seed)
    if spec.epochs is not None:
        cfg["train"]["epochs"] = float(spec.epochs)
    mcfg = resolve_model(args.model)
    tcfg = _effective_train_knobs(cfg, mcfg)
    seed = int(tcfg["seed"])
    domain = domains.get(args.domain)

    gpu_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_visible = ""

    from obtune import prompts as oprompts
    from obtune.seedutil import set_seed

    set_seed(seed)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from obtune.provenance import RunManifest, sha256_dir

    rank = int(cfg["peft"]["r"])
    out_dir = Path(args.out) if args.out else adapter_dir(args.domain, args.model, spec.name, rank, seed,
                                                          root="adapters_fullft" if spec.full_ft else "adapters")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id_for(args.domain, args.model, spec.name, seed)

    tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    render_mode = oprompts.template_mode(tokenizer)
    if mcfg.get("render_mode") and mcfg["render_mode"] != render_mode:
        raise SystemExit(f"models.yaml says render_mode={mcfg['render_mode']} but the tokenizer resolves to "
                         f"{render_mode}; fix the config before training")

    # ---- data ------------------------------------------------------------------------
    train_rows = build_mixture(spec, args.domain, "train", seed)
    val_rows = build_mixture(spec, args.domain, "val", seed)
    if tcfg.get("val_size"):
        val_rows = val_rows[: int(tcfg["val_size"])]
    balance = direction_balance(train_rows)
    print(f"[bidir.train] {args.domain} {spec.name}: {json.dumps(balance)}", flush=True)

    def to_examples(rows: Sequence[TrainRow]) -> list[dict[str, Any]]:
        return [oprompts.to_trl_example(prompts.build_example(r.model_dump(), domain), tokenizer) for r in rows]

    train_ex, val_ex = to_examples(train_rows), to_examples(val_rows)
    if spec.loss != "ce":
        train_ex = _attach_auxiliary(train_ex, train_rows, spec, domain_mod, tokenizer, oprompts)
        val_ex = _attach_auxiliary(val_ex, val_rows, spec, domain_mod, tokenizer, oprompts)
    keep, length_stats = measure_lengths(train_ex, [r.task for r in train_rows], tokenizer, int(tcfg["max_seq_len"]))
    print(f"[bidir.train] lengths: {json.dumps(length_stats)}", flush=True)
    train_ex = [train_ex[i] for i in keep]
    train_rows = [train_rows[i] for i in keep]
    vkeep, val_length_stats = measure_lengths(val_ex, [r.task for r in val_rows], tokenizer, int(tcfg["max_seq_len"]))
    val_ex = [val_ex[i] for i in vkeep]

    train_ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex) if val_ex else None

    dataset_meta = {"domain": args.domain, "arm": spec.name, "tasks": list(spec.tasks),
                    "reverse_fraction": spec.reverse_fraction, "replay_share": spec.replay_share,
                    "mixed_task": spec.mixed_task, "relearn_k": spec.relearn_k, "init_from": spec.init_from,
                    "full_ft": spec.full_ft, "n_train": len(train_rows), "n_val": len(val_ex),
                    "balance": balance, "lengths": length_stats, "val_lengths": val_length_stats,
                    **prompts.provenance_block(), **oprompts.render_provenance(tokenizer)}

    init_adapter = args.init_adapter
    if spec.init_from and not init_adapter:
        init_adapter = str(adapter_dir(args.domain, args.model, spec.init_from, rank, seed) / "final")
    if init_adapter and not Path(init_adapter).exists():
        raise SystemExit(f"--init-adapter {init_adapter} does not exist (train {spec.init_from} first)")

    manifest = (
        RunManifest(
            experiment=f"bidir/{args.domain}", run_id=run_id, seed=seed,
            config_path=str(cfg.get("_config_path", args.train_config)),
            config_resolved={**{k: v for k, v in cfg.items() if not k.startswith("_")}, "train": tcfg},
            model_hf_id=mcfg["hf_id"],
            adapter={"path": str(out_dir), "train_cond": f"{args.domain}:{spec.name}", "rank": rank,
                     "base_model": mcfg["hf_id"], "init_adapter": init_adapter, "full_ft": spec.full_ft},
            gpu_visible=gpu_visible,
            extra={"dataset": dataset_meta, **_bidir_git()},
        )
        .capture_git()
        .hash_scripts([str(PROJECT_ROOT / p) for p in SCRIPTS_FOR_PROVENANCE])
    )
    manifest.write(out_dir)

    if length_stats["drop_rate"] > MAX_DROP_FRACTION:
        raise SystemExit(f"dropped {length_stats['drop_rate']:.1%} at max_seq_len={tcfg['max_seq_len']} "
                         f"(p95={length_stats['len_p95']}, max={length_stats['len_max']}); raise max_seq_len")
    if not train_rows:
        raise SystemExit("no training rows survived length filtering")

    # ---- model -----------------------------------------------------------------------
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    dtype = torch.bfloat16 if tcfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_id"], dtype=dtype if use_cuda else torch.float32,
        attn_implementation=tcfg.get("attn_implementation", "sdpa"), device_map=None)
    model.config.use_cache = False

    peft_cfg = None
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    elif not spec.full_ft:
        kw: dict[str, Any] = {}
        if mcfg.get("peft_exclude_modules"):
            kw["exclude_modules"] = list(mcfg["peft_exclude_modules"])
        peft_cfg = LoraConfig(r=rank, lora_alpha=int(cfg["peft"]["alpha"]), lora_dropout=float(cfg["peft"]["dropout"]),
                              target_modules=list(cfg["peft"]["target_modules"]),
                              task_type=cfg["peft"].get("task_type", "CAUSAL_LM"), bias="none", **kw)

    lr = float(tcfg.get("lr", 1e-4)) if not spec.full_ft else float(tcfg.get("full_ft_lr", 1e-5))
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg["per_device_batch"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]),
        num_train_epochs=float(tcfg.get("epochs", 3)),
        learning_rate=lr,
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        max_length=int(tcfg["max_seq_len"]),
        packing=False,
        completion_only_loss=True,
        bf16=use_cuda and dtype is torch.bfloat16,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)) and use_cuda,
        save_strategy=tcfg.get("save_strategy", "epoch"),
        save_total_limit=None,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=int(tcfg.get("eval_steps", 200)),
        per_device_eval_batch_size=int(tcfg["per_device_batch"]),
        logging_steps=int(tcfg.get("logging_steps", 20)),
        seed=seed, data_seed=seed, report_to=[], use_cpu=not use_cuda,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        dataloader_num_workers=2,
    )
    trainer_cls, trainer_kw = SFTTrainer, {}
    if spec.loss != "ce":
        from bidir.losses import TRAINERS
        trainer_cls = TRAINERS[spec.loss]
        aux = dict(cfg.get("attribution", {}) or {})
        trainer_kw = {k: v for k, v in aux.items() if k.startswith(spec.loss)}
    trainer = trainer_cls(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=val_ds,
                          processing_class=tokenizer, peft_config=peft_cfg, **trainer_kw)
    if spec.loss != "ce":
        from bidir.collators import AuxiliaryCollator
        trainer.data_collator = AuxiliaryCollator(trainer.data_collator, tokenizer,
                                                  int(tcfg["max_seq_len"]))

    # Loss-mask gate on one real batch (obtune CLAUDE.md §4): prompt tokens must be -100.
    batch = next(iter(trainer.get_train_dataloader()))
    supervised = int((batch["labels"] != -100).sum())
    total = int(batch["attention_mask"].sum())
    if supervised == 0 or supervised >= total:
        raise SystemExit(f"loss mask gate failed: {supervised}/{total} tokens supervised")
    print(f"[bidir.train] loss-mask gate ok: {supervised}/{total} tokens supervised in batch 1", flush=True)

    if args.dry_run:
        manifest.extra["dry_run"] = True
        manifest.finalize().write(out_dir)
        print("[bidir.train] DRY RUN — no optimizer step taken", flush=True)
        return 0

    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    summary = {
        "run_id": run_id, "arm": spec.name, "domain": args.domain, "model": args.model, "seed": seed,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"), "steps": result.global_step,
        "n_train": len(train_rows), "n_val": len(val_ex), "balance": balance, "lengths": length_stats,
        "checkpoints": sorted(p.name for p in out_dir.glob("checkpoint-*")),
    }
    if hasattr(trainer, "exposure_report"):
        # What the matched `mix*` baseline is matched TO. In the manifest so the matching can
        # be audited rather than asserted (RQ6).
        summary["direction_exposure"] = trainer.exposure_report()
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    manifest.extra["training_summary"] = summary
    manifest.adapter["sha256"] = sha256_dir(out_dir / "final")
    manifest.finalize().write(out_dir)
    print(f"[bidir.train] done: {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
