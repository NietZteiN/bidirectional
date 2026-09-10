"""Experiment 5: does the collapsed model still represent the direction it was asked for?

    python -m bidir.mech.direction_probe --domain mt_en-de --model llama32-3b --arms base,sft,mix5

Two reads on the same forward pass, because they answer different halves of H1:

  * PROBE. A logistic probe on the residual stream at the last prompt position, trained to
    predict which direction the instruction asked for. If the probe still decodes the
    direction at high accuracy in a collapsed model, the instruction is represented and the
    failure is downstream of reading it — which is H1. If probe accuracy falls with reverse
    success, the model stopped representing the request and H2 is live.
  * FIRST-TOKEN MASS. Where the model puts probability on the first generated token, under a
    reverse instruction. H1 predicts mass on the forward output mode regardless of the
    instruction.

The probe is trained and tested on DISJOINT pairs with a fixed split, and its chance level is
reported beside its accuracy — a two-class probe at 50 % is measuring nothing, and a probe
that reaches 99 % on four examples is measuring itself.

vLLM does not expose hidden states, so this path uses an HF forward pass. It must build
prompts through the same `bidir.prompts` + obtune template machinery as training and eval, or
the activations come from a different distribution than the accuracies they are compared to.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bidir import arms as arm_registry
from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import load_pairs
from bidir.train import adapter_dir


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="base,sft,mix5")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200, help="pairs; each yields one forward and one reverse prompt")
    ap.add_argument("--layer-fracs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args(argv)

    ensure_obtune()
    import numpy as np
    import torch
    from peft import PeftModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from obtune import prompts as oprompts

    mcfg = resolve_model(a.model)
    dcfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    n_layers = int(mcfg["n_layers"])
    layers = sorted({max(1, min(n_layers, round(float(f) * n_layers))) for f in a.layer_fracs.split(",")})

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # the last position must be the last PROMPT token, not padding

    texts, labels = [], []
    for i in insts:
        for d, lab in (("forward", 0), ("reverse", 1)):
            texts.append(oprompts.render_chat(prompts.build_messages(i, d, mod, "simple"), tok))
            labels.append(lab)
    y = np.array(labels)

    results = []
    for name in [x.strip() for x in a.arms.split(",") if x.strip()]:
        spec = arm_registry.resolve(name)
        adapter = None
        if spec.trains:
            p = adapter_dir(a.domain, a.model, name, a.rank, a.seed) / "final"
            if not p.exists():
                raise SystemExit(f"adapter missing: {p}")
            adapter = str(p)

        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", device_map="cuda")
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        feats = {l: [] for l in layers}
        first_tok = []
        with torch.no_grad():
            for s in range(0, len(texts), a.batch_size):
                batch = tok(texts[s:s + a.batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=int(mcfg["max_seq_len"])).to("cuda")
                out = model(**batch, output_hidden_states=True)
                for l in layers:
                    feats[l].append(out.hidden_states[l][:, -1, :].float().cpu().numpy())
                first_tok.append(out.logits[:, -1, :].argmax(-1).cpu().numpy())
        first_tok = np.concatenate(first_tok)

        row = {"system": name, "chance": float(max(y.mean(), 1 - y.mean())), "layers": {}}
        for l in layers:
            X = np.concatenate(feats[l])
            # 5-fold CV rather than one split: the probe is the measurement, so its variance
            # across splits is part of the reading.
            scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5)
            row["layers"][str(l)] = {"probe_acc": float(scores.mean()), "probe_std": float(scores.std())}
            print(f"  {name:<8s} layer {l:<3d} direction probe = {scores.mean():.3f} "
                  f"+/- {scores.std():.3f} (chance {row['chance']:.3f})", flush=True)

        fwd_first, rev_first = first_tok[0::2], first_tok[1::2]
        row["first_token_same_under_both_instructions"] = float((fwd_first == rev_first).mean())
        print(f"  {name:<8s} first token identical under both instructions: "
              f"{row['first_token_same_under_both_instructions']:.3f}", flush=True)
        results.append(row)

        del model
        torch.cuda.empty_cache()

    out_dir = RESULTS_DIR / "mech" / "direction_probe" / f"{a.domain}__{a.model}_s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(
        {"domain": a.domain, "model": a.model, "seed": a.seed, "n_pairs": len(insts),
         "layers": layers, "n_layers": n_layers,
         "hypothesis": "H1 predicts the probe stays high in a collapsed model (the direction is "
                       "still represented) while first-token mass goes to the forward mode anyway",
         "finished_utc": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2))
    print(f"[probe] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
