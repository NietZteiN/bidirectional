"""Auxiliary objectives for the attribution arms (RQ6).

The paper's methodological claim is that when an auxiliary objective on paired data is
credited with restoring the inverse direction, a direction-matched SFT baseline accounts for
the gain. To test that on a method, the method has to be implemented faithfully AND its
direction exposure has to be measurable — otherwise "matched" is an assertion.

So every trainer here reports `direction_exposure`: the number of supervised tokens it spends
on each direction. That number is what the matched `mix*` baseline is matched TO, and it goes
in the run manifest so the matching can be audited rather than trusted.

  UnlikelihoodTrainer   Zan et al. (2024)'s fix for off-target translation. Alongside the
                        usual CE on the correct target, it constructs an
                        INSTRUCTION-CONFLICTING sample — the same instruction, the wrong
                        direction's output — and pushes its likelihood DOWN. Direction
                        exposure is non-zero and negative-signed, which is precisely the
                        interesting case: the method never trains the reverse mapping, it only
                        penalizes applying the forward one when asked for the reverse.

  RoundTripTrainer      Round-trip consistency. The model produces a forward output under
                        no_grad, and the reverse direction is then supervised to reconstruct
                        the original input FROM THE MODEL'S OWN OUTPUT. This is the case where
                        reverse supervision is built into the objective, so its direction
                        exposure is large and positive — and the prediction is that a `mix`
                        arm at the same exposure matches it.

Both subclass TRL's SFTTrainer so the recipe (LoRA, masking, packing=False, batch shape) is
byte-identical to every other arm. Only the loss differs, which is the whole design rule.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from trl import SFTTrainer


def _shift(logits: torch.Tensor, labels: torch.Tensor):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


class DirectionAccountingMixin:
    """Counts supervised tokens per direction, so 'matched on direction exposure' is a number."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_exposure: dict[str, int] = defaultdict(int)

    def _set_signature_columns_if_needed(self):
        # THE AUXILIARY TEXT MUST SURVIVE TO THE COLLATOR. Trainer keeps only the signature
        # columns (TRL: input_ids, labels, seq_lengths), so `conflict_*`/`roundtrip_*` were
        # stripped before AuxiliaryCollator saw them, every batch fell back to plain CE, and the
        # "unlikelihood" adapters of 2026-09-26 were forward SFT under another name. Every other
        # column is still removed exactly as in the `ce` arms.
        super()._set_signature_columns_if_needed()
        from bidir.collators import AUX_TEXT_FIELDS
        self._signature_columns = list(self._signature_columns) + [
            f for f in AUX_TEXT_FIELDS if f not in self._signature_columns]

    def _account(self, key: str, labels: torch.Tensor) -> None:
        self.direction_exposure[key] += int((labels != -100).sum().item())

    def exposure_report(self) -> dict[str, Any]:
        total = sum(abs(v) for v in self.direction_exposure.values()) or 1
        return {"supervised_tokens_by_direction": dict(sorted(self.direction_exposure.items())),
                "share": {k: v / total for k, v in sorted(self.direction_exposure.items())}}


class UnlikelihoodTrainer(DirectionAccountingMixin, SFTTrainer):
    """L = CE(correct target) + lam * UL(conflicting target), UL = -log(1 - p).

    The conflicting sample is supplied by the collator under `conflict_input_ids` /
    `conflict_labels`; `bidir.train` builds it as the same prompt with the OTHER direction's
    text as the completion. Batches without one fall back to plain CE, so a mixed batch is
    still well defined.
    """

    def __init__(self, *args, unlikelihood_lambda: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlikelihood_lambda = float(unlikelihood_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        conflict = {k[len("conflict_"):]: inputs.pop(k) for k in list(inputs)
                    if k.startswith("conflict_")}
        labels = inputs.get("labels")
        # LABELS STAY OUT OF THE FORWARD. TRL >= 1.9 defaults to loss_type="chunked_nll", which
        # patches the model so that a forward WITH labels skips lm_head and returns logits=None
        # (2026-09-26: every unlikelihood/roundtrip pack died here). This loss is computed below
        # from the logits, so the model must run its plain forward.
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits, shifted = _shift(out.logits, labels)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), shifted.view(-1), ignore_index=-100)
        self._account("forward_ce", shifted)
        loss = ce

        if conflict.get("input_ids") is not None:
            c_labels = conflict["labels"]
            c_out = model(input_ids=conflict["input_ids"],
                          attention_mask=conflict.get("attention_mask"))
            c_logits, c_shift = _shift(c_out.logits, c_labels)
            mask = c_shift != -100
            if mask.any():
                logp = F.log_softmax(c_logits.float(), dim=-1)
                tgt = c_shift.clamp_min(0).unsqueeze(-1)
                p = logp.gather(-1, tgt).squeeze(-1).exp()
                # -log(1 - p), clamped away from p = 1 where the gradient is unbounded.
                ul = -torch.log((1.0 - p).clamp_min(1e-6))[mask].mean()
                loss = ce + self.unlikelihood_lambda * ul
                self._account("conflict_unlikelihood", c_shift)
                self.log({"ce": float(ce.detach()), "unlikelihood": float(ul.detach())})
        return (loss, out) if return_outputs else loss


class RoundTripTrainer(DirectionAccountingMixin, SFTTrainer):
    """L = CE(forward) + lam * CE(reverse | the model's OWN forward output).

    The forward output is generated under `no_grad` and greedy, so the reverse pass is
    supervised on what the model actually produces rather than on the gold target. Using the
    gold target instead would make this arm literally a reverse example, which is the
    hypothesis under test and not an implementation of the method.

    `roundtrip_every` throttles generation: the round-trip term is expensive, and applying it
    on a fraction of steps is how the published recipes run it too. The realised fraction is
    recorded so the compute comparison is honest.
    """

    def __init__(self, *args, roundtrip_lambda: float = 1.0, roundtrip_every: int = 4,
                 roundtrip_max_new_tokens: int = 256, **kwargs):
        # Named with the `roundtrip_` prefix because train.py forwards every config key that
        # starts with the loss name; `max_new_tokens` reached SFTTrainer and raised TypeError.
        super().__init__(*args, **kwargs)
        self.roundtrip_lambda = float(roundtrip_lambda)
        self.roundtrip_every = int(roundtrip_every)
        self.max_new_tokens = int(roundtrip_max_new_tokens)
        self._steps = 0
        self._roundtrip_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        rt_prompt = inputs.pop("roundtrip_prompt_ids", None)
        rt_target = inputs.pop("roundtrip_target_ids", None)
        labels = inputs.get("labels")
        # LABELS STAY OUT OF THE FORWARD. TRL >= 1.9 defaults to loss_type="chunked_nll", which
        # patches the model so that a forward WITH labels skips lm_head and returns logits=None
        # (2026-09-26: every unlikelihood/roundtrip pack died here). This loss is computed below
        # from the logits, so the model must run its plain forward.
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits, shifted = _shift(out.logits, labels)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), shifted.view(-1), ignore_index=-100)
        self._account("forward_ce", shifted)
        self._steps += 1

        if rt_prompt is not None and rt_target is not None and self._steps % self.roundtrip_every == 0:
            model.eval()
            with torch.no_grad():
                gen = model.generate(input_ids=rt_prompt, max_new_tokens=self.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=self.processing_class.pad_token_id)
            model.train()
            produced = gen[:, rt_prompt.shape[1]:]
            # Reverse pass: [reverse instruction + what the model just produced] -> original input.
            rev_in = torch.cat([produced, rt_target], dim=1)
            rev_labels = rev_in.clone()
            rev_labels[:, :produced.shape[1]] = -100
            rev_out = model(input_ids=rev_in)
            r_logits, r_shift = _shift(rev_out.logits, rev_labels)
            rt = F.cross_entropy(r_logits.view(-1, r_logits.size(-1)), r_shift.view(-1), ignore_index=-100)
            if torch.isfinite(rt):
                loss = loss + self.roundtrip_lambda * rt
                self._account("roundtrip_reverse_ce", r_shift)
                self._roundtrip_steps += 1
                self.log({"roundtrip_ce": float(rt.detach())})
        return (loss, out) if return_outputs else loss

    def exposure_report(self) -> dict[str, Any]:
        r = super().exposure_report()
        r["roundtrip_steps"] = self._roundtrip_steps
        r["total_steps"] = self._steps
        r["roundtrip_fraction"] = self._roundtrip_steps / max(1, self._steps)
        return r


class ContrastiveTrainer(DirectionAccountingMixin, SFTTrainer):
    """CE + symmetric independently encoded paired InfoNCE; engineering defaults."""
    def __init__(self,*args,contrastive_lambda=0.1,contrastive_temperature=0.1,**kwargs):
        super().__init__(*args,**kwargs)
        self.contrastive_lambda=float(contrastive_lambda)
        self.contrastive_temperature=float(contrastive_temperature)
        self._contrastive_gradient_checked=False
        self._contrastive_pairs=0
        self._contrastive_side_tokens={"a":0,"b":0}

    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        from bidir.contrastive import masked_mean,symmetric_nce
        aux={k:inputs.pop(k) for k in list(inputs) if k.startswith("contrastive_")}
        required={f"contrastive_{side}_{field}" for side in ("a","b") for field in ("input_ids","attention_mask")}
        if not required.issubset(aux):
            raise RuntimeError("contrastive auxiliary fields missing; refusing silent CE fallback")
        directions=inputs.pop("generation_direction",None)
        labels=inputs["labels"]
        ce,out=super().compute_loss(model,inputs,return_outputs=True,**kw)
        if self.contrastive_lambda == 0:
            return (ce,out) if return_outputs else ce
        embeddings=[]
        batch=inputs["input_ids"].shape[0]
        for side in ("a","b"):
            ids=aux[f"contrastive_{side}_input_ids"]
            mask=aux[f"contrastive_{side}_attention_mask"]
            # Base transformer avoids allocating vocabulary logits for isolated side encodings.
            base=model.get_base_model() if hasattr(model,"get_base_model") else model
            hidden=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
            emb=masked_mean(hidden,mask).reshape(batch,5,-1)
            embeddings.append(emb)
            if model.training:
                self._contrastive_side_tokens[side]+=int(mask.sum())
        a,b=embeddings
        nce=symmetric_nce(a[:,0],b[:,0],b,a,self.contrastive_temperature)
        if model.training and not self._contrastive_gradient_checked:
            params=[p for p in model.parameters() if p.requires_grad]
            gradients=torch.autograd.grad(nce,params,retain_graph=True,allow_unused=True)
            norm=sum(float(g.detach().float().abs().sum()) for g in gradients if g is not None)
            if not norm > 0:
                raise RuntimeError("contrastive first-batch gradient gate failed")
            self._contrastive_gradient_checked=True
            print(f"[contrastive] first-batch auxiliary gradient gate passed: {norm}",flush=True)
        if model.training:
            self._contrastive_pairs+=batch
            if directions is None:
                raise RuntimeError("generation direction metadata missing")
            for code,name in ((0,"forward_ce"),(1,"reverse_ce")):
                self._account(name,labels[directions == code])
        # Token-normalized CE may already span the accumulation window. Trainer
        # skips its usual division in that case; independently scale mean NCE.
        scale = 1
        if kw.get("num_items_in_batch") is not None and self.model_accepts_loss_kwargs:
            scale = self.current_gradient_accumulation_steps
        loss=ce+self.contrastive_lambda*nce/scale
        self.log({"ce":float(ce.detach()),"contrastive":float(nce.detach())})
        return (loss,out) if return_outputs else loss

    def exposure_report(self):
        return {**super().exposure_report(),"contrastive_pairs":self._contrastive_pairs,
                "contrastive_side_tokens":self._contrastive_side_tokens,
                "contrastive_gradient_gate_passed":self._contrastive_gradient_checked,
                "negative_count":4,"contrastive_lambda":self.contrastive_lambda,
                "contrastive_temperature":self.contrastive_temperature}


class ExtraCETrainer(DirectionAccountingMixin,SFTTrainer):
    """Six independent dropout CE evaluations averaged within each microstep.

    A fixed compute proxy, not an assertion of equal measured GPU time or FLOPs.
    """
    def __init__(self,*args,extra_ce_passes=6,**kwargs):
        super().__init__(*args,**kwargs)
        self.extra_ce_passes=int(extra_ce_passes)
        if self.extra_ce_passes < 1:
            raise ValueError("positive CE pass count required")

    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        losses=[]
        labels=inputs["labels"]
        directions=inputs.pop("generation_direction",None)
        repeats=self.extra_ce_passes if model.training else 1
        for _ in range(repeats):
            loss,out=super().compute_loss(model,dict(inputs),return_outputs=True,**kw)
            losses.append(loss)
            if model.training:
                if directions is None:
                    raise RuntimeError("generation direction metadata missing")
                for code,name in ((0,"forward_ce"),(1,"reverse_ce")):
                    self._account(name,labels[directions == code])
        loss=sum(losses)/repeats
        return (loss,out) if return_outputs else loss

    def exposure_report(self):
        return {**super().exposure_report(),"extra_ce_passes":self.extra_ce_passes,
                "compute_match":"fixed proxy; actual timings required"}


TRAINERS = {"unlikelihood": UnlikelihoodTrainer, "roundtrip": RoundTripTrainer, "contrastive": ContrastiveTrainer, "extra_ce": ExtraCETrainer}
