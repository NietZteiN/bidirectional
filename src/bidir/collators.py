"""Collator that carries the attribution arms' auxiliary sequences alongside the normal batch.

TRL builds and masks the primary example; this wraps its collator and adds the extra tensors
the objectives in `bidir.losses` need. Wrapping rather than replacing is deliberate: the
primary example must be tokenized, padded and masked EXACTLY as it is in every other arm, or
the attribution comparison measures the collator instead of the objective.

Auxiliary fields are pre-rendered as text by `bidir.train` and tokenized here, so the
tokenizer is applied once, in one place, with one padding side.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

#: Extra text columns an example may carry. Each becomes `<name>_input_ids` (+ labels where
#: the field is supervised).
AUX_TEXT_FIELDS = ("conflict_prompt", "conflict_completion", "roundtrip_prompt", "roundtrip_target")


class AuxiliaryCollator:
    def __init__(self, base_collator, tokenizer, max_length: int):
        self.base = base_collator
        self.tok = tokenizer
        self.max_length = int(max_length)

    def _encode(self, texts: Sequence[str], max_length: Optional[int] = None):
        return self.tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length or self.max_length, add_special_tokens=False)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        aux = {f: [ex.get(f) for ex in features] for f in AUX_TEXT_FIELDS}
        clean = [{k: v for k, v in ex.items() if k not in AUX_TEXT_FIELDS} for ex in features]
        batch = self.base(clean)

        if all(x for x in aux["conflict_prompt"]) and all(x is not None for x in aux["conflict_completion"]):
            full = [p + c for p, c in zip(aux["conflict_prompt"], aux["conflict_completion"])]
            enc = self._encode(full)
            plen = [len(x) for x in self.tok(list(aux["conflict_prompt"]), add_special_tokens=False)["input_ids"]]
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for i, n in enumerate(plen):
                labels[i, :n] = -100          # only the conflicting COMPLETION is penalized
            batch["conflict_input_ids"] = enc["input_ids"]
            batch["conflict_attention_mask"] = enc["attention_mask"]
            batch["conflict_labels"] = labels

        if all(x for x in aux["roundtrip_prompt"]) and all(x for x in aux["roundtrip_target"]):
            # Left padding: `generate` continues from the LAST position, so right padding would
            # make the model continue from pad tokens.
            side, self.tok.padding_side = self.tok.padding_side, "left"
            try:
                batch["roundtrip_prompt_ids"] = self._encode(aux["roundtrip_prompt"])["input_ids"]
            finally:
                self.tok.padding_side = side
            batch["roundtrip_target_ids"] = self._encode(aux["roundtrip_target"])["input_ids"]

        return batch
