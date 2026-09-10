"""The arm registry — one place that says what each arm IS.

Kept as data (obtune/src/obtune/srh/arms.py's pattern) because the design rests on arms
differing in exactly one thing at a time, and that is auditable in a table and not across
thirteen YAML files. Cell configs name an arm and a cell; everything else comes from here.

Budget matching. Every `mix*` arm REPLACES forward pairs with their own reversal, partitioned
by pair_id, so instance count, sequence tokens and optimizer steps stay matched to `sft` and
the only thing varying along the dose ladder is the share of pairs seen backwards. `replay`
replaces the same share with generic instruction rows, so `replay - sft` is ordinary
forgetting and `mix - replay` is what direction specifically buys. `flip` and `fwd2x` are the
doubled-budget references and are the only arms that cost 2x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ArmSpec:
    name: str
    tasks: tuple[str, ...] = ("fwd",)
    #: replace this share of forward PAIRS with their reversal (by pair_id)
    reverse_fraction: Optional[float] = None
    #: replace this share of forward PAIRS with generic instruction rows (tulu-3 SFT mix)
    replay_share: Optional[float] = None
    #: forward pairs are 20 % of a five-task SFT set drawn from the other domains
    mixed_task: bool = False
    epochs: Optional[float] = None
    full_ft: bool = False
    #: continue training from this arm's adapter (relearn-k)
    init_from: Optional[str] = None
    #: train on exactly k reversed pairs (relearn-k); None = whole mixture
    relearn_k: Optional[int] = None
    #: ce | unlikelihood | roundtrip — attribution arms add an objective on paired data
    loss: str = "ce"
    #: extra task pools for the CFT arm (code only)
    aux_tasks: tuple[str, ...] = ()
    role: str = ""
    #: for provenance: which arm this one is compared against by construction
    matched_to: Optional[str] = None

    @property
    def trains(self) -> bool:
        return self.name != "base"

    @property
    def cost_units(self) -> float:
        """Adapter-time relative to `sft` (for packing and budgets)."""
        if not self.trains:
            return 0.0
        u = 1.0
        if self.epochs:
            u *= self.epochs / 3.0
        if "fwd" in self.tasks and "rev" in self.tasks and self.reverse_fraction is None:
            u *= 2.0  # flip: both directions of every pair
        if self.relearn_k is not None:
            u = 0.15
        if self.full_ft:
            u *= 2.0
        if self.loss != "ce":
            u *= 1.5
        return u


ARMS: dict[str, ArmSpec] = {
    "base": ArmSpec("base", tasks=(), role="untouched control"),
    "sft": ArmSpec("sft", ("fwd",), role="forward only: the collapse"),
    "fwd2x": ArmSpec("fwd2x", ("fwd",), epochs=6.0, matched_to="flip",
                     role="forward only, twice the epochs: not a matter of training longer"),
    "rev": ArmSpec("rev", ("rev",), matched_to="sft",
                   role="reverse only: the reverse ceiling and the kill-gate"),
    "flip": ArmSpec("flip", ("fwd", "rev"), role="forward plus every pair reversed: doubled-data reference"),
    "replay": ArmSpec("replay", ("fwd",), replay_share=0.5, matched_to="mix50",
                      role="forward plus generic instruction data at mix50's replaced share: directional loss vs ordinary forgetting"),
    "mixedtask": ArmSpec("mixedtask", ("fwd",), mixed_task=True, matched_to="sft",
                         role="forward pairs as 20 % of a five-task SFT set: does collapse survive realistic mixtures"),
    "fullft_sft": ArmSpec("fullft_sft", ("fwd",), full_ft=True, matched_to="sft", role="LoRA artifact check"),
    "fullft_mix5": ArmSpec("fullft_mix5", ("fwd",), reverse_fraction=0.05, full_ft=True, matched_to="mix5",
                           role="LoRA artifact check"),
    "cft": ArmSpec("cft", ("fwd",), aux_tasks=("pos", "neg"), matched_to="sft",
                   role="attribution worked example: forward plus equivalence judgements (code)"),
    "unlikelihood": ArmSpec("unlikelihood", ("fwd",), loss="unlikelihood", matched_to="mix5",
                            role="attribution, second case: Zan et al. off-target fix (MT)"),
    "roundtrip": ArmSpec("roundtrip", ("fwd",), loss="roundtrip", matched_to="mix5",
                         role="attribution, third case: round-trip consistency loss (MT or SQL)"),
}

#: The dose ladder. Generated in a loop so five entries cannot disagree on anything but the
#: one number (the same reasoning as obtune's srh/arms.py).
for _pct in (1, 5, 10, 25, 50):
    ARMS[f"mix{_pct}"] = ArmSpec(
        f"mix{_pct}", ("fwd",), reverse_fraction=_pct / 100.0, matched_to="sft",
        role=f"dose rung: {_pct} % of pairs seen in reverse instead of forward, budget-matched to sft",
    )
del _pct

#: Relearning cost from `sft`: k reversed pairs, k in the ladder from the plan (§8.3).
for _k in (10, 50, 200, 1000):
    ARMS[f"relearn{_k}"] = ArmSpec(
        f"relearn{_k}", ("rev",), init_from="sft", relearn_k=_k, matched_to="sft",
        role=f"from sft, train on {_k} reversed pairs: erased vs suppressed",
    )
del _k

#: Arm sets by tier. `GATE` is the September decision gate; `FULL` the small-scale grid;
#: `CORE` the 7-12B cells. base is always evaluated and never trained.
GATE: tuple[str, ...] = ("sft", "mix5", "mix50", "rev")
CORE: tuple[str, ...] = ("sft", "mix5", "mix50", "replay", "rev")
FULL: tuple[str, ...] = ("sft", "fwd2x", "rev", "mix1", "mix5", "mix10", "mix25", "mix50",
                         "flip", "replay", "mixedtask")
RELEARN: tuple[str, ...] = ("relearn10", "relearn50", "relearn200", "relearn1000")
FULLFT: tuple[str, ...] = ("fullft_sft", "fullft_mix5")

TIERS: dict[str, tuple[str, ...]] = {"gate": GATE, "core": CORE, "full": FULL,
                                     "relearn": RELEARN, "fullft": FULLFT}


def resolve(name: str) -> ArmSpec:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]


def units(names: "list[str] | tuple[str, ...]") -> float:
    return sum(resolve(n).cost_units for n in names)
