# Pre-registration

*Committed 2026-09-10, before any adapter in this project was trained.*

The workshop paper's discipline was that thresholds and contrasts are fixed before results are
seen. This file is that commitment for the ACL paper. It is committed to git, so its date is
checkable, and it is written in the imperative because it is a constraint rather than a plan.

Anything below may be changed **only** by adding a dated amendment at the bottom that says what
changed and why. Nothing is edited in place.

---

## 1. Primary contrasts

Every headline claim is one of these differences, reported with a cluster-bootstrap CI over
pairs, computed within a single evaluation pass:

| contrast | answers |
|---|---|
| `sft − base` on reverse | does forward-only training destroy the inverse capability? |
| `sft − base` on general probes | is the loss disproportionate to general degradation? |
| `mix5 − sft` on reverse | does a small reverse dose prevent it? |
| `replay − sft` on reverse | is it directional, or ordinary forgetting? |
| `mix50 − replay` on reverse | what does direction specifically buy over any equal substitution? |
| `mix50 − flip` on reverse | is replacing forward data as good as doubling it? |
| `mix5 − sft` on forward | is the cure free on the trained direction? |
| `<objective> − <matched mix>` | does a direction-matched baseline account for the method's gain? |

## 2. Thresholds

τ is read from the **base** model's own score distribution on the eval set, at the 25th
percentile of scorable generations, before any fine-tuned model is scored. It is written into
the domain config with the model, date and quantile that produced it
(`scripts/15_base_gate.py --write`). Domains whose criterion is exact — `fmt`, `fmt_novel`,
`code` forward, `sql` forward, `exec` — have no τ.

A missing threshold raises. It does not default.

## 3. The decision gate

**Cells:** `mt_en-de`, `mt_de-en`, `sql` (the NLP test), and `code` (the known-positive
control), on `llama32-3b`, seed 17, arms `sft, mix5, mix50, rev`.

**Pass if,** in at least one NLP cell:
1. `sft − base` on strict reverse is at most **−50 % relative**, and
2. IFEval `sft − base` is better than **−5 points** (so the loss is not general
   instruction-following collapse), and
3. `rev` strict reverse is well above zero in that same cell — the kill-gate. If `rev` ≈ 0 the
   reverse direction is not learnable on that corpus at that scale and no null is
   interpretable.

**If it fails**, the paper becomes a boundary-conditions paper — when does one-way fine-tuning
collapse the inverse, and why does code differ — and the plan is reset before more compute is
spent. That outcome is stated here so it cannot later be reframed as the intended result.

## 4. Predictions

Recorded so they can be wrong.

- **RQ1.** Collapse in most domains; reverse success falls to near zero while general probes
  fall a few points.
- **RQ2.** The MT asymmetry replicates: collapse for X→en training, not for en→X (Zhu et al.
  2024). Collapse survives a mixed-task set where the paired task is 20 % of training.
- **RQ3.** The recoverable ceiling under `mix50` tracks the share of information the transform
  preserves. On the synthetic ladder it falls monotonically from `fmt` to `fmt_lossy100`; in
  `code`, structural transforms (S1, S2) recover and renaming transforms (L1b, L1r, L2) sit at
  a floor no reverse share moves.
- **RQ4.** The knee is below 10 % reverse share in every domain. Forward accuracy is unchanged.
  General-ability cost is a property of fine-tuning, not of direction — `mix50 − sft` on the
  probes spans zero.
- **RQ5.** Suppression, not erasure: a non-zero recovered fraction under prompting, reverse
  collapsing at small α while forward still climbs, and relearning from `sft` outrunning the
  never-had control (`fmt_novel`).
- **RQ6.** A direction-matched SFT baseline accounts for all three objectives' gains.

## 5. Reporting rules

- One evaluation pass per table; contrasts only within a pass. Never quote a cross-pass
  difference finer than the measured determinism floor (0.5 pp until
  `scripts/30_determinism_floor.py` says otherwise).
- Both directions **and** general ability are reported for every arm in every domain, so the
  disproportion is visible per domain rather than argued once.
- Echo and off-target rates are reported beside every strict rate, never folded into it.
- Continuous metrics (COMET, chrF++, triple F1) are reported beside strict rates, because a
  strict rate alone cannot separate "slightly worse" from "collapsed".
- Every criterion whose ceiling is a frozen model — `sql` reverse, `d2t` forward — reports that
  model's own accuracy on the references beside it.
- Seeds {17, 42, 1234} at small scale, {17, 42} at 8-12B. Per-arm intervals are over pairs, not
  over seeds; seed variation is reported as its own band.

## 6. What would falsify the headline claim

- `rev` near zero wherever `sft` collapses — the capability was never learnable there, so
  nothing was destroyed.
- `replay − sft` matching `mix5 − sft` — the effect is ordinary forgetting and direction is not
  the variable.
- General probes falling as much as reverse success — the loss is not disproportionate, and
  "directional collapse" is the wrong name for it.
- A knee that is not below 25 % in any domain — the cure is not cheap and the prescription
  does not follow.

---

## Amendments

*(none)*
