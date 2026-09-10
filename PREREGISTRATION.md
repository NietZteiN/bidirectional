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

### Amendment 1 — 2026-09-10, before any adapter was trained

Six changes, all made while the file could still be changed honestly. Nothing here was
prompted by a result; no model in this project has been fine-tuned yet.

**1. The RQ3 ladder's rungs were mis-specified, and are redefined.**

The rungs were named by leaf-drop share: {0, 10, 50, 100} %. But the criterion is exact
structural equality, so an instance's inverse is determined only if **no** leaf was dropped —
which makes determinability `(1 − drop)^leaves`, not `1 − drop`. Measured against this
generator's own distribution (mean 11.7 leaves per document, n = 20,000), those rungs give
instance-level determinability of **100 / 34.5 / 1.3 / 0 %**: three of four sit at the floor.
RQ3 claims the recoverable ceiling *tracks* invertibility, and tracking cannot be tested with
the points bunched at zero.

Rungs are now named by the quantity RQ3 is actually about — the share of instances whose
inverse is determined — and the drop shares were solved numerically to hit it:

| cell | drop share | instances determined |
|---|---|---|
| `fmt` | 0 | 100 % |
| `fmt_det75` | 0.0255 | 74.6 % |
| `fmt_det50` | 0.0636 | 48.0 % |
| `fmt_det25` | 0.1355 | 22.6 % |
| `fmt_det00` | 1.0 | 0 % |

The ladder cells also drop the `mdtable-csv` subtask, which has no leaf values to remove and
was therefore always determinable — leaving it in diluted every rung to `1/3 + 2/3 × target`
and made the rung's own name false.

**2. Two metrics are declared for the ladder, and one of them is new.**

`leaf_recall` — the share of the original document's leaves the output reproduces, by key path
— is declared now as a **secondary** metric for the `fmt*` cells. Exact match is all-or-nothing
per instance, so above a low drop rate almost every instance fails and the rungs stop being
distinguishable; leaf recall keeps measuring past that point. `strict` remains primary.
`strict_determinable_only` — the strict rate restricted to instances whose inverse is
determined — is also declared: mixing determined and undetermined instances into one rate hides
exactly the floor RQ3 predicts.

**3. "The knee is below 10 %" is given an operational definition.**

As written it was unfalsifiable. The knee is now: **the smallest dose rung whose reverse strict
rate reaches at least 90 % of `mix50`'s reverse strict rate in that cell.** The prediction is
that this rung is `mix5` or lower in most domains — stated as *most*, not *every*, because one
domain supported it in the workshop paper and generalising from one is what this paper exists
to stop doing.

**4. The gate's relative threshold gets the precondition it always relied on.**

`sft − base ≤ −50 % relative` is only interpretable when the base rate is far enough from zero
for a halving to mean anything. That precondition already exists in the code —
`15_base_gate.py` refuses a (model, domain) pair whose base rate is below 10 % — but it was not
written here. **A cell counts toward the gate only if the untouched model's reverse strict rate
is at least 10 %.** A cell below that is reported as uninformative rather than as evidence
either way.

**5. Multiple comparisons.**

Each primary contrast is a separate pre-registered test within its own domain, and no
correction is applied across domains. In exchange: **every domain that was run is reported,
including the ones that show nothing.** The count of domains and arms run appears in the
paper's setup section, so a reader can apply their own correction. No contrast is promoted to
the abstract on the strength of being the largest.

**6. Stopping rule.**

Seeds are fixed in advance: {17, 42, 1234} at small scale, {17, 42} at 8–12B. Seeds are not
added after seeing results. If a cell looks unstable and more seeds are run anyway, they are
reported as a separate, labelled robustness check, never pooled into the headline interval.

Also recorded, since it belongs with the predictions in §4: **`mix50 − flip` on reverse is
predicted to span zero** — replacing forward data with its reversal is expected to be as good
as doubling the data, which is what makes "free" the right word.
