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


### Amendment 2 — 2026-09-10, before any adapter was trained

**The `replay` arm did not satisfy its own specification, and is corrected.**

`replay` is specified in §1 as replacing a share of forward pairs with generic instruction data
**at matched budget**, which is what makes `replay − sft` a read on ordinary forgetting and
`mix50 − replay` a read on what direction specifically buys. It was matched on row count only.
Measured on 2026-09-10, before any training: an unmatched tulu-3 sample ran **1.24×** the
rendered character budget of the `mt_en-de` pairs it replaced. Both contrasts would then have
confounded direction with token budget.

Replay rows are now drawn by nearest-length matching against the **rendered** examples they
replace — the rendered length, not the raw pair, because the domain's own instruction wrapper is
most of the sequence for some cells (SQL pairs are 185 raw characters and 988 rendered ones, so
targeting the raw length left that arm at 0.80×). The pool was widened and its length cap raised
to 2,600 characters so candidates exist at every target length.

Realised ratios against `sft`, mean rendered characters per example: `mt_en-de` 1.01,
`sql` 1.00, `fmt` 1.02, `code` 0.98, `d2t` 1.00, `exec` 1.00.

This is a correction to make an arm meet its stated specification, not a change to the design.
The contrasts, thresholds and predictions in §1–§6 are untouched. It is recorded here because
the arm's construction materially changed and the change is only checkable if it is written
down. `tests/test_mixture.py` now asserts every matched arm stays within 10 % of `sft`, and
`scripts/16_audit_criteria.py` re-checks it with the real tokenizer.

### Amendment 3 — 2026-09-10, before any adapter was trained

**`relearn-k` would not have trained, and the guard that should have caught it was blind.**

Two compounding defects, both found by arithmetic rather than by a run, and both fixed before
any adapter exists.

*The arm did not train.* `relearn-k` fine-tunes a collapsed model on as few as **10** reversed
pairs. Under the shared recipe's effective batch of 64, ten rows is fewer than one optimizer
step: measured across the panel, `relearn10` takes **zero** steps on `llama32-3b` and
`gemma3-12b` and one on `olmo2-1b`, and `relearn50` takes two or three. Its adapter would have
been byte-identical to `sft`, and the relearning curve would have read *no recovery at small k*
— which is exactly the signature of erasure. RQ5's central claim would have been decided by
batch arithmetic.

*The guard could not see it.* `bidir.evaluate`'s adapter-effectiveness check compares each
system's generations against `base`. But `relearn*` **starts from `sft`**, so it differs from
`base` no matter what, and an arm that trained not at all passes silently.

Both are corrected. For relearn arms the batch shrinks to fit `k` and the **optimizer-step count
is held constant across k** (40 steps, in `configs/train/_base_lora.yaml`). That also makes the
measurement better than it was specified: relearning cost is now denominated in **data** —
which is the question RQ5 asks — with compute held fixed instead of varying with `k`. If larger
`k` also bought more steps, a rising curve could not separate "more data helps" from "more
training helps". The trainer additionally refuses to save an adapter that took zero steps, and
the effectiveness guard now compares each arm against the adapter it was **initialised from**
rather than always against `base`.

This changes no contrast, threshold or prediction in §1–§6. §4's RQ5 prediction — that
relearning from `sft` outruns the never-had control — is now testable rather than
predetermined.

### Amendment 4 — 2026-09-10, before any adapter was trained

Three domains added, one criterion changed, and three scorer defects fixed. All before any
model in this project has been fine-tuned; none of it is a response to a result.

**1. Three domains are added, and declared here with their predictions.**

Chosen from `docs/TASK_CATALOGUE.md` on the argument in `docs/CANDIDATE_DOMAINS.md`: each closes
a hole in the paper's argument rather than adding breadth.

| cell | catalogue | criterion | prediction |
|---|---|---|---|
| `algebra` | #56 expansion ↔ factorization | symbolic equivalence, and the reverse output must genuinely be factored | collapse in the reverse (factoring) direction; the paper's generality claim needs a formal domain and had none |
| `diacritics` | #66 diacritic restoration | exact match after NFC, reverse must not echo | collapse in the reverse (restoring) direction. Because stripping is a deterministic character map with **nothing to learn**, a collapse here cannot be explained by the forward task consuming capacity — the confound every other domain leaves open |
| `automata` | #30 Game of Life | forward: simulated next state; reverse: **any** grid that steps to the target | collapse in the reverse (predecessor) direction, with a ceiling set by search difficulty rather than by missing information |

**`automata` splits an axis §4's RQ3 prediction ran together.** "Invertibility" is two properties:
whether the inverse is *determined by the input*, and whether it is *findable*. The `fmt_det*`
ladder varies the first — where the answer is no, no reverse dose can help. In `automata` every
instance has a predecessor by construction, so the inverse is fully determined and only the
search is hard (NP-hard in general). The refined prediction: **a reverse dose buys recovery
against computational hardness but not against missing information.** That is a sharper claim
than the original and can fail.

**2. `code`'s primary reverse criterion changes from the published one to execution.**

Measured by feeding the gold source back through the scorer: obtune's published criterion
rejects **about 60 % of perfect answers**, because it requires low CodeBLEU similarity to the
obfuscated program and the identifier-preserving transforms (`S1`, `S2`) leave the original
genuinely similar to its variant. A criterion whose ceiling on correct answers is 0.4 cannot
support a claim of the form "the base scored X and training collapsed it" — the base could never
have reached X.

Primary is now **execution equivalence ∧ not-echo**, which has a measured ceiling of 1.00. The
published criterion is retained and reported as `strict_paper_criterion`, which is what it is
for: comparability with the workshop paper. This is a change to a pre-registered criterion and
is therefore recorded here in full; it was made because the criterion demonstrably cannot award
a correct answer, not because its numbers were inconvenient — no model has produced any.

**3. Three scorer defects, found by the oracle audit and fixed.**

- `code` read `output_canon` from obtune's eval items, where the field is `output_repr`. Every
  case therefore compared the real output against `None` and returned `mismatch`: **the
  known-positive control could not have scored above zero in either direction.**
- `exec` constructed `BatchItem(..., cases=[...])`, but that dataclass takes `program_id` and
  `args_reprs`. It raised `TypeError` on every trial, so the domain could never have run.
- `sql`'s forward criterion needs `sqlparse`, a dependency of the Spider test-suite evaluator
  that was not installed. Added to `env/extras.txt`.

None of these would have crashed the campaign. Each would have produced a clean, plausible
table of zeros or a silently skipped cell.

**4. Budget and grid.** The three domains raise the small tier from 54 to 81 cells and the
training total from ~330 to ~391 GPU-h, within the plan's envelope. `mixedtask`'s five-task
roster is deliberately **not** extended, because changing it would change what `mixedtask − sft`
means between the two halves of the grid.

### Amendment 5 — 2026-09-10, before any adapter was trained

Prompted by a literature check ([`RELATED_WORK.md`](RELATED_WORK.md)). Two additions, both
pre-registered here before the experiments they concern have been run.

**1. A rank sweep on `rev`, and `fullft_*` reread as a capacity control.**

"Directional Optimization Asymmetry in Transformers" (arXiv:2511.19997) reports that **LoRA hits
a sharp capacity wall on high-entropy inverse mappings**. This project is LoRA-based at r=32
throughout, and `rev` is the **kill-gate**: a near-zero `rev` is our licence to call a direction
unlearnable and every other null uninterpretable in that cell. If the wall rather than the model
produced that zero, the licence is void.

So: `rev` is additionally trained at r ∈ {16, 32, 64, 128} in one domain, and the existing
`fullft_*` arms are read as a direct control on this objection rather than only as the
LoRA-forgets-less check they were specified as. **Prediction: `rev` at r=32 is within the seed
band of `rev` at r=128** — i.e. the kill-gate is not capacity-limited. If it is not, every
`rev`-based conclusion is restated with the rank caveat attached, and the kill-gate threshold is
re-derived at the largest rank run.

**2. Instruction sensitivity is measured on the `mix*` arms, not only on `sft`.**

"When Inverse Data Outperforms" (arXiv:2509.13079) reports that naively mixing forward and
reverse data during SFT **weakens the directional distinction** — the model becomes less able to
tell which direction it was asked for. Our `mix*` arms are exactly that mixture, and the paper's
prescription is to use them, so a cost they carry is a cost the prescription carries.

Mechanism experiment 6 (`bidir.mech.sensitivity`) already measures this quantity; it was
specified to run on collapsed models. It now also runs across the dose ladder.
**Prediction: instruction sensitivity falls monotonically as the reverse share rises, and the
fall is small below the knee.** If sensitivity collapses at the doses the paper recommends, the
dose ladder has an upper bound as well as a lower one and the prescription becomes a range rather
than a floor — which is a finding, not a failure, and is registered as such here rather than
discovered later.

### Amendment 6 — 2026-09-10, before any adapter was trained

**Amendment 5's first item was based on an abstract and is withdrawn. The second stands, with a
narrower claim.**

Amendment 5 was written from abstracts. The papers have now been read
([`papers/`](papers/), cited by page in [`RELATED_WORK.md`](RELATED_WORK.md)), and one of the two
additions was aimed at a claim its source does not make.

**Withdrawn: the `rev` rank sweep.** I registered it because arXiv:2511.19997's abstract reports
that "LoRA encounters a sharp capacity wall on high-entropy inverse mappings", which I read as a
threat to the kill-gate. The paper's Table 3 (p. 8) shows inverse excess loss of 5.06 at r=8,
4.85 at r=64 and 4.75 at r=256 — rank buys almost nothing — and forward excess loss of
4.85/1.66/1.60, so the wall is in **both directions**. It is not a directional finding about
LoRA; LoRA simply underperforms scratch and full fine-tuning on that task either way. The task
is also random i.i.d. strings with, in their words, no pattern appearing more often than chance
(p. 2) — pure memorization over 40,000 arbitrary pairs, at GPT-2 Small scale. A low-rank update
is the wrong instrument for a lookup table, which says little about structured inverses like
deobfuscation or factorization.

The registered prediction ("`rev` at r=32 within the seed band of r=128") is therefore withdrawn
rather than left to be quietly not-run. The `fullft_*` arms remain the honest control for "is
this a LoRA artifact", which is a question a reviewer will still ask.

**Gained instead: an analytic prediction for RQ3.** The same paper's branching factor *K* is
invertibility in formal dress — the forward map is deterministic (H = 0), the inverse one-to-many
with entropy floor H(A|B) = log K — and at **K=1 they find forward and reverse converge
identically** once the floor is accounted for (p. 3). That is our `fmt` baseline prediction,
derived analytically on a semantics-free task. **Registered: `fmt` (100 % determinable) shows the
smallest forward-reverse gap of any cell in the grid, and `automata` — determined but
computationally hard — shows a large gap despite a determinability of 1.0.** If `automata`'s gap
is small, the two kinds of hardness are not separable by this design and RQ3's refinement in
Amendment 4 fails.

**Retained, with a narrower claim: instruction sensitivity across the dose ladder.**
arXiv:2509.13079's "directional distinction" turns out to be a log-likelihood margin between
preferred and dispreferred reasoning paths, narrowed to 0.05--0.1 by mixing (p. 5) — not
instruction-following sensitivity, which is what mechanism experiment 6 measures. Their mixture
is also a single 1:1 blend, where our knee is predicted below 10 %. The experiment still runs
across the ladder, but the prediction is downgraded from "sensitivity falls monotonically" to:
**sensitivity at the recommended dose is within the seed band of `sft`**, with any fall
concentrated at 50 % and above. A fall at 5 % would be the surprising result, and is the one
worth having registered.