# The argument, and why it runs in this order

*Draft 2026-09-10. **No experiment has been run.** This file is the spine — what the paper
claims, in what order, why that order, what each section has to establish, and what would break
it. Numbers live in [`main.tex`](main.tex) as `\NUM{}` placeholders that render visibly unfilled.*

---

## The one-sentence version

Many NLP tasks come in pairs that are the same data read two ways; fine-tuning on one direction
destroys the model's pre-existing ability in the other; the loss is far larger than general
benchmarks register; and replacing a few percent of the forward pairs with their own reversal
prevents it at no cost in data, tokens or compute.

## What is and is not new

**Not new.** That forward-only fine-tuning degrades a pre-existing inverse capability. Nikiema
et al. (2025) name it *cognitive specialization* and demonstrate it for code obfuscation. Zhu
et al. (2024) report the same shape in machine translation two years earlier, as an
English-on-target moderator. **The paper must say this in the first paragraph.** A framing that
presents the phenomenon as newly discovered will be caught, and both groups may review it.

**New.** Five things, in the order the paper argues them:

1. It is **general** — five paired domains, three model lineages, three seeds — and the loss is
   **disproportionate**: the inverse goes to near zero while general probes move a few points.
2. **Invertibility bounds** what a reverse dose can restore, and there are *two* kinds of
   hardness that the word "invertibility" runs together.
3. The **cure is free in the strict sense**: matched on instances, sequence tokens *and*
   optimizer steps simultaneously.
4. The capability is **suppressed, not erased**, on four instruments that can disagree.
5. Therefore: any method credited with restoring an inverse capability on paired data owes a
   **direction-matched baseline**.

---

## Why this order

The sections are not a list of experiments; each one is load-bearing for the next.

**Phenomenon before cure.** A repair claim is meaningless until the loss is established on the
same instruments that will later show the repair. §4 exists to make §5 legible.

**Cure before mechanism.** This is the ordering choice worth defending. The dose result is
itself evidence about mechanism: if 5 % of pairs seen backwards restores most of the capability,
whatever training removed cannot have removed much. A reader who has seen §5 arrives at §7
already believing suppression is likely, and §7's job becomes confirming it with instruments
that could have said otherwise — rather than persuading from scratch. Putting mechanism first
would waste that.

**Mechanism before attribution.** §8 is a corollary, not a finding: once direction is
established as the operative variable, any method that adds directional exposure inherits the
credit. It is half a page because it is an implication, and because the workshop paper already
carries the worked case.

**Bound between cure and mechanism.** §6 sits where it does because it is the honest limit on
§5's prescription, and a reader who has just been told the cure is cheap should immediately be
told where it does not work. Deferring it to Limitations would be salesmanship.

---

## Section by section: what each must establish

### §1 Introduction
Open on the structure that makes the phenomenon possible: **paired tasks, where the reverse
training set is obtained by swapping which half of each pair is the input.** That is the hook,
because it makes the cure free and the failure absurd — the data to prevent it was already in
hand.

Then: the failure, the disproportion, the cure, the bound, the mechanism, the consequence. Cite
Nikiema et al. and Zhu et al. here, not in §9.

The contribution paragraph should be a **test a reader can apply**, not a list — the workshop
paper's v4 lesson. Ours: *for any method that adds instances to paired training data, train the
arm that supplies the same directional content and none of the method.* It earns its place
because the ablation arm is also the remedy, so running it pays either way.

### §2 Directional collapse, defined
One term, defined once. A paired task; forward and reverse; the strict criterion; **echo and
off-target as first-class failure modes, never folded into success**. This section is short and
exists so that §4–§8 never have to re-explain what is being measured.

Non-obvious point to make here: echo is not hygiene. In JSON→YAML a model that copies its input
*parses and compares structurally equal*. Without the echo guard the task is not a task.

### §3 Setup
Domains, models, arms, budget accounting. The **budget claim is the one to make precisely**,
because "free" is the word the paper will be attacked on:

> Every `mix` arm replaces forward pairs with their own reversal, partitioned by pair identity.
> Instance count, sequence tokens and optimizer steps are matched to the forward-only baseline
> at every dose. Golovneva et al. (2024) distinguish data-matched from compute-matched reverse
> training; because our reversal replaces rather than adds, it is matched on both at once.

That sentence is the cleanest available statement of why "free" is literal here, and it is
checkable against the literature.

### §4 Collapse is general and disproportionate
RQ1 and RQ2. Both directions **and** general ability, per domain, so the disproportion is
visible per row rather than asserted once.

Three controls carry this section, and each answers a specific alternative explanation:
- `replay` — is it ordinary forgetting? Same share replaced, generic instruction data instead.
- IFEval — did the model just stop following instructions?
- `rev` — was the direction ever learnable here? If `rev` ≈ 0 the cell is uninterpretable and is
  reported as such rather than as a collapse.

RQ2's moderators: the MT prediction is now **direction-specific and risky** — collapse predicted
where English is on the target side (`mt_de-en`, `mt_zh-en`), little or none where it is not. If
all four MT cells collapse equally, Zhu et al.'s moderator does not survive at our scale and this
becomes a non-replication. That is registered.

### §5 A small reversed dose restores it at no cost
RQ4, and the paper's prescription. The dose ladder across domains, forward accuracy per
condition, general-ability cost.

The knee has an operational definition — the smallest rung reaching 90 % of `mix50`'s reverse
rate — because "the knee is below 10 %" was unfalsifiable as originally written.

**The cost sentence must be scoped every time it appears.** Fine-tuning costs general ability at
*any* direction mix; the dose is free relative to the forward-only baseline, not relative to the
untouched model. The workshop paper lost a claim to a contaminated probe here and the scoping is
what survived it.

### §6 Invertibility bounds what can be restored
RQ3, and the section where the paper says something the prior work cannot.

**"Invertibility" is two properties, and running them together is a mistake the literature
makes.** An inverse can be *undetermined* — the information is not in the input, and no amount
of reverse data can help. Or it can be *determined but hard to find* — the information is there
and the search is the obstacle. These predict different things about a cure, and the paper tests
them separately:

- the synthetic ladder varies determinability directly, in four rungs spaced by the share of
  instances whose inverse is determined;
- `automata` holds determinability at 1.0 — every instance has a predecessor by construction —
  and varies only search difficulty;
- within `code`, structural transforms preserve identifiers while renaming transforms destroy
  them, which is the same distinction in a natural domain.

The prediction: **a reverse dose buys recovery against computational hardness but not against
missing information.** Dirasym (2025) gives the analytic anchor — at branching factor K=1 their
forward and reverse converge identically, which is our lossless baseline derived on a
semantics-free task.

### §7 Suppressed, not erased
RQ5, the longest section, and four instruments that are allowed to disagree.

1. **Elicitation** — a capability that is gone cannot be prompted back; one that is intact does
   not need to be. A non-zero recovered fraction is the first suppression signal.
2. **Adapter scaling** — reverse collapsing at small α while forward still climbs makes the
   collapse a cheap direction in weight space.
3. **Relearning cost** — recovery from `sft` that outruns a **never-had control** is latent
   knowledge. This is also where the reversal-curse contrast becomes experimental: that
   literature describes capabilities never acquired, and `fmt_novel` is that case, constructed.
4. **Spectral repair** — a closed-form filter on the update, no data and no retraining. If
   reverse returns while forward holds, the capability was masked by a removable residue.

**They are reported as votes, never averaged.** They measure different things and a disagreement
is a finding. The honest caveat rides with instrument 4: a LoRA delta is already low-rank, so
there is no noise bulk to remove, and a null there is weak evidence rather than evidence for
erasure.

### §8 Consequences for attribution
Half a page. Three objectives, each against SFT on the same directional mixture at matched
sequence tokens. **Tone is diagnostic, not prosecutorial** — the workshop paper's rule, and the
authors of the method may review this. The precise claim about Nikiema et al. is that they
*declare* a bidirectional baseline and report no number for it, which is verifiable in their §6
and is not the same as "they never ran it".

### §9 Related work
Four groups, each with an explicit differentiation sentence:
- **reversal curse** — capability never acquired, versus one acquired and lost; the relearning
  contrast is what makes that experimental rather than rhetorical;
- **forgetting** — diffuse general loss, versus targeted and total on the inverse; `replay`
  separates them;
- **fine-tuning as wrapper/suppression** — same lens, applied to a capability rather than an
  alignment behaviour;
- **direction in MT** — treated there as an off-target problem specific to translation; we show
  it is one instance of a cross-domain phenomenon with the same cure.

### §10 Conclusion
The general test, restated. Not a summary of results.

---

## What would break the paper

Registered in [`../PREREGISTRATION.md`](../PREREGISTRATION.md) §6, and worth keeping in view
while drafting because each has a section that would have to change:

| if | then |
|---|---|
| `rev` ≈ 0 wherever `sft` collapses | the capability was never learnable there; nothing was destroyed; §4 loses its cells |
| `replay − sft` ≈ `mix5 − sft` | it is ordinary forgetting and direction is not the variable; §4's control kills §5 |
| general probes fall as much as the inverse | the loss is not disproportionate and "collapse" is the wrong name; §1's framing goes |
| no knee below 25 % in any domain | the cure is not cheap and the prescription does not follow; §5 becomes a negative result |
| all four MT cells behave alike | Zhu et al.'s moderator does not replicate; §4's RQ2 becomes a non-replication |
| `automata`'s directional gap is small | the two kinds of hardness are not separable by this design; §6's refinement fails |

**And the gate.** If no NLP domain collapses, the paper becomes a boundary-conditions paper —
*when does one-way fine-tuning collapse the inverse, and why does code differ* — with a different
spine: §4 and §6 merge into a moderator analysis, §5 becomes conditional, §7 survives unchanged
because it is about the code domain where the effect is known to exist. That is publishable and
is not a fallback to be improvised after the fact.

---

## Drafting rules carried over from the workshop paper

- Every strong claim carries its CI **in the sentence**, not a footnote.
- "Free" is scoped to training budget everywhere it appears.
- One evaluation pass per table; never quote a cross-pass difference finer than the measured
  determinism floor.
- No number is hand-typed. Tables come from `scripts/51_tables.py`, figures from
  `scripts/52_figs.py`, and `paper/NUMBERS.md` maps each to the runs that produced it.
- Do not claim every auxiliary-task result on paired data is a directional artifact. We measured
  some; the boundary is what a reviewer will press on.
