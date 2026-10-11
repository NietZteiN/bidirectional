> Current scope, October 10: Amendment 56 pauses all experiments and automatic admission.
> Manuscript completion from existing evidence is the priority. All 32 experiment jobs are
> held; the controller is stopped. See [pause plan](../docs/EXPERIMENT_PAUSE.md).
> Amendment 55's resumed campaign and the completed sprint remain historical records.

# Workshop framing and the current extension — October 10, 2026

The framing source is our local non-archival ATTRIB/NeurIPS workshop draft,
[“Attributing Gains in Paired Training: Objective or Data Direction?”](../papers/theflipflip_workshop2026.tex)
(the accompanying [PDF](../papers/theflipflip_workshop2026.pdf) retains the earlier
“The Free Flip” title). Treat it as a workshop draft, not a
verified acceptance or a newly established result. Its introduction and decomposition
sections supply the conceptual starting point: an auxiliary method can change both the
objective and directional exposure; matching total instances/tokens/compute does not
resolve that confound. The practical direction-matched control is also a remedy.

The current paper extends this starting point along three questions:

1. **Across fields:** show the same replacement intervention on code, formatting,
   translation, SQL/semantic parsing and algebra, with all task orientations and both
   directions visible. Use the 1% formatting/code examples plus source-generated
   translation, SQL and algebra rates; retain opposite-orientation adverse results and
   the replicated units null. Task families are not independent replicas of one effect.
2. **How much input:** establish the full dose/forward-cost curve, rather than carry over
   the workshop's half-reversal result or imply that 1% works uniformly.
3. **Why it can help:** write the forward/backward conditional CE objective explicitly.
   The hypothesis is that small backward supervision maintains task-conditioned output
   selection. Gold-versus-copy rankings and matched layer removals support this account
   in selected cells; Gemma absolute-NLL and units local-update controls limit competing
   explanations. No universal memory-erasure or persistent-gradient-conflict claim.

CFT denotes the reconstructed contrastive fine-tuning recipe using code-equivalence
judgments, not conflict-aware fine-tuning; distinguish it from paired embedding alignment.

The workshop's numerical decomposition and older model/corpus results are not copied into
the current tables. Its stronger “all recovery belongs to direction,” uniform-free-cost,
disproportionate-general-loss and universal-invertibility claims are not inherited.
Current mixed-direction contrastive wins and task-specific forward costs stay in the
argument. The manuscript uses this framing directly; it does not fabricate a bibliographic
entry or public publication status for the local anonymous workshop draft. All new prose
rates come from `MAIN_RESULTS.json` and the existing audited campaigns, with within-pass
comparisons only. The experiment queue is paused under Amendment 56.

# Current paper story — October 10, 2026

Forward-only SFT can improve the trained task while damaging its backward counterpart.
The practical intervention is to reverse a small share of existing pairs and keep ordinary
cross-entropy. Lead with the fixed-seed formatting and code examples: 1% reversal gives
substantial backward recovery with little forward change. Use source-generated numbers;
do not claim a universal 1% optimum or universal superiority over auxiliary losses.

The main presentation now has two percentage tables:

- Table 1: all eleven Llama3B seed-17 task orientations, forward/backward rows, base,
  SFT, every reversal dose and every registered main-grid control. Blue identifies low
  doses; red identifies SFT backward rates at or below half of the same-pass base.
- Table 2: every eligible original seed-17 contrastive task/model cell and every audited
  seed-17 auxiliary-objective cell, with both directions and fresh within-pass baselines.
  Preserve mixed-direction contrastive gains, round-trip tradeoffs and the units null.

Highlighting is descriptive. Missing dose rungs are dashes. Model/seed coverage and paired
intervals remain in the appendix and source audit. Memorization or memory erasure is not
established by failed backward generation. Mechanism evidence motivates selective
interference in one cell; it is supporting diagnosis, not the central practical claim.
[MAIN_RESULTS.json](MAIN_RESULTS.json) records every displayed campaign and source hash;
`scripts/117_main_results.py` regenerates both main tables before the paper builds.
The experiment campaign is paused under Amendment 56. Earlier sprint scope below is
historical; the current user-requested pause governs experimental work.

# Active drafting scope — October10, 2026

Use the completed evidence to tell the story: one-way fine-tuning can damage an existing
reverse capability; ordinary reversed-pair SFT is the essential controlled baseline;
objective-specific gains must survive directional/exposure controls; failed gates, units
null results and recipe/forward-cost sensitivity limit generality. The paper already has
substantial seed coverage. No new seed training is required in this sprint.

Amendment54 keeps five fixed seed17 explanation jobs and the two revised Gemma smokes.
The selected jobs test likelihood/gradient geometry in two domains and two families,
retain the units null control, and test all formatting layer bands against matched
controls. Describe new findings as exploratory; preserve all outcomes.19 registered
mechanism production jobs and the unfinished repaired Gemma-format CL campaign are
deferred. [Sprint plan](../docs/PAPER_SPRINT_PLAN.md) records the24-hour target.

All five diagnostics are now complete and proof-valid. The draft reports gold/copy
rankings in both directions, both length conventions, the Gemma absolute-likelihood
counterexample, the units null control, gradient limits and every layer-control contrast.
Middle-band formatting recovery supports depth selectivity in one cell. Local reverse
updates also improve the null cell, so neither local improvement nor final gradient
geometry identifies a universal cause. Generated tables and the full evidence bundle
supply the measurements; no favorable outcome is excluded.

The earlier argument and registration history follow; future-tense replication plans
below are superseded by this scope decision.

# Current story — October 9, 2026

The manuscript now follows this spine:

1. One-way fine-tuning can damage an existing reverse capability that forward-only evaluation misses.
2. Matched directional mixtures and controls test whether ordinary SFT preserves it economically.
3. Original Python/C++, logic, and units gates failed for both small models; retain this boundary evidence. All six explicit-output diagnostics also failed. Gemma12B passes the original units gate and completes replicated null-collapse campaigns:ordinary SFT improves both directions. Replicate by validity, not by effect sign. Llama8B also passes Python/C++ after a technical retry and uses the registered same-pass pilot protocol. Other larger-model gates retain their recorded verdicts; they reuse the original semantic corpora.
4. Paired contrastive baselines test gains beyond reverse exposure and extra compute; all twelve original model/task/seed cells are complete. Fifteen scale/numerical-domain extension cells are separately registered and automatically admitted after engineering checks. Neither a general benefit nor equivalence is established.
5. Recovery instruments test accessibility without presupposing universal suppression.
6. The practical contribution is evaluation in both directions and direction/compute-accounted attribution.

This replaces the stronger historical spine below. Quantitative statements are integrated
through the source-backed generators. The snapshot deduplicates repeated main evaluation campaigns by
model/task/training seed, using the latest valid completion timestamp and retaining earlier
passes in the audit. These observations are not independent domains. Blocked invertibility
experiments do not support a universal repair bound.
The title is now “Directional Collapse in Fine-Tuning: Preserving the Way Back.”

October 8 additions, completed in the current manuscript: the main preservation section reports a dose/forward/reverse/
compute figure. [The appendix](planned_evaluations.tex) reports audited general-ability
controls, independent real-data transfer, frozen prompt/recipe robustness, and an anonymous
reproducibility package, with completed interpretation and an explicit reporting audit.
These protocols are now registered as Amendment 50 and implemented in the autonomous
smoke-gated queue; registration is distinct from completed measurements. Existing
new-domain/contrastive jobs retain first priority. Keep the replicated units null-collapse
case in the main synthesis. Credit prior directional degradation and reverse training; the
proposed contribution is the joint preservation comparison and controlled attribution.
Disproportionate-loss, compute-saving, generality and mechanism claims each need their own
evidence; the historical stronger claims below remain superseded.

October8 control integration: both generated contrastive tables now show all five primary
comparisons per completed cell, including extra-CE proxies and shuffled pairs. Mixed-direction
alignment sometimes exceeds the extra-CE proxy on format conversion; effects differ across
models/seeds. Forward-only alignment has no consistent control-relative benefit, and empirical
zero intervals in translation floor cells cannot establish equivalence. These are interim
within-cell observations, with forward retention and replication required for interpretation.

October9 completed robustness integration: all prompt/template and recipe campaigns are
complete. The appendix reports source-generated ranges across seeds, with both directions
and matched mixtures retained. Translation and format show directional loss under the frozen
prompt variants. Units remains a null-collapse case under the original recipe, but the higher
learning-rate recipe includes reverse loss and improvement across seeds; keep that recipe
qualification in the story. Python/C++ has large seed-dependent forward mixture costs, so
preservation cannot be described as uniformly inexpensive. All OPUS campaigns fail the
unchanged echo gate; they establish a criterion boundary and supply no eligible tuned
transfer comparison. All nine general-ability campaigns are complete; their endpoint-specific
findings are integrated. Reverse-task recovery does not establish general-ability preservation.

October9 mechanism review: we have stronger evidence about recoverability than about why
a small reverse dose preserves it. Existing prompt recovery, model/seed-matched relearning,
adapter scaling and spectral interventions disagree. The model/seed-matched never-had
control is an invented format task, so cross-task difficulty limits the learning-curve inference.
Do not treat the aggregate report's votes as a causal finding or pool seeds that did not collapse.
The source-signed exploratory lead and its limitations are retained in
[MECHANISM_REVIEW.json](MECHANISM_REVIEW.json).

The format/Llama3B layer-ablation diagnostic is a useful lead: removing a middle-to-late
LoRA band restores some reverse success while retaining forward performance. Its source is
`results/mech/layer_ablate/fmt__llama32-3b__sft_s17/ablation.json`, with successful status
`runs/status/m4_ablate_fmt_llama32-3b.425259.json`. This is one200-pair exploratory diagnostic
with aggregate metrics; uncertainty and controls must be rerun with per-instance trials.
The direction probe remains decodable, but paired directions also differ in input language/
format; high probe accuracy alone does not show that the instruction is being used. Identical
first tokens do not identify an output mode. The same-input instruction-sensitivity diagnostic
also creates inputs invalid for one instruction and does not uniformly support ignoring direction.

Working hypothesis: forward-only updates can bias task-conditioned output selection and
interfere with reverse generation; a small dose of correct reversed pairs may maintain that
conditioning. This is a hypothesis, not an established internal mechanism. LoRA preserves
base weights by construction, so recovering their behavior alone is weak evidence of retained
knowledge; full-weight checks matter. The replicated units null and recipe sensitivity limit
any claim of inevitable collapse or a universal suppression mechanism.

The next explanatory checks should distinguish hypotheses rather than add more recovery
instruments: (1) correct reverse-answer likelihood/ranking versus free generation, with
copy/wrong-format candidates and matched base/SFT/replay/mix/CL controls; (2) fixed layer
interventions against equal-update-norm global scaling and random removals, reporting both
directions and every intervention; (3) forward/reverse gradient conflict and small controlled
updates on calibration data, evaluated on disjoint held-out pairs; (4) correctly paired reverse
supervision versus same-exposure shuffled targets if the existing objective controls leave
correspondence unresolved. Keep an explicit null cell and training-seed replications. Probes
need pair-grouped splits and input-only controls before instruction-routing claims. New runs
need a frozen separate exploratory protocol, tested workers and successful engineering proofs.

October7 repair validation: the first full-weight projection repair completed and its saved
weights passed integrity checks. Its fresh paired generation evaluation is complete, improving both forward and reverse
success relative to the original full-SFT checkpoint. This is one fixed-threshold exploratory
checkpoint; no universal repair or validated IID-noise claim follows.

---

# The argument, and why it runs in this order

*Historical draft 2026-09-10, written before experiments. This file is the spine — what the paper
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
sequence tokens. The strongest case is **RevThink** (Chen et al., NAACL 2025): its loss is a sum
of three cross-entropy terms over a union of formats, which *is* SFT on that union, and none of
its baselines holds data direction fixed. That is not an error on their part --- it is what the
general problem looks like when it is structural rather than accidental, which is exactly why
the test belongs in §1. **Tone is diagnostic, not prosecutorial** — the workshop paper's rule, and the
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


Amendment52 now freezes the explanatory panel in `configs/mechanism_panel.json` and
`docs/MECHANISM_PLAN.md`: formatting/Llama3B, translation/Llama3B, translation/Gemma4B,
and the units/Gemma12 reverse-null control, each at three training seeds. Six model/mode
GPU smokes precede 24 production jobs. Native completion-only likelihood covers eight
checkpoint arms. Four fixed depth-band removals are compared with global scaling and
two random controls matched on retained bf16 effective delta-W norm; edit norms are
reported separately. Validation gradients motivate equal-parameter-norm ephemeral
steps tested on disjoint held-out pairs. These balanced local steps do not simulate
the original mix5 budget. All controls, bands, seeds and adverse outcomes are retained.

`paper/MECHANISM_DIAGNOSTICS.json` automatically indexes only proof-valid production
evidence and its source hashes. Likelihood and generation use separate engines; no
new mechanism finding is claimed merely because the jobs are registered or submitted.


Amendment53 repairs the Gemma HF diagnostic after strict autograd exposed disconnected
vision factors in the legacy adapters. All42 fixed-panel Gemma checkpoints have exactly
zero vision B factors, so the existing layer delta-norm controls remain valid. Local
gradients and equal-norm updates now use text LoRA factors only, with strict checks
retained for disconnected text parameters. Two revised GPU smokes precede the six Gemma
HF production slots; both revised smokes and the selected translation/unit HF diagnostics
are now proof-valid. Other registered diagnostics remain unmeasured. Preserve the failed original
attempts and do not interpret an implementation failure as a scientific null.
