# Contrastive baseline for directional collapse

**Priority decision, 2026-10-03:** the user placed new domains and contrastive baselines
first, before further legacy experiments, full-FT/d2t implementation gaps, and paper
integration. Implement and validate these pilot tracks next. Existing submitted jobs
continue while this work proceeds. See [TASKS.md](../TASKS.md).

Draft, 2026-10-02; implementation update October 3. Loss, pooling, and fixed-negative
components are implemented and tested. A real-model engineering smoke is submitted;
trainer integration and GPU checks now pass. Amendment 37 freezes exploratory settings
a priori instead of using a validation search. Three eligible full pilot cells are submitted;
the fourth awaits its missing fmt/Gemma gate. Six-pass CE controls are compute proxies,
not exact matches: the accumulation smoke measured 12.53 s versus 7.45 s for CL on Llama.

## October8 execution update

The original fmt/de→en × Llama3B/Gemma4B × seeds17/42/1234 campaign is complete:
12 fresh ten-system evaluations, each with all five registered primary contrasts.
Amendment48 adds fmt/de→en on Llama8B/Gemma12B and units/Gemma12B at the same
three seeds (15 extension cells). Actual two-step larger-model CL/CE trainer checks
block admission until successful; conservative microbatch1 preserves effective64
on H200. Amendment49 adds matched larger-model microbatch4 profiles:future arms use4
only after both CL/CE checks are valid, faster and fit70% of GPU memory. Units negatives use exact physical quantities, with all-anchor split/filter
audits before admission. Seed17 validity, independently of effect sign, admits later
seeds. All base gates, fixed settings, compute caveats and within-pass rules remain.

The design below records the original proposal; Amendment37 fixed settings without
a validation search, and Amendments38/41 register measured throughput refinements.

## Research question

Does aligning representations of paired task inputs/outputs preserve reverse generation beyond
what ordinary reverse-direction SFT achieves? This belongs in the objective-attribution section,
with reverse accuracy as the endpoint. Improved embedding retrieval alone is not recovery.

Use an InfoNCE-style paired representation loss, inspired by
[Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).
[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) is useful background for
multiple-positive variants, but its class-label objective is not the same as this paired
sequence adaptation. Describe this baseline as our adaptation, not a reproduction of either
paper or an established solution to directional collapse.

## Proposed objective

For training pair (a_i,b_i), encode a and b independently with the shared causal LM. Pool
non-padding content-token hidden states at the final layer and L2-normalize them. Freeze the
pooling rule before experiments; use no additional projection head in the primary baseline.

L = L_CE + lambda * (L_NCE(a→b) + L_NCE(b→a)) / 2.

For each anchor, its paired counterpart is positive; other non-equivalent training instances
provide negatives. The InfoNCE term is negative log softmax of cosine similarity divided by
fixed temperature, with the positive among the candidates. Backpropagate through both sides.

Encode isolated sides, not representations of a teacher-forced concatenation: the latter can
copy the counterpart through attention. Use one fixed neutral content wrapper for both sides;
no reverse-generation prompt is supplied by the contrastive term. Keep generation CE and its
masking exactly as in the existing arms.

This still exposes the model to both sides and their correspondence. It is not a baseline with
no inverse information. Reverse CE-token counts do not quantify contrastive supervision, so
report side-token exposure, pair associations, and gradient-bearing objectives separately.

## Arms and comparisons

| Proposed arm | Generative supervision | Additional objective | Purpose |
|---|---|---|---|
| `cl_fwd` | Forward CE | Correct-pair symmetric InfoNCE | Can alignment preserve inverse generation without reverse CE? |
| `cl_mix5` | Existing mix5 CE | Same InfoNCE | Does alignment add value after reverse examples are supplied? |
| `sft_extra_ce` | Forward CE | Extra forward CE sized to approximate CL compute | Is additional optimization/compute enough? |
| `mix5_extra_ce` | Existing mix5 CE | Extra CE at the same direction share | Compute control for cl_mix5 |
| `cl_shuffled` | Forward CE | InfoNCE with fixed shuffled positive mapping | Does correct pair correspondence matter? |

Evaluate base, existing sft/mix5/replay/rev, and the five new arms in ONE fresh pass per cell.
Re-evaluating existing adapters makes within-pass contrasts valid; do not subtract an old
summary from a new contrastive result.

Primary contrasts: cl_fwd−sft, cl_fwd−sft_extra_ce, cl_fwd−cl_shuffled,
cl_mix5−mix5, and cl_mix5−mix5_extra_ce. Also report cl_fwd−mix5 as a practical comparison,
with the explicit caveat that the objectives provide different forms of inverse information.
A shuffled-positive loss can itself harm learning; it is a correspondence ablation, not a
complete replacement for the compute controls.

## Matching and negatives

- Use the same training split, unique pair set, optimizer, LoRA rank, learning-rate schedule,
  seed, effective batch, and optimizer-step count as the corresponding CE arm.
- Profile CL's actual forward/backward cost. Match extra-CE controls as closely as practical
  using extra loss evaluations within each optimizer step; record residual FLOP, token, and
  GPU-time differences. Do not claim exact compute equivalence from equal steps alone.
- Fix the negative count and seeded negative schedule independently of GPU microbatch size.
  Start with four negatives per anchor; gradient accumulation does not create an in-batch
  contrastive batch of 64. All negatives come from training, never validation or evaluation.
- Filter equivalent/duplicate content before sampling. For fmt, use parsed canonical document
  identity. In MT, text deduplication cannot prove semantic inequivalence; record that false
  negatives remain a limitation and audit a sample before launching.
- Avoid hard-negative mining in the first pilot: it adds an objective-selection confound.
- Candidate hyperparameters: lambda {0.1,1.0}, temperature {0.05,0.1}. Select one configuration
  on training/validation only using a preregistered forward/reverse validation rule and equal
  search budgets for relevant controls; never select from final evaluation results.

## Staged execution

1. Preregister loss, pooling, negative sampling, hyperparameter selection, comparisons,
   promotion rule, and compute-matching tolerances before any tuned result is inspected.
2. Add arm specs, loss trainer, auxiliary collator fields, configs, and exposure accounting.
   Auxiliary fields must survive TRL column removal; assert a nonzero contrastive gradient
   on the FIRST training batch. The earlier attribution arms silently lost auxiliary fields.
3. Test positive indexing, duplicate masking, padding invariance, independently encoded sides,
   gradients through both encodings, fixed negative schedules across microbatch refits, and
   lambda=0 equivalence to the corresponding CE loss. Run a GPU smoke test.
4. Pilot fmt and de→en on Llama 3B and Gemma 4B, seed 17: four domain/model cells and five new
   arms each, at most **20 pilot training runs**, plus separately budgeted validation search.
   Existing gates must still pass. Use H100/H200 and profile the first CL run before estimating
   cost or walltime; the extra encodings and hidden states can substantially increase memory.
5. Replicate valid implementations at seeds 42/1234, including null effects. Add en→de as a
   boundary-condition cell, then code or new domains once their negative filters and gates are
   validated. Promotion must not depend only on observing a beneficial effect.
6. Generate paired bootstrap contrasts and provenance. Primary endpoint is strict reverse
   generation; also report forward retention, echo/off-target, format failure, general-ability
   probes, compute, and auxiliary exposure. Apply the existing determinism floor and specify
   correction/reporting for the family of primary comparisons before results.

Interpretation: reverse recovery beyond matched CE controls supports an objective-specific
contribution. A retrieval gain without generation recovery does not. Similarity to mix5 is
not equivalence unless the registered interval/margin supports that claim.

## Checklist

- [x] Preregister the baseline and contrasts.
- [x] Implement loss/collator/accounting and fixed-negative construction.
- [x] Validate real gradients, loss behavior, semantic filters, and compute controls.
- [x] Run GPU smoke/profile and freeze preregistered settings without test-set selection.
- [x] Run the four-cell pilot with all arms evaluated together.
- [ ] Replicate and extend valid runs, including null findings.
- [ ] Integrate objective-attribution contrasts into the paper.
