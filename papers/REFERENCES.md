# `papers/` — the literature this project positions against

Every PDF here has been **read**, not just abstracted, and every claim in
[`../RELATED_WORK.md`](../RELATED_WORK.md) cites a page. The rule is simple: a related-work
claim that cannot be checked against the PDF in this directory is not a claim.

Read one with:

```bash
python scripts/90_paper_text.py papers/<file>.pdf --pages 1-6
python scripts/90_paper_text.py papers/<file>.pdf --grep "LoRA" --context 3
```

| file | what it is | why it is here |
|---|---|---|
| `theflipflip_workshop2026.pdf` (+ `.tex`) | **Our own workshop paper** — the directional-confound result, ATTRIB @ NeurIPS 2026 | the ACL paper extends it; its attribution claim is the RQ6 seed |
| `nikiema2025contrastive.pdf` | Nikiema et al. 2025, [arXiv:2509.05553](https://arxiv.org/abs/2509.05553) | **the closest prior work.** Names the phenomenon ("cognitive specialization") first, in one domain, and proposes CFT |
| `hasanov2026pathnottaken.pdf` | Hasanov et al., ACL 2026, [arXiv:2604.20917](https://arxiv.org/abs/2604.20917) | DexBench: duality of program-execution reasoning. Evaluation only; supports our premise |
| `zhu2024finetuning.pdf` | Zhu et al., EMNLP 2024, [arXiv:2404.14122](https://arxiv.org/abs/2404.14122) | the MT direction asymmetry our `mt_*` cells replicate as the RQ2 moderator |
| `dirasym2025synthetic.pdf` | Directional Optimization Asymmetry, [arXiv:2511.19997](https://arxiv.org/abs/2511.19997) | entropy-controlled synthetic study; its branching factor **K is our RQ3 axis in formal dress** |
| `inversedata2025pitfalls.pdf` | When Inverse Data Outperforms, [arXiv:2509.13079](https://arxiv.org/abs/2509.13079) | mixing forward and reverse data has a cost; bears on the `mix*` arms |
| `latentgen2026illusion.pdf` | Illusion of Latent Generalization, [arXiv:2604.04943](https://arxiv.org/abs/2604.04943) | bidirectional objectives need not build a unified representation; bears on RQ5 |
| `spectral2026unforgetting.pdf` | Spectral Unforgetting, [arXiv:2605.20296](https://arxiv.org/abs/2605.20296) | post-hoc recovery of damaged capability — a strong form of the suppression hypothesis |
| `berglund2024reversal.pdf` | The Reversal Curse, [arXiv:2309.12288](https://arxiv.org/abs/2309.12288) | the lineage we are explicitly **not** in: capability never acquired |
| `golovneva2024reversetraining.pdf` | Reverse Training, [arXiv:2403.13799](https://arxiv.org/abs/2403.13799) | data-level cure for the reversal curse, evaluated data- and compute-matched |

`../../obtune/papers/` holds 28 more on obfuscation, decompilation and code execution, indexed
in its own `REFERENCES.md`. Nothing is duplicated here that is not directly cited.

---

# Read notes

One paragraph per paper: what it actually shows, and what it means for us. Written 2026-09-10
after reading each past the abstract. Page numbers are to the PDFs in this directory.

### `theflipflip_workshop2026.pdf` — ours (ATTRIB @ NeurIPS 2026)
*The Free Flip: Directional Composition Confounds Attribution in Paired Training Data.* The
attribution problem, stated generally: a method that adds instances to paired data also changes
how many examples of each **direction** the model sees, and instance count, token count,
optimizer steps and compute can all be matched exactly while direction exposure differs. Measured
on one case via a 2×2 holding the direction mix fixed: the published contrastive objective
contributes **−0.2 pp [−0.7, +0.3]** against **+30.9** for the direction of the data. The
direction-matched arm also repairs the failure, reaching **32.8 %** reverse accuracy at unchanged
cost. Two stated limits: repair works only where the rewrite is invertible, and fine-tuning costs
~5 points of held-out general ability at *any* direction mix — so "free" is relative to the
forward-only baseline, not to the untouched model. The ACL paper is this generalized.

### `nikiema2025contrastive.pdf` — the closest prior work
Names **cognitive specialization** (Definition 2, p.2): "a learning pathology where training on
*T* creates directional bias, degrading performance on *T*⁻¹". Six models, three obfuscation
techniques, 10,000 Java programs from CodeNet. Proposes CFT (positives, negatives, forward
generation). **Verified fact for our attribution claim:** they *declare* a Bidirectional
Fine-Tuning baseline on p.6 — "CFT effectiveness is assessed through comparison against Standard
Fine-Tuning (SFT) ... and Bidirectional Fine-Tuning (BFT)" — and the string `BFT` appears
**exactly once in the paper, never in a results table**. So "they name the baseline and report no
number for it" is exact; "they never ran it" would be false.

### `chen2025revthink.pdf` — NAACL 2025, the mirror image
RevThink: a teacher generates (question, forward reasoning, backward question, backward
reasoning); the student trains on three tasks. +13.53 % over zero-shot across 12 datasets, and
10 % of the forward reasoning beats 10x more under standard fine-tuning. **Measures FORWARD
accuracy** — adding backward data to improve forward reasoning — where we measure the reverse
direction being destroyed. Their backward data is teacher-generated and costs inference; ours is
a field swap and costs nothing.

The reason it matters for §8: their loss (p. 4) is `L = (1/3n) Σ [ℓ(fwd) + ℓ(bwd question) +
ℓ(bwd reasoning)]` with `ℓ` token cross-entropy — joint next-token CE over a union of three
pools, which is what SFT on the union is. The baselines (SKD, AnsAug) differ in data direction
too, so nothing holds direction fixed. Same structure as CFT, bigger venue, 12 datasets.

### `hasanov2026pathnottaken.pdf` — DexBench, ACL 2026
Evaluation only, no fine-tuning. Forward = predict execution behaviour for an input; backward =
**counterfactual input mutation** toward a target execution path (p.3) — narrower than
CRUXEval-style input prediction, so it is *adjacent to* our `exec` cell rather than overlapping
it. 445 paired instances, 13 models including Gemini 2.5 Flash, GPT-5 Mini, Claude Sonnet 4,
Grok-4 Reasoning, QwQ-32B, Qwen2.5-32B/72B, Llama-3.3-70B. Backward columns are far below forward
throughout, and the weakest models sit at 0.0 on several (p.5). **Supports our premise** that
both directions are real, non-saturated tasks. Code and data public:
`https://github.com/sail-ucf/dexbench` (p.2 fn.2) — evaluating our arms on it is ~1 GPU-hour.

### `zhu2024finetuning.pdf` — EMNLP 2024, and it sharpens RQ2
LLMs translate well after fine-tuning on **as few as 32 parallel sentences**, and single-direction
fine-tuning **elicits other directions** — pretrained models already hold multilingual capability
that tuning surfaces (p.1, p.9). The exception is the finding we care about: *"it is crucial to
pick the right direction — we recommend **not placing English on the target side**"* (p.2),
because English-on-target causes task misinterpretation. **Consequence for our cells:**
`mt_de-en` (forward = de→en, English *target*) is the direction predicted to damage; `mt_en-de`
(forward = en→de) is not. Registered in Amendment 7. It also tempers RQ1 for MT: this domain may
show *less* collapse than code, because the base capability is elicited rather than overwritten.

### `dirasym2025synthetic.pdf` — the one I had backwards
Random i.i.d. 8-char strings, "no token, substring, or structural pattern appears with
higher-than-random frequency" (p.2) — pure memorization of 40,000 arbitrary pairs, GPT-2 Small.
Branching factor *K* controls invertibility: forward deterministic (H = 0), inverse one-to-many
with entropy floor log *K*. **At K = 1 forward and reverse converge identically** once the floor
is accounted for; asymmetry appears only as the inverse becomes many-to-one (p.3). The LoRA
"capacity wall" (Table 3, p.8) is inverse excess loss 5.06/4.85/4.75 at r = 8/64/256 and forward
4.85/1.66/1.60 — **both directions, and rank barely helps**, so it is not a directional finding
about LoRA. Their *K* is our RQ3 axis in formal dress; their framework cannot express `automata`
(determined but hard) because their difficulty is purely entropic.

### `inversedata2025pitfalls.pdf`
`r1k` (1,000 inverted `s1k` reasoning traces) beats forward `s1k` by **1.6–6.8 %**. A **1:1**
mixture narrows the preferred/dispreferred log-likelihood margin to **0.05–0.1** (p.5) — their
"directional distinction" is that margin, not instruction-following sensitivity, and their regime
is a full blend where our knee is predicted below 10 %. Bounds the transfer to our `mix*` arms;
the sensitivity experiment still runs the ladder, with a downgraded prediction.

### `latentgen2026illusion.pdf`
Bidirectional objectives reach non-zero reversal accuracy where plain NTP collapses, but
**"reversal accuracy requires training signal that explicitly makes the source entity a
prediction target"**, and probes are consistent with forward and reverse stored as **distinct
entries** rather than one direction-agnostic representation (p.1). Two consequences: it explains
*why* `rev`/`mix` arms should work (they make the reverse output a prediction target) and
predicts that objectives which do not make it a target should not; and it is the direct precedent
for `bidir.mech.direction_probe`, which uses the same instrument.

### `spectral2026unforgetting.pdf` — a mechanism experiment we do not have
DG-Hard: treat the fine-tuning update Δ = W_ft − W_base as low-rank task signal in an IID-like
noise residual, apply the **Donoho–Gavish hard singular-value threshold** per weight-delta
matrix, keep the structured high-energy part, discard the spectral bulk. **Checkpoint-only,
closed-form, no data and no retraining.** Across 14 (model, task) settings, 13 show at least one
single-benchmark collapse, and DG-Hard achieves the strongest balanced repair; it even restores
safety alignment degraded by benign fine-tuning using no alignment data. Their framing is ours:
*"part of fine-tuning-induced capability loss is not an unavoidable consequence of
specialization, but a removable spectral residue in the weight update itself."* **This is a
direct, cheap test of suppressed-versus-erased** — stronger than α-scaling, which scales the
whole delta uniformly. Proposed as mechanism experiment 7 (Amendment 7). Code:
`https://github.com/BrickleRex/dghard`.

### `golovneva2024reversetraining.pdf`
Reverse training uses all words twice, **doubling tokens**; four reversal types (token, word,
entity-preserving, random-segment). Entity-preserving and random-segment mitigate and sometimes
eliminate the reversal curse. Key methodological point for us: they distinguish **data-matched**
(reverse training wins on standard tasks) from **compute-matched** (reverse training wins far
more on reversal tasks) — and their scheme *adds* tokens. **Our `mix*` arms replace instead, so
they are matched on instances, sequence tokens and optimizer steps simultaneously — a third,
strictly tighter regime than either of theirs.** Worth saying explicitly in the paper; it is the
cleanest one-line statement of why "free" is literal here.

### `berglund2024reversal.pdf` — ICLR 2024, the lineage we are not in
Fine-tune GPT-3 and Llama-1 on fictitious facts in one order ("Daphne Barrington is the director
of ..."), then query both orders. Models answer the trained order and fail the reverse; not
alleviated by data augmentation. Also tested on real celebrities with ChatGPT. The capability was
**never acquired** — which is exactly the contrast our `relearn-k` versus `fmt_novel` comparison
makes experimental rather than rhetorical.


---

## The eight carried over from the plan's bibliography — now read

Pulled and read 2026-09-11. Three of them change how the paper positions itself; the rest are
background and are cited as such.

### `scialom2022rehearsal.pdf` — the closest antecedent to the dose result, and not a threat
Continual-T0 learns 8 new tasks "while maintaining almost 100 % of the initial performance on
all the previous datasets ... obtained by using only **1 % of data for memory buffer**" (p. 2).
A small fraction of the right data does prevent forgetting, and that shape is not ours to claim.

But the regimes differ in a way that **sharpens RQ1 rather than weakening RQ4**. Their buffer
replays *actual examples of the earlier task*; our reverse dose is the *same pairs read the
other way*, constructed for free. Theirs preserves a capability the model was **trained** on;
ours preserves one it had from **pretraining** and was never trained on. Theirs **adds** the
buffer; ours **replaces**, holding budget fixed.

And the sharpening: **our `replay` arm is essentially their rehearsal.** If directional collapse
were ordinary forgetting, Scialom predicts that generic replay at the same share should prevent
it. The registered prediction is that it does not, while an equal share of reversed pairs does.
That contrast is a much better use of this citation than treating it as prior art for "small
doses work".

### `kotha2024forgetting.pdf` — far closer to H1 than "background"
"Language models implicitly infer the task of the prompt and fine-tuning skews this inference
towards tasks in the fine-tuning distribution" (p. 1). They propose **Conjugate Prompting** —
make the task look farther from the fine-tuning distribution while requiring the same capability
— and recover some pretraining capability.

That is our H1 stated in their vocabulary: forward-only training skews task inference so that
the model applies the trained mapping whatever direction is requested. Mechanism experiment 6
(instruction sensitivity) measures exactly their quantity, and experiment 1 (the elicitation
ladder) is a close relative of conjugate prompting. **This should be cited as a precedent for
the hypothesis, not filed under forgetting.** Directional collapse may be the special case where
the skewed "task" is the direction itself.

### `lee2024dpo.pdf` — suppression, mechanistically, in another domain
"Capabilities learned from pre-training are **not removed, but rather bypassed**" — DPO learns
an offset "distributed amongst its layers" to bypass the regions that elicit toxicity (pp. 1–2).
The distributed-across-layers finding bears directly on mechanism experiment 4 (layer ablation):
if the collapse is likewise spread rather than localised, a single-layer-group ablation should
find nothing, and that would be a result rather than a null.

Together with `jain2024wrapper` below, this means **the suppression hypothesis is already
well-supported in adjacent settings**, and §7 should say so. Our contribution there is the
*directional* case, the never-had control that makes erased-versus-suppressed experimental, and
four instruments rather than one.

### `jain2024wrapper.pdf`
In controlled synthetic settings with pruning and probing: "(i) fine-tuning rarely alters the
underlying model capabilities; (ii) a minimal transformation, which we call a **wrapper**, is
typically learned on top" (p. 1). The precedent for reading fine-tuning as modulation rather
than replacement.

### `biderman2024lora.pdf`
LoRA substantially underperforms full fine-tuning on target domains (code, maths) but "better
maintains the base model's performance on tasks outside the target domain", more so than weight
decay or dropout (p. 1). **A live qualification on this project**, which is LoRA throughout: if
LoRA forgets less, our collapse magnitudes are if anything *conservative*, and the `fullft_*`
arms exist to say by how much.

### `luo2023forgetting.pdf`
Catastrophic forgetting observed across 1b–7b during continual instruction tuning, and
**worsening with scale** in that range. Relevant to the large tier: if collapse also grows with
scale, that is consistent rather than surprising.

### `longpre2023flan.pdf`
Ablations across the Flan Collection find "task balancing and enrichment techniques are
overlooked but critical" (p. 1) — gains attributed to methods often belong to the data mixture.
Background for §8, and the general form of our specific claim.

### `lipton2018troubling.pdf`
Names the pattern directly: "**Failure to identify the sources of empirical gains**, e.g.
emphasizing unnecessary modifications to neural architectures when gains actually stem from
hyper-parameter tuning" (p. 1). Our §8 is an instance with direction in place of
hyper-parameters. Their guidance on tone — keep examples short and specific, since criticising
individual papers is sensitive — is the rule the attribution section already follows.
