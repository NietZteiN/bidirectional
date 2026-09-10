# Related work, and what is left for this paper to claim

*Written 2026-09-10, prompted by: does "The Path Not Taken: Duality in Reasoning about Program
Execution" (ACL 2026) already take this idea?*

**Short answer: no.** But a different paper is much closer than that one, it is already known to
this project, and the honest framing of the contribution depends on saying so first.

**Read, not abstracted.** Every paper discussed here is in [`papers/`](papers/) and has been read
past the abstract; claims below cite the page they came from. Reading changed three of the four
conclusions I had drawn from abstracts alone, and §4.1 was reversed outright — which is the
argument for keeping the PDFs in the repo rather than a list of links.

---

## 1. The paper that prompted this

**Hasanov, Hassan Sibat, Karmaker, Yadavally (2026). "The Path Not Taken: Duality in Reasoning
about Program Execution." ACL 2026 Main.** [arXiv:2604.20917](https://arxiv.org/abs/2604.20917)

They argue that understanding program execution requires evaluating its **duality**: predicting a
program's behaviour for a given input, and inferring how the input must be mutated toward a
behavioural objective. They instantiate this as **DexBench**, 445 paired instances, and evaluate
13 LLMs.

**It does not take this idea, and the difference is structural rather than a matter of degree:**

| | DexBench | this paper |
|---|---|---|
| what is studied | how well **existing** models do both directions | what **fine-tuning on one direction does to the other** |
| does it fine-tune? | **no** — 13 models evaluated as released | yes; the phenomenon only exists under fine-tuning |
| does it measure degradation? | no | that is the headline |
| does it propose a remedy? | no | the dose-response cure is the prescription |
| scope | one domain (program execution) | six paired domains, three model lineages |
| size | 445 paired instances | ~120k pairs across 17 cells |

**It is nonetheless a paper this one must cite, prominently, for two reasons.**

First, it **supports the premise**. This paper's claim is that fine-tuning destroys a capability
the base model *already had*. That premise needs evidence that base models can do both
directions, and DexBench is exactly that evidence for the execution domain — from a
contemporaneous ACL paper, independently collected.

Second, it is **adjacent to the `exec` cell, though less overlapping than the abstract suggested**.
Reading the task definition (p.~3): their backward task is *counterfactual input mutation* —
given a program, an input, and a target execution path, mutate the input so execution follows
that path instead. Our `exec` cell is CRUXEval output ↔ input prediction: given a program and an
output, find any input producing it. Both are inverse-execution reasoning; they are not the same
task. The duality *framing* for program execution is theirs and should be credited as such,
without implying we run their benchmark.

**Their code and data are public** — `https://github.com/sail-ucf/dexbench` (p.~2, footnote 2).
So evaluating our arms on DexBench directly is available for roughly **1 GPU-hour**, and would
let the paper report a directional-collapse result on an independent benchmark rather than only
on its own corpora. That is the single most valuable action item in this document, and it is now
concrete rather than conditional.

---

## 2. The paper that is actually closest — and it is already ours

**Nikiema, Samhi, Moumoula, Djiré, Kaboré, Klein, Bissyandé (2025). "Using Contrastive Learning
to Improve Two-Way Reasoning in Large Language Models: The Obfuscation Task as a Case Study."**
[arXiv:2509.05553](https://arxiv.org/abs/2509.05553)

They document **cognitive specialization**: fine-tuning on forward transformations not only
fails to produce reverse capability but *actively degrades pre-existing bidirectional reasoning
in base models*. They propose Contrastive Fine-Tuning (CFT) as the cure.

**That is this paper's phenomenon, named first, in one domain.** Any framing that presents
directional collapse as newly discovered is wrong and a reviewer will say so. What remains, and
what the paper should lead with:

1. **Generality**, tested rather than asserted — six paired domains, three model lineages, three
   seeds. Nikiema et al. show it for code obfuscation.
2. **Invertibility as a predictor** of what a reverse dose can recover, including the separation
   of *undetermined* inverses (`fmt_det*`) from *determined but hard* ones (`automata`).
3. **The dose-response cure.** How little reversed data suffices, budget-matched at every rung,
   is a prescription rather than an observation.
4. **Mechanism** — suppressed versus erased, with a never-had control (`fmt_novel`) that turns
   relearning speed into evidence.
5. **Attribution** — that a direction-matched SFT baseline accounts for CFT's gain. This is the
   workshop paper's result and it is a claim *about* Nikiema et al., which is exactly why the
   tone rule in `paper_bidirectional/README.md` matters: diagnostic, not prosecutorial, and they
   may review this.

**The load-bearing fact for that claim, verified in the PDF rather than trusted.** Nikiema et al.
*do* declare a Bidirectional Fine-Tuning baseline: "CFT effectiveness is assessed through
comparison against Standard Fine-Tuning (SFT) ... and Bidirectional Fine-Tuning (BFT) using
forward generation plus reverse deobfuscation tasks" (p.~6). The string `BFT` appears **exactly
once in the paper**, in that sentence, and **never in a results table**. So the precise and
defensible statement is the one the workshop draft already makes — they name the baseline and
report no number for it — and *not* "they never ran the obvious baseline", which would be false.
Checked by full-text search of `papers/nikiema2025contrastive.pdf`.

Scale, for the generality argument: six models, three obfuscation techniques, 10,000 Java
programs from CodeNet (p.~1--3). Our grid is five model lineages across six paired domains, which
is the axis their design does not cover.

---

## 3. Direction in machine translation

**Zhu, Chen, Zhang, Haddow, Shen, Klakow (2024). "Fine-Tuning Large Language Models to Translate:
Will a Touch of Noisy Data in Misaligned Languages Suffice?" EMNLP 2024 Main.**
[aclanthology.org/2024.emnlp-main.24](https://aclanthology.org/2024.emnlp-main.24/)

Fine-tuning on a single direction generalizes to others — **except** that English on the target
side causes task misinterpretation that damages translation into non-English. That is directional
collapse with a moderator, in MT, two years earlier.

This is why every MT cell runs in **both** training directions (`mt_en-de` and `mt_de-en`,
`mt_en-zh` and `mt_zh-en`). If collapse appears for X→en and not en→X, that asymmetry is the
RQ2 moderator result and a replication of theirs, not a new phenomenon. The plan already says
this; the citation is confirmed and the wording of their finding is as recorded.

---

## 4. Work found in this search that changes what we do

These were not in the plan. Two have design consequences.

### 4.1 The LoRA capacity wall — I had this wrong from the abstract

**"Directional Optimization Asymmetry in Transformers: A Synthetic Stress Test."**
[arXiv:2511.19997](https://arxiv.org/abs/2511.19997) · `papers/dirasym2025synthetic.pdf`

From the abstract I flagged "LoRA encounters a sharp capacity wall on high-entropy inverse
mappings" as a live threat to our kill-gate, and proposed a rank sweep to settle it. **Reading
the paper changes that on three counts.**

*The task has no structure at all.* The corpus is random strings of length 8 drawn i.i.d.
uniform, and "by design, no token, substring, or structural pattern appears with higher-than-
random frequency" (p.~2). It is a pure memorization task over 40,000 arbitrary pairs. A low-rank
update is exactly the wrong instrument for memorizing a lookup table, and that says little about
inverse tasks with real structure — deobfuscation, translation, factorization. Our inverses are
hard because the search is hard, not because the mapping is arbitrary.

*A rank sweep would not have settled it anyway.* Table 3 (p.~8): inverse excess loss is 5.06 at
r=8, 4.85 at r=64, 4.75 at r=256. Rank buys almost nothing. And the wall is in **both**
directions — forward excess loss is 4.85/1.66/1.60 — so it is not a directional finding about
LoRA at all; LoRA simply underperforms scratch and full fine-tuning on this task, in both
directions. My Amendment 5 prediction ("`rev` at r=32 within the seed band of r=128") was aimed
at a claim the paper does not make.

*The scale is GPT-2 Small.* Against our 1--12B panel.

**What the paper actually gives us is better than a caveat — it is a formal version of RQ3.**
Their branching factor *K* is invertibility made precise: the forward map is deterministic
(H = 0), the inverse is one-to-many with an analytic entropy floor H(A|B) = log K. And their
control result (p.~3): "Transformers trained on the bijective case K=1 exhibit matched forward
and reverse convergence after accounting for the entropy floor ... the asymmetries observed for
K>1 emerge only once the inverse task becomes many-to-one."

That is our `fmt` (100 % determinable) baseline prediction, derived analytically on a
semantics-free task. Our `fmt_det*` ladder varies the same quantity empirically, and our
`automata` cell isolates the other half — an inverse that is *determined* (K=1 in their terms)
but computationally hard. Their framework cannot express that case, because their difficulty is
entirely entropic. **This is a citation that strengthens RQ3, not a threat to it.**

Retained from the original worry, at much lower priority: the `fullft_*` arms are still the
honest control for "is this a LoRA artifact?", and the question will still be asked in review.
Keep them; drop the rank sweep unless a reviewer asks.

### 4.2 Mixing forward and reverse data may have a cost we do not measure

**"When Inverse Data Outperforms: Exploring the Pitfalls of Mixed Data in Multi-Stage
Fine-Tuning."** [arXiv:2509.13079](https://arxiv.org/abs/2509.13079)

They build `r1k` by inverting 1,000 forward examples from `s1k`. SFT on the reverse set alone
*beats* the forward set by 1.6–6.8%. But **"naively mixing forward and reverse data during SFT
weakens the directional distinction"** — the model becomes less able to tell which direction it
was asked for.

Our `mix*` arms are exactly "naively mixing forward and reverse data". Their finding is not a
contradiction of ours — they study reasoning-trace distillation, we study paired transformations
— but it predicts a **cost we do not currently measure**: direction confusion.

**Two differences that reading turned up, and that bound the transfer.** Their "directional
distinction" is a *log-likelihood margin between preferred and dispreferred reasoning paths*,
narrowed to 0.05--0.1 by mixing (p.~5) — not instruction-following sensitivity, which is what our
mechanism experiment 6 measures. And their mixture is a single 1:1 blend of reasoning traces,
where our ladder predicts a knee **below 10 %** reverse share. Their result is about a regime our
prescription does not occupy.

So the prediction is worth registering but should not be overstated: mechanism experiment 6 now
runs across the whole dose ladder rather than only on collapsed models, and if sensitivity does
fall at the doses we recommend, the ladder acquires an upper bound as well as a lower one. If it
does not, the natural reading is that a small dose sits below the regime where mixing costs
discrimination — which is itself worth one sentence, and is why running it is cheap insurance
rather than a concession.

### 4.3 Bidirectional objectives need not create a unified representation

**"The Illusion of Latent Generalization: Bi-directionality and the Reversal Curse."**
[arXiv:2604.04943](https://arxiv.org/abs/2604.04943)

Objectives with bidirectional supervision mitigate the reversal curse, but representation
distances and linear probes are consistent with **forward and reverse being stored as distinct
entries**, not a single direction-agnostic concept. Objective-level fixes can improve reversal
behaviour "without necessarily inducing the kind of latent generalization one might expect".

Directly relevant to RQ5 and to `bidir.mech.direction_probe`, which uses the same instrument
(linear probes on the residual stream). It also justifies the shared-system-prompt choice: if
directions can be stored separately while behaviour looks bidirectional, then two personas would
let disjoint circuits masquerade as bidirectionality, which is why there is one system prompt for
both directions.

### 4.4 Post-hoc recovery of damaged capabilities

**"Spectral Unforgetting: Post-Hoc Recovery of Damaged Capabilities Without Retraining."**
[arXiv:2605.20296](https://arxiv.org/abs/2605.20296)

Not yet read beyond the title and search snippet. If a capability damaged by fine-tuning can be
recovered *without retraining*, that is a strong form of the suppression hypothesis and belongs
in RQ5's discussion — possibly as a cheaper alternative to the α-scaling and layer-ablation
experiments. **Flagged for reading, not yet relied on.**

---

## 5. Standing background (unchanged, from the plan)

The reversal-curse lineage — Berglund et al. 2024, Allen-Zhu and Li 2023, Golovneva et al. 2024
(reverse training), Kitouni et al. 2024 — describes a capability **never acquired**. Directional
collapse destroys one the base model **has**. The relearning-cost experiment against
`fmt_novel` is what makes that distinction experimental rather than rhetorical, and §4.1 and §4.3
above are now the two most relevant recent entries in that lineage.

Forgetting (Luo et al. 2023, Kotha et al. 2024, Zheng et al. 2025, Scialom et al. 2022,
Biderman et al. 2024) measures diffuse general loss; here the loss is targeted and total on the
inverse while general probes fall a few points. The `replay` arm is what separates the two, and
§4.1's LoRA caveat is a reason to take Biderman et al.'s "LoRA forgets less" seriously as a
confound rather than only as a citation.

---

## 6. What to do about it

1. **Cite Nikiema et al. 2025 in the first paragraph**, and never frame the phenomenon as newly
   discovered. The contribution is generality, the invertibility predictor, the dose cure, the
   mechanism, and the attribution result.
2. **Cite Hasanov et al. 2026 as support for the premise**, and credit the duality framing for
   program execution to them when discussing the `exec` cell.
3. **Try to evaluate on DexBench directly** if the instances are public. ~1 GPU-hour for a
   directional-collapse result on an independent benchmark.
4. **Drop the planned rank sweep** (§4.1: it targets a claim the paper does not make), keep
   `fullft_*` as the LoRA-artifact control, and cite arXiv:2511.19997 as analytic support for
   RQ3's information-theoretic axis instead.
5. **Run mechanism experiment 6 on the `mix*` arms**, and pre-register the prediction that
   instruction sensitivity falls as the reverse share rises.
6. **All ten papers in `papers/` are now read**, with per-paper notes in
   [`papers/REFERENCES.md`](papers/REFERENCES.md). Three findings from that pass are registered
   in Amendments 6 and 7: the withdrawn LoRA rank sweep, the direction-specific MT prediction,
   and DG-Hard as mechanism experiment 7.
7. ~~Implement mechanism experiment 7~~ — **done**, `bidir.mech.spectral`, in the RQ5 floor.
   Remaining: it currently refuses full fine-tuning deltas, which is the arm it most wants, since
   a LoRA delta is already low-rank and has no noise bulk to remove.

---

## Sources

- [The Path Not Taken: Duality in Reasoning about Program Execution (arXiv:2604.20917)](https://arxiv.org/abs/2604.20917)
- [Using Contrastive Learning to Improve Two-Way Reasoning in LLMs (arXiv:2509.05553)](https://arxiv.org/abs/2509.05553)
- [Fine-Tuning Large Language Models to Translate (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.24/)
- [Directional Optimization Asymmetry in Transformers (arXiv:2511.19997)](https://arxiv.org/abs/2511.19997)
- [When Inverse Data Outperforms (arXiv:2509.13079)](https://arxiv.org/abs/2509.13079)
- [The Illusion of Latent Generalization (arXiv:2604.04943)](https://arxiv.org/abs/2604.04943)
- [Spectral Unforgetting (arXiv:2605.20296)](https://arxiv.org/abs/2605.20296)
- [The Reversal Curse (arXiv:2309.12288)](https://arxiv.org/abs/2309.12288)
- [Reverse Training to Nurse the Reversal Curse (arXiv:2403.13799)](https://arxiv.org/abs/2403.13799)
