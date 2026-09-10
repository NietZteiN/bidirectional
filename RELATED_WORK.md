# Related work, and what is left for this paper to claim

*Written 2026-09-10, prompted by: does "The Path Not Taken: Duality in Reasoning about Program
Execution" (ACL 2026) already take this idea?*

**Short answer: no.** But a different paper is much closer than that one, it is already known to
this project, and the honest framing of the contribution depends on saying so first.

Everything below was checked against the papers' own abstracts on 2026-09-10. Abstracts only —
none of these has been read in full, and several claims worth relying on (especially §4) need
the method sections before they go in a submission.

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

Second, it **overlaps the `exec` cell's territory**. Our `exec` domain is CRUXEval output ↔ input
prediction, which is the same duality on the same kind of data. The paper should say plainly
that the duality framing for program execution is theirs, and that our contribution in that cell
is the fine-tuning result, not the observation that both directions are worth measuring.

If DexBench's paired instances are released, **evaluating on it directly is worth ~1 GPU-hour**
and would let the paper report a directional-collapse result on someone else's benchmark rather
than only its own corpora. That is a cheap, strong robustness check and is the single most
valuable action item in this document.

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

### 4.1 A LoRA capacity wall on inverse mappings — a live threat to our design

**"Directional Optimization Asymmetry in Transformers: A Synthetic Stress Test."**
[arXiv:2511.19997](https://arxiv.org/abs/2511.19997)

A fully synthetic, entropy-controlled benchmark: forward tasks with zero conditional entropy,
inverse tasks with analytically determined entropy floors. Even scratch-trained GPT-2 shows a
reproducible directional optimization gap (1.16 nats at K=5), far larger than an MLP's on the
same data. Pre-trained initialization shifts it but does not remove it.

**The sentence that matters for us: "LoRA encounters a sharp capacity wall on high-entropy
inverse mappings."**

This paper is LoRA-based throughout (r=32). If LoRA specifically underperforms on inverse
mappings, then a low `rev` score may be a capacity artifact rather than evidence about
learnability — and `rev` is the **kill-gate**: if it is near zero we declare the direction
unlearnable and treat every other null as uninterpretable. That interpretation would be wrong if
the wall, not the model, produced the zero.

**Consequences, in order of cost:**
- The `fullft_*` arms already exist for exactly this class of objection and should be read as a
  *direct* control on it, not only as the LoRA-forgetting check they were specified as.
- Add a **rank sweep on `rev`** (r ∈ {16, 32, 64, 128}) in at least one domain. Cheap — four
  adapters — and it converts "is `rev` capacity-limited?" from an argument into a measurement.
  obtune already ran a rank sweep, so the recipe exists.
- State it in Limitations regardless of what the sweep shows.

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

We are better placed to catch it than they were, and largely by accident. Mechanism experiment 6
(`bidir.mech.sensitivity`) measures exactly this: how much the output changes when only the
instruction changes. It was written to test H1 on collapsed models. **It should also be run on
the `mix*` arms**, where their result predicts sensitivity *falls* as the reverse share rises.
That is a free, pre-registerable prediction on an experiment already built — and if it holds, the
dose ladder acquires an upper arm as well as a lower one, which strengthens the prescription
rather than weakening it.

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
4. **Add a rank sweep on `rev`** in one domain, and read `fullft_*` as a control on the LoRA
   capacity wall. State the caveat in Limitations either way.
5. **Run mechanism experiment 6 on the `mix*` arms**, and pre-register the prediction that
   instruction sensitivity falls as the reverse share rises.
6. **Read in full before submission**: 2511.19997 §on LoRA, 2509.13079's method, 2605.20296.
   Everything here rests on abstracts.

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
