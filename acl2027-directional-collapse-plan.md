# ACL 2027 plan. Directional collapse under one-way fine-tuning

Working title options

- One-Way Fine-Tuning Erases the Way Back. Directional Collapse in Paired Tasks and a Zero-Cost Cure
- Directional Collapse. Fine-Tuning on One Direction of a Paired Task Silently Destroys the Other
- A Little Reverse Data Goes a Long Way. Preserving Inverse Capabilities in Paired Fine-Tuning

## 1. Pitch

Many NLP tasks come in pairs whose training instances are the same data read in two directions (obfuscate and deobfuscate, translate en→de and de→en, text-to-SQL and SQL-to-text, RDF-to-text and text-to-RDF). Fine-tuning a model on one direction destroys the ability it already had in the other, the loss is far larger than what general benchmarks register, and only where the transformation is invertible can the loss be undone. Replacing a small share of the forward pairs with their own reversal prevents the collapse at no extra cost. We characterize the phenomenon across five paired domains and three model families, show that the collapsed capability is suppressed rather than erased, and show that auxiliary objectives credited with restoring inverse capabilities are explained by the reversed data they carry.

## 2. What changes relative to the workshop paper

| Workshop paper | ACL paper |
|---|---|
| Headline is attribution (CFT's gain belongs to data direction) | Headline is the phenomenon (directional collapse) and its cure |
| One domain (code obfuscation), one family, single-seed 7B | Five paired domains, three families, three seeds at small scale |
| Generality asserted (translation, compilation, format conversion) | Generality tested, with invertibility as a predictor |
| Limitation. Only invertible transforms recover | Prediction. Recovery ceiling tracks invertibility |
| CFT replication is the main result | CFT is one worked example in a short attribution section, alongside an MT objective-based fix |
| No mechanism | Suppression versus erasure, tested behaviorally and with adapter analysis |
| Contrast with reversal curse and forgetting is argued in prose | Contrast is experimental (replay arm, relearning cost, IFEval control) |

Suggested single term throughout. "Directional collapse." The workshop paper uses "asymmetric capability collapse" and "silent capability erasure" interchangeably. One term, defined once.

## 3. Research questions

Each RQ lists the prediction from the workshop results and the experiment that decides it.

**RQ1. Generality and disproportion.** Does forward-only fine-tuning collapse a pre-existing inverse capability across paired NLP tasks and model families, and is the loss disproportionate to general degradation?
Prediction. Yes in most domains. Reverse success falls to near zero while general probes fall a few points.
Decides it. base versus sft on reverse success and on held-out general probes, per domain and family. The replay arm (forward data plus generic instruction data at matched volume) separates directional loss from ordinary forgetting. IFEval and an unrelated direction-specified task separate it from a general loss of instruction following.

**RQ2. Moderators.** What determines whether collapse happens?
Motivation. The MT literature already contains a partial instance with a moderator (see §4). Fine-tuning with English on the target side damages translation into other languages, while fine-tuning toward a non-English target generalizes across directions (Zhu et al. 2024, EMNLP).
Candidates to test. Which direction is trained (forward target coincides with a dominant pretraining output mode or not), instruct versus base model, LoRA versus full fine-tuning, scale, and whether the forward task is a minority of a mixed-task SFT set.
Decides it. MT arms trained in each direction separately for two language pairs. Full fine-tuning arms for sft and mix5. A mixed-task arm where forward pairs are 20% of a five-task SFT mixture.

**RQ3. Invertibility ceiling.** Does the recoverable ceiling under bidirectional training track how much of the input's information the transformation preserves, independent of reverse-data share?
Prediction. Yes. Structural and lossless transforms recover fully, renaming and designed-lossy transforms sit at a floor no reverse share moves.
Decides it. Within-domain contrasts (structural versus renaming in code, lossless versus designed-lossy in format conversion) plus the synthetic ladder in §4, where information loss is set in known steps.

**RQ4. Dose and cost.** How much reversed data is needed, and is the cure free on forward accuracy, compute, and general ability?
Prediction. Knee below 10% in every domain. Forward accuracy unchanged. General cost is a property of fine-tuning, not of direction.
Decides it. Dose ladder {0, 1, 5, 10, 25, 50}% by replacement. Budget accounting in sequence tokens and supervised tokens as in the workshop paper. Forward accuracy per condition.

**RQ5. Erased or suppressed?** Is the inverse capability removed from the weights or masked by what forward-only training installs, and how does that differ from the reversal curse?
Hypotheses. H1, forward-only training replaces the instruction-conditioned mapping with an unconditional one (the model applies the trained mapping or echoes regardless of the direction requested). H2, the reverse mapping's representation is degraded. The 5% dose result, the echo rates, and the partial CoT recovery in the workshop paper's Appendix L already favor H1.
Decides it. The six experiments in §8. The relearning-cost contrast also separates collapse from the reversal curse. A capability that was never present should need far more reversed data than one that collapsed.

**RQ6. Attribution.** When an auxiliary objective on paired data is credited with restoring the inverse direction, does a direction-matched SFT baseline account for the gain?
Prediction. Yes for CFT (already shown). The MT unlikelihood fix for off-target translation (Zan et al. 2024) and a round-trip consistency loss are the natural second and third cases.
Decides it. Each method against SFT on the same directional mixture at matched compute.

## 4. Tasks and datasets

Design rule. Both directions must be real tasks that the base model can already do, reverse pairs must be free (swap the fields), and every domain needs a strict per-instance criterion with a correctness check, since the workshop paper showed the published deobfuscation criterion was permissive.

| Domain | Forward / reverse | Train source | Eval source | Forward criterion | Reverse strict criterion | Invertibility | Status |
|---|---|---|---|---|---|---|---|
| Code obfuscation (Python, JS) | clean→obfuscated / obfuscated→clean | existing corpus (APPS, CRUXEval, HumanEval) | 300 held-out | execution | published criterion ∧ execution | structural invertible, renaming lossy | done |
| Machine translation | en→de, en→zh, and each reverse | News Commentary or Europarl subset, ~7k pairs | FLORES-200 devtest (1,012), WMT test sets | correct target language ∧ COMET-22 ≥ τ | same ∧ not echo | near-lossless | new |
| Text-to-SQL | NL→SQL / SQL→NL | Spider train (7,000) | Spider dev subset (500); BIRD dev as harder variant | execution accuracy via test-suite evaluation | round-trip execution equivalence through a frozen parser ∧ not echo; BLEU and BERTScore to reference as secondary | near-lossless (style lost, semantics kept) | new |
| Data-to-text | RDF→text / text→RDF | WebNLG 2020 v3.0 (~13k train, sample 7k) | WebNLG 2020 test | round-trip triple recall through a frozen extractor, or chrF++ ≥ τ | exact triple-set match | near-lossless | new |
| Format conversion (synthetic) | JSON→YAML, JSON→XML, Markdown table→CSV, plus designed lossy variants | generated, 7k | generated, 500 | parse ∧ structural equality | parse ∧ structural equality | lossless by construction | new, cheap |
| Execution prediction (optional) | output prediction / input prediction | existing corpus with test inputs | CRUXEval (800) | execution | execution, any valid input passes | set-valued but determinable | optional |

Secondary candidates if compute allows. Formality transfer (GYAFC, needs Yahoo L6 access), transliteration (Dakshina), decompilation (LLM4Decompile, HumanEval-Decompile).

**The synthetic invertibility ladder (RQ3).** Inside format conversion, define variants that drop a known fraction of the information (keep all values, drop 10% of leaf values, drop 50%, drop all values and keep only key paths). The recoverable ceiling under mix50 should track the fraction of instances whose inverse is determinable from the input. Natural domains confound invertibility with difficulty, so this ladder is the clean test, and it costs almost nothing to run.

**Execution prediction is worth including.** Output prediction and input prediction are a paired direction with execution-based verification, the base model does both, and the stacking paper's observation that breadth SFT forgets input prediction looks like the same phenomenon. Confirming it here ties the series together.

**Sizing.** About 7k forward instances per epoch per domain to match the code corpus, so budgets are comparable across domains. 300 to 500 evaluation instances per domain, identical across all arms, never in any training split. Thresholds τ are set from the base model's score distribution before any fine-tuned model is scored, and continuous metrics (COMET, chrF++, triple F1) are reported beside the strict rates.

**What the literature check found for MT, and why it helps.** Zhu et al. (2024, EMNLP main) report that fine-tuning an LLM on a single translation direction generalizes to other directions, except that English on the target side causes task misinterpretation that damages translation into non-English languages. That is a directional collapse of the reverse direction with a moderator (it happens for X→en training, not en→X). The language-diversity paper (arXiv 2505.13090, 2025) reports that more training directions remove off-target output, and cites Caswell et al. (2025) for catastrophic forgetting under X→en training on multi-parallel data (verify the title). Li et al. (2023, mFTI) see only a small reversed-direction drop after tuning on 16 directions. Zan et al. (2024) fix off-target translation with instruction-conflicting samples and an unlikelihood loss, an objective-based fix. Implications. The MT section should replicate Zhu et al.'s asymmetry rather than claim a new phenomenon, show it has the same dose cure and the same suppression signature as code, use the asymmetry as the moderator test for RQ2, and test the unlikelihood fix against direction-matched SFT for RQ6. A known MT anomaly that turns out to be one case of a cross-domain phenomenon is a stronger story than a novelty claim would be, but Zhu et al. must be cited prominently. Read Zhu et al. and Caswell et al. closely before designing the MT arms.

## 5. Models and recipe

Families. Pin the current release of each when you start.

- Qwen2.5-Coder 1.5B and 7B for continuity with the workshop paper on the code domains, and Qwen3 1.7B and 8B for the NLP domains.
- Llama 3.2 3B and Llama 3.1 8B.
- Gemma 3 4B and 12B (1B and 4B if compute-bound).
- One fully open-data model (OLMo) on at least one domain. Its released pretraining data lets you check that the reverse direction was actually present before fine-tuning, which the "capability the base already had" premise rests on, and it answers the contamination question directly.

Base versus instruct. Run instruct models as the primary grid for the NLP domains. That is the practitioner case, and a reverse instruction being ignored is part of the mechanism story. Replicate one domain on base models.

Recipe. LoRA rank 32, three epochs, learning rate 1e-4, batch 64, one recipe for every arm, as in the workshop paper. Seeds {17, 42, 1234} at small scale, at least two seeds at 7 to 12B for core arms. Single shared system prompt across directions (flipsym showed no difference at 1.5B). Add full fine-tuning arms for sft and mix5 on one small model per family, since LoRA is known to forget less than full fine-tuning (Biderman et al. 2024) and reviewers will ask.

## 6. Arms and grid

| Arm | Training examples | Answers |
|---|---|---|
| base | none | untouched control |
| sft | forward only | the collapse |
| fwd2x | forward only, twice the epochs | not a matter of training longer |
| rev | reverse only | reverse ceiling |
| mix1, mix5, mix10, mix25, mix50 | forward with that share replaced by reversed pairs | dose ladder, budget-matched to sft |
| flip | forward plus all pairs reversed | doubled-data reference |
| replay | forward plus generic instruction data at mix50's replaced share | directional loss versus ordinary forgetting |
| mixed-task | forward pairs as 20% of a five-task SFT set | does collapse survive realistic mixtures |
| full-ft | sft and mix5 with full fine-tuning | LoRA artifact check |
| relearn-k | sft, then k ∈ {10, 50, 200, 1000} reversed examples | erased versus suppressed, reversal-curse contrast |
| cft | forward plus equivalence judgements (code) | attribution, worked example |
| unlikelihood | Zan et al. recipe (MT) | attribution, second case |
| roundtrip | forward plus a round-trip consistency loss (MT or SQL) | attribution, third case |

Grid. Full arm set at small scale (1.5 to 4B) for every domain, three families, three seeds. Core cells (base, sft, mix5, mix50, replay, rev) at 7 to 12B for two families with two seeds. Mechanism and attribution arms on one family at both sizes. The workshop paper's sparse-grid logic, same reasoning.

Compute. If the 8.8 GPU-hour figure was for cftflip at 3.74× sequence tokens, a forward-sized 7B run is about 2.4 GPU-hours on the same hardware, and a 1.5B run under one. Rough totals, to be rescaled against your cluster. Small-scale grid, 5 domains × 3 families × 13 arms × 3 seeds ≈ 585 runs at about 1 GPU-hour ≈ 600 GPU-hours. Large-scale core, 5 × 2 × 6 × 2 = 120 runs at about 3 GPU-hours ≈ 360. Evaluation, mechanism, and probes ≈ 250. On the order of 1,200 GPU-hours for the full version, roughly half for the minimum version in §12.

## 7. Measurement

- Per-domain strict criteria as in §4. Echo (output equals input) and off-target (wrong language, wrong format) are tracked separately and never counted as success, since both rose under forward-only training in the workshop paper.
- Both directions and general ability reported for every arm in every domain, so the disproportion is visible per domain rather than argued once.
- General probes with documented zero overlap. MBPP+ for code arms (not HumanEval+). MMLU or MMLU-Pro, GSM8K, and IFEval for NLP arms through lm-evaluation-harness. Write down the overlap check for each probe before running it.
- Uncertainty. Cluster bootstrap by instance, paired contrasts within a single evaluation pass, cross-pass floor stated once. Never quote a cross-pass difference below the floor.
- Pre-register the thresholds, the arm list, and the primary contrasts (sft minus base, mix5 minus sft, replay minus sft, mix50 minus flip) before the large-scale runs.

## 8. Mechanism experiments (RQ5)

1. **Elicitation ladder.** Simple, few-shot, chain-of-thought, augmented prompts, per domain. Partial recovery under CoT for sft (5.3% against 21.9% for base in the workshop paper) is the first suppression signal. Extend it and check whether the recovered fraction is constant across domains.
2. **Adapter scaling.** Scale the LoRA delta by α ∈ [0, 1] and plot forward and reverse success against α. If reverse collapses at small α while forward rises slowly, the collapse is a cheap direction in weight space, consistent with H1.
3. **Relearning cost.** From sft, train on k reversed examples and measure recovery at each k. Compare against a model that never had the capability (a novel synthetic transform, or a family where base scores zero on reverse). Fast relearning from sft is latent knowledge, and slow relearning matching the never-had curve is erasure. This one experiment carries the reversal-curse contrast and the unlearning connection.
4. **Layer localization.** Ablate the LoRA delta by layer group and measure reverse recovery. Task-vector negation (Ilharco et al. 2023) as a second view. If the collapse lives in a few layers, say which and whether they are the same across domains.
5. **Direction probes.** Linear probes on the residual stream for the requested direction and for the target format or language. Logit lens on the first output tokens. H1 predicts the sft model still represents the reverse instruction and places first-token mass on the forward output mode anyway.
6. **Instruction sensitivity.** Vary the direction instruction while holding the input fixed and measure how much the output changes. The unconditional-mapping hypothesis predicts near-zero sensitivity after sft and restored sensitivity after mix5.

Floor and upside. Experiments 1 to 3 are behavioral and cheap, and together they settle erased versus suppressed. Treat 4 to 6 as the upside that makes the section distinctive, not as a requirement for submission.

## 9. Attribution section (RQ6)

Keep it to half a page. Three methods that add an objective on paired data, each against SFT on the same directional mixture at matched sequence tokens. CFT (code) is already done. The unlikelihood fix for off-target translation (Zan et al. 2024) explicitly constructs direction-conflicting instances, so its direction exposure is measurable and matchable. A round-trip consistency loss is the case where reverse supervision is built into the objective. If direction-matched SFT matches all three, the methodological claim from the workshop paper stands on three cases instead of one.

## 10. Related work map

Verify venues and years before citing. Items marked from your bib are ones I know only through the workshop paper.

**Reversal curse and direction in training data.** Berglund et al. 2024 (ICLR) on "A is B" not yielding "B is A". Allen-Zhu and Li 2023 (Physics of LMs 3.2) on reverse knowledge search failing. Golovneva et al. 2024 on reverse training as a data-level cure, evaluated data-matched and compute-matched. Kitouni et al. 2024 (NeurIPS) on the factorization curse. Differentiation. Those describe a capability never acquired; directional collapse destroys one the base model has, the inverse is semantic rather than token order, and the dose is tiny. The relearning-cost contrast makes the distinction experimental.

**Catastrophic and spurious forgetting under fine-tuning.** Luo et al. 2023 on forgetting during continual fine-tuning. Kotha et al. 2024 (ICLR) on forgetting as a shift in implicit task inference, recoverable by conjugate prompting. Zheng et al. 2025 (ICLR) on spurious forgetting, degraded task alignment rather than lost knowledge. Scialom et al. 2022 (EMNLP) on rehearsal with about 1% of prior data preventing forgetting in continual instruction tuning, the closest analogue to the 5% result. Biderman et al. 2024 (TMLR) on LoRA learning less and forgetting less. Shuttleworth et al. 2024 on LoRA intruder dimensions. Qi et al. 2024 (ICLR) on benign fine-tuning removing safety behavior silently. Zhang et al. 2025 (Inverse IFEval) on cognitive inertia after SFT. Differentiation. Forgetting is measured as diffuse general loss; here the loss is targeted and total on the inverse while general loss is modest, and the paired structure lets us test direction-specific replay against generic replay.

**Fine-tuning as wrapper or suppression rather than erasure.** Jain et al. 2024 (ICLR) on fine-tuning learning a thin wrapper over pretrained capabilities. Prakash et al. 2024 (ICLR) on fine-tuning enhancing existing mechanisms. Lee et al. 2024 (ICML) on DPO suppressing rather than removing toxicity. Ilharco et al. 2023 (ICLR) on task arithmetic. Minder et al. 2025 on crosscoders for concepts introduced by fine-tuning. Yang et al. 2025 (arXiv 2504.09757) on restoring a small subset of weights to recover lost alignment. Differentiation. Same lens, applied to a capability rather than an alignment behavior, with dose and relearning cost as behavioral tests.

**Unlearning evaluation, the suppression-versus-removal toolkit.** Lynch et al. 2024 on eight ways to test robust unlearning. Patil et al. 2024 (ICLR) on whether sensitive information can be deleted. Deeb and Roger 2024 on whether unlearning removes information from weights. Zhang et al. 2024 on unlearning failing under quantization. Wang et al. 2025 on invariance for unlearning resilient to fine-tuning, with task-vector analysis of unlearning and fine-tuning directions. Differentiation. Forward-only fine-tuning is accidental unlearning of the inverse; the same audits apply and the same suppression result appears.

**Direction in machine translation and multilingual fine-tuning.** Sennrich et al. 2016 (ACL) back-translation. He et al. 2016 (NeurIPS) dual learning. Johnson et al. 2017 (TACL) multilingual NMT. Zhang et al. 2020 (ACL) on off-target translation. Xu et al. 2024 (ICLR, ALMA) and Alves et al. 2024 (COLM, Tower), both of which train bidirectionally by default. Li et al. 2023 (mFTI). Zhu et al. 2024 (EMNLP) on single-direction fine-tuning, the closest prior observation. Caswell et al. 2025 (verify). The language-diversity paper (arXiv 2505.13090). Zan et al. 2024 on unlikelihood for off-target. Sennrich et al. 2024 on language-contrastive decoding. Lample et al. 2018 and Artetxe et al. 2018 (ICLR) on back-translation as cycle consistency in unsupervised MT. Differentiation. MT treats this as an off-target or misinterpretation problem specific to translation; we show it is one instance of a cross-domain phenomenon with the same dose cure, and we test the objective-based fixes against the direction-matched baseline.

**Code transformation pairs.** Nikiema et al. 2025 (CFT). Roziere et al. 2021 (NeurIPS, DOBF). Roziere et al. 2020 (NeurIPS, TransCoder, back-translation for code). Tan et al. 2024 (EMNLP Findings, LLM4Decompile). Gu et al. 2024 (ICML, CRUXEval, input versus output prediction). Hu et al. 2026 and Guzmán Lorenzo 2026, from your bib.

**Attributing gains to data rather than objective.** Longpre et al. 2023 (ICML, Flan Collection) on data mixture driving gains. Zhou et al. 2023 (NeurIPS, LIMA) on the superficial alignment hypothesis, relevant to why a small dose suffices. Chu et al. 2025 on SFT memorizing and RL generalizing. Lipton and Steinhardt 2018 on mis-attribution in ML scholarship. Dodge et al. 2019 (EMNLP, Show Your Work) and Bouthillier et al. 2021 on reporting and variance. Differentiation. We name directional exposure as a specific variable that standard compute and token matching leave uncontrolled.

## 11. Paper outline (8 pages)

1. Introduction (1 page). The phenomenon, the disproportion, the cure, the mechanism, the attribution consequence.
2. Directional collapse, defined (0.5). Paired tasks, direction, strict criteria, echo and off-target.
3. Setup (1). Domains, models, arms, budget accounting.
4. Collapse is general and disproportionate (1). RQ1 with the replay and IFEval controls. RQ2 moderators, led by the MT asymmetry.
5. A small reversed dose restores it at no cost (1). RQ4 dose ladders across domains, forward accuracy, general cost.
6. Invertibility bounds what can be restored (0.75). RQ3, within-domain contrasts and the synthetic ladder.
7. Suppressed, not erased (1.5). RQ5, elicitation, adapter scaling, relearning cost, then localization and probes.
8. Consequences for attribution (0.5). RQ6, three methods against direction-matched SFT.
9. Related work (0.75).
10. Conclusion (0.25).
Appendices. Per-domain criteria and thresholds, readability treatment (carried over), seeds, determinism floor, full tables, prompts.

## 12. Timeline, minimum version, decision gate

ARR January 2027 (deadline typically mid-January, verify).

- **September.** Read Zhu et al. 2024, Caswell et al. 2025, Kotha et al. 2024, Jain et al. 2024, Zheng et al. 2025 closely. Freeze per-domain strict criteria. Build the MT and SQL pipelines. Run the decision-gate experiment below.
- **October.** WebNLG and format conversion pipelines, second family, seeds, replay and mixed-task arms, dose ladders. Keep the first week light for Fulbright.
- **November.** 7 to 12B core cells. Mechanism experiments 1 to 3, then 4 to 6 as time allows. Attribution arms.
- **December.** Writing, third family, robustness, camera-quality figures. Graduation.
- **Early January.** Internal review, submit.

**Decision gate, end of September.** At 1.5 to 4B, one family, run base, sft, mix5, mix50 for MT (both training directions, one pair) and text-to-SQL. Under 20 GPU-hours. If collapse appears in at least one NLP domain, proceed. If it does not, the paper becomes a boundary-conditions paper (when does one-way fine-tuning collapse the inverse, and why does code differ), which is still publishable but has a different shape, and the plan should be reset before more compute is spent.

**Minimum viable version.** Three domains (code, MT, SQL), two families, three seeds at small scale, one seed at 7 to 8B, mechanism experiments 1 to 3, attribution with CFT only. That is a complete long paper. The full version adds WebNLG, the format ladder, a third family, experiments 4 to 6, and the two MT attribution cases.

**Risks.**
- Collapse may not appear for en→X training in MT. That is a result if framed as a moderator, so design the MT arms in both directions from the start.
- Reverse criteria for text tasks are noisier than execution. Fix thresholds from base distributions before seeing fine-tuned results, and report continuous metrics beside the strict rates.
- Mechanism experiments 4 to 6 may be inconclusive. Experiments 1 to 3 are the floor.
- Time. The authorship-translation paper is aimed at the same cycle. If the gate passes but October slips, EMNLP 2027 (ARR around May) costs the idea nothing.
- Check ARR's current policy on submissions extending non-archival workshop papers.

## 13. Open decisions and things to verify

- Instruct or base as the primary grid for NLP domains (recommendation above is instruct, with one base replicate).
- Language pairs. en–de is the safe high-resource pair. en–zh or en–ja gives a script change that makes off-target output trivially detectable. Two pairs is enough.
- Whether to include execution prediction as a sixth domain. Cheap given the existing corpus, and it links the series.
- Frozen round-trip parsers for SQL→NL and RDF→text scoring. Choose a model that is not in the training grid and report its own accuracy on the references so the round-trip criterion has a known ceiling.
- Verify venues and years for every citation in §10, and the exact title of Caswell et al. 2025.
- Confirm the 8.8 GPU-hour figure refers to the cftflip cell so the compute estimate can be rescaled.
