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
