"""Mechanism experiments (RQ5): is the inverse capability erased, or suppressed?

The floor and the upside are separated deliberately (plan §8). Experiments 1-3 are behavioral,
cheap, and together they settle erased-versus-suppressed; 4-6 are the mechanistic upside that
makes the section distinctive but are not a dependency for submission.

  1  elicitation   simple / few-shot / CoT / augmented prompting on a collapsed model.
                   Partial recovery under CoT is the first suppression signal.
  2  alpha_scale   scale the LoRA delta by a in [0, 1]. If reverse collapses at small a while
                   forward rises slowly, the collapse is a cheap direction in weight space.
  3  relearn       (an ARM, not a module: `relearn{10,50,200,1000}` in bidir.arms) — from sft,
                   train on k reversed pairs. Fast relearning is latent knowledge; a curve
                   matching a never-had control is erasure. This carries the reversal-curse
                   contrast.
  4  layer_ablate  zero the LoRA delta by layer group and measure reverse recovery.
  5  direction_probe  linear probes for the requested direction on the residual stream, and a
                   logit-lens read of where first-token mass goes.
  6  sensitivity   vary the direction instruction with the input fixed; H1 predicts near-zero
                   output sensitivity after sft and restored sensitivity after mix5.

H1 (the hypothesis, not the obfuscator): forward-only training replaces the
instruction-conditioned mapping with an unconditional one. H2: the reverse mapping's
representation is degraded. The workshop paper's 5 % dose result, echo rates, and partial CoT
recovery already favour H1.
"""
