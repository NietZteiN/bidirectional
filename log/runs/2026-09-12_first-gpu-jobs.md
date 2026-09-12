# 2026-09-12 — the first GPU jobs, and what they revealed

Three things were unblocked or found today. Two are infrastructure; the third changed a
pre-registered rule.

## 1. vLLM was broken on every GPU node, and the lock could not have told us

`vllm==0.26.0` resolves from PyPI to a **CUDA 13** build. Its `_C_stable_libtorch.abi3.so` has
`NEEDED libcudart.so.13`, and juno's driver is r550/CUDA 12.4, so on a GPU node the import dies:

    ImportError: libcudart.so.13  (vllm/platforms/cuda.py:23)

**It imports fine on the login node**, where there is no GPU and `vllm.platforms` never resolves
`CudaPlatform`. That is why it survived every check until job 388586 — the first job that
actually asked for a card.

obtune's working env has `vllm-0.26.0+cu129`, installed from the project's GitHub release asset
and linked against `libcudart.so.12`. Found by `readelf -d` on both copies of the extension, then
by reading obtune's `direct_url.json`. The local version `+cu129` satisfies `==0.26.0`, so
**nothing in the 222-package lock records the difference** — replaying the lock reproduces the
broken env exactly. `env/setup_env.sh` now installs the wheel by URL and asserts the suffix.

*The transferable lesson, now in CLAUDE.md §2: a package that imports on the login node has not
been verified.*

Confirmed working on g-08-05 (H200 NVL, driver 550.163.01): engine built, CUDA graphs captured,
200 prompts at ~46k input tok/s, COMET scored on the same card at batch 64.

## 2. Two engines on one card, in a path that runs on every SQL eval

`bidir.engine.get_engine`'s own docstring said "two engines on one GPU would each claim
`gpu_memory_utilization` of the card and the second would OOM" — and `sql` and `d2t` did exactly
that. Both round-trip through a frozen granite-3.1-8b that is resident **beside** the model under
test, so the domain engine at 0.85 plus `generate_with`'s hardcoded 0.45 asked for **1.30 of the
device**. vLLM's utilisation is a share of total device memory reserved outright per engine;
engines do not negotiate.

This sat in the eval path, not just the gate, and would have died inside vLLM's memory profiler
several frames from anything naming the first engine. Fixed by splitting each affected domain's
budget 0.45/0.45 and by having `get_engine` track claimed utilisation and refuse past 0.95,
naming both models and the sum.

## 3. The base gate's precondition could not fail — see Amendment 13

τ is the 0.25 quantile of the base model's **own** distribution, so the base's strict rate is
`1 − tau_quantile ≈ 0.75` by construction. The first run returned forward **0.7450** and reverse
**0.7500** on `mt_en-de`. Those restate the quantile; they do not measure competence. Amendment 4
had registered exactly this check as the precondition for the `−50 %` contrast.

The gate now additionally requires τ to clear the **echo baseline's** 90th percentile by 0.05 —
the model's own input copied to the output, scored through the identical metric and direction.
Measured on the data rather than taken from the metric literature, so it transfers to chrF₂ on
`d2t` unchanged.

## A free harness validation

`mt_en-de` **reverse** and `mt_de-en` **forward** are the same measurement: de→en on the same 200
FLORES devtest sentences. Verified identical — same 200 pairs, same order, byte-identical rendered
prompts. Measured in two separate passes:

| pass | τ (COMET-22) |
|---|---|
| `mt_en-de` reverse | 0.8619 |
| `mt_de-en` forward | 0.8609 |

Agreement to **0.0010**, from two independent cell definitions reaching the same task.

Note what it also says about determinism: these two passes shared one process and one engine, and
still did not agree bitwise. So consecutive identical batches on one engine are *nearly* but not
exactly reproducible, and 0.0010 COMET is a **lower** bound on the cross-pass floor —
`scripts/30_determinism_floor.py` measures the real one across fresh engines, which is why it was
rewritten today to stop looping inside a single process.

## Base model, en↔de, llama32-3b seed 17, n=200

| cell | direction | τ | rate | format_fail | echo |
|---|---|---|---|---|---|
| `mt_en-de` | forward (en→de) | 0.7649 | 0.7450 | 0.010 | 0.000 |
| `mt_en-de` | reverse (de→en) | 0.8619 | 0.7500 | 0.000 | 0.000 |
| `mt_de-en` | forward (de→en) | 0.8609 | 0.7500 | 0.000 | 0.000 |

The model is meaningfully better at **de→en than en→de** (τ 0.86 vs 0.76). That asymmetry is the
RQ2 moderator — Zhu et al. (2024) report collapse for X→en but not en→X — so the two cells were
always going to differ, and now there is a base-level number to read that against.


## The determinism floor, measured (RUN_PLAN §9 step 2)

`scripts/30_determinism_floor.py --domain mt_en-de --model llama32-3b --passes 3`, on an H200,
each pass in its own process with its own engine and a companion load of 4:

| direction | rates across the three passes | spread | generations differing | graded trials flipped |
|---|---|---|---|---|
| forward (en→de) | 0.7533, 0.7533, 0.7533 | **0.00 pp** | 0.3 % | **0 / 300** |
| reverse (de→en) | 0.7433, 0.7433, 0.7433 | **0.00 pp** | 0.0 % | **0 / 300** |

Much tighter than the workshop paper's 6–8 % of generations differing and 4–8 graded flips per
1,500. Two things make that credible rather than suspicious:

* it was measured across **separate engine instantiations**, which is the whole reason the
  script was rewritten today — a single-process loop would have reported 0.00 pp for a reason
  that meant nothing;
* the residual 0.3 % forward is non-zero, so the measurement is sensitive to *something*. It is
  the reverse direction that is bit-identical across all three passes.

**The reported rule is unchanged: `max(0.5, measured)` = 0.5 pp.** A floor of 0.00 pp from 300
instances and three passes bounds the flip rate below roughly 1 %, not at zero, and the paper
should not quote a resolution its sample cannot support. So "one pass per table, contrasts only
within a pass" stays — now as a conservative convention with a measured number behind it rather
than an assertion.

One consistency note: the floor's rates (0.7533 / 0.7433) sit slightly off the gate's
(0.7450 / 0.7500) because the floor takes the first 300 test instances and the gate the first
200. Both are subsets of the same 1,012.
