# Where the project is, and what is left to run

*Generated 2026-09-14. Numbers are measured on this cluster, not estimated: per-adapter costs
come from completed jobs (h100 0.185 h/unit for MT-shaped domains, ×2.7 for long-sequence
domains, ×3.0 on a30), and the throughput figure is what we actually achieved.*

---

## 1. How long

**~378 GPU-hours of useful work remain** across everything below.

The binding constraint is **not** GPU-hours — it is concurrent slots. Measured over the 64 hours
to 2026-09-14 we completed 60 jobs totalling 56.8 GPU-hours, i.e. an effective throughput of
**0.89 GPU-hours per wall-hour** — under one sustained GPU. The juno QoS caps this *account* at
4 concurrent jobs across four projects, and `h100`/`a30` are uncapped but heavily contended (38
and 77 jobs pending from other users when last checked).

| sustained slots | everything below |
|---|---|
| 0.89 (what we were actually getting) | ~18 days |
| 1 | 15.7 days |
| **3 — this project's allocation** | **5.2 days** |
| 6 | 2.6 days |

**This project holds three cells at a time** (`scripts/95_runner.py --max-inflight 3`; a cell is
one training job plus its chained eval, so three is the number of GPUs at peak). Three rather
than four because four other projects share this account and this cluster. The runner is
idempotent, so re-running it simply tops the allocation back up as cells finish.

**The lever is concurrency, not hardware.** The cluster has ~65 GPUs; we are using less than
one. `juno-pri` would give 8 slots at priority 200000 but preempts every other user on a shared
cluster, so it is a decision to be taken deliberately and not by me.

---

## 2. Done

- [x] **Environment** — vLLM `+cu129` wheel (the lock does not capture it), two-engine GPU budget,
      adaptive micro-batch so a job runs on any card with the effective batch held at 64
- [x] **Corpora** — 20 domains built, leakage-checked on content, oracle-audited on a compute node
- [x] **Determinism floor** — 0.00 pp spread, 0/300 graded flips across three fresh engines
- [x] **Decision gate** — PASSED. 3 of 4 cells collapse:
      `mt_de-en` −100 %, `code` −88.9 %, `sql` −54.9 %; `mt_en-de` −12.1 % (no collapse)
- [x] **General-ability control** — IFEval/GSM8K flat: reverse falls 55–100 % while IFEval moves
      0 to −6.5 pp, only `code`'s −6.47 distinguishable from zero
- [x] **Phase 2 wave 1** — full 11-arm set on `mt_de-en`, `mt_en-de`, `sql`, `code` at seed 17
- [x] **26 preregistration amendments**, every one dated and written before the result it governs

### What wave 1 established

- **The knee is at or below 1 %.** `mix1` recovers 99.4 % of base on `mt_de-en`, 95.6 % on `sql`
- **It is direction, not substitution.** `mix50 − replay` = **+40.43 pp** [+37.04, +43.42] on
  `sql`; `replay` itself lands *below* `sft`
- **Not undertraining.** `fwd2x` (double the forward epochs) leaves reverse at 0.0138 / 0.2669
- **Replacing ≈ doubling.** `flip` and `mix50` agree to ≤1.1 pp in all three cells

---

## 3. Running now

- [ ] `ev_code` — the Phase 2 eval for `code` (its 11 adapters exist; the eval was never
      submitted when the grid refused the cell for h200)
- [ ] `gate_diacritics` re-run, after the criterion fix below

---

## 4. Blocked, and why

Eight domains fail their base gate. **Two different reasons, and they need different answers.**

- [ ] **Criterion, not ability** — the gate is rejecting correct answers
  - `diacritics` forward: rate 0.020 with **format_fail 0.910**. *Diagnosed and fixed*: the model
    normalises `’` → `'` and the comparison counted that as "changed the letters". Re-gating.
  - `algebra` (format_fail 0.31) and `algebra_rev` (0.17/0.31) — **not yet diagnosed**; capture
    outputs before touching the criterion, since `automata` proves the cause is not always format
  - `fmt` reverse (0.175) and `fmt_det75` reverse (0.200) — just over the 0.15 threshold
- [ ] **Genuine inability** — no fix warranted
  - `automata`: the model emits grids of the wrong *shape* (7 columns where gold has 6). It
    cannot compute a Game of Life step. The gate is right to refuse the cell.
  - `fmt_det00`: 0.000 both directions with low format_fail — the model parses and is wrong
- [ ] **Never gated**: `coverage`, `d2t` (its gate hit a 3 h walltime — needs longer, it holds a
      second resident model), `exec`, `fmt_novel`

---

## 5. To run, in priority order

Each line is what it buys, not just what it costs.

- [ ] **P1 — finish seed-17 breadth · 9 GPU-h · ~0.4 d**
  `mt_en-zh`, `mt_zh-en`, `relation` — all three gated and passing.
  **`relation` is the one that matters**: it separates directional collapse from the Reversal
  Curse, and its gate already shows the base reciting these facts *both* ways (0.711 / 0.652,
  zero format failures).
  `mt_en-zh`/`mt_zh-en` test whether the collapse asymmetry is about English or about the
  model's stronger direction.
- [ ] **P2 — the format-blocked cells at seed 17 · 16 GPU-h · +0.8 d**
  Includes **`algebra` + `algebra_rev`**, the pair that separates "directional" from "you trained
  the stronger direction" — the confound no MT cell can resolve.
- [ ] **P3 — seeds 42 and 1234 on the four core cells · 43 GPU-h · +1.9 d**
  Error bars. Nothing in the paper should quote a single-seed number.
- [ ] **P4 — mechanism on the collapsed cells · 14 GPU-h · +0.7 d**
  `relearn{10,50,200,1000}` and the elicitation ladder: **erased or merely suppressed?** This is
  the paper's most distinctive section and nothing has run for it yet.
- [ ] **P5 — `gemma3-4b`, seed 17, 7 cells · 33 GPU-h · +1.5 d** — generality across families
- [ ] **P6 — `olmo2-1b`, seed 17, 7 cells · 16 GPU-h · +0.7 d** — the contamination-audit model
- [ ] **P7 — large tier, `llama31-8b` + `gemma3-12b` · 116 GPU-h · +5.4 d** — does it survive scale
- [ ] **P8 — `fullft`, the RQ3 ladder, `exec`/`coverage` · 130 GPU-h · +6.1 d**
  `fullft_*` matters more than its cost suggests: everything so far is LoRA, and Biderman et al.
  report LoRA forgets *less*, so our magnitudes may be conservative.

**Cumulative: P1–P4 is 82 GPU-h (~4 days at current throughput) and would support a complete
paper at one model.** P5–P8 is breadth and scale.

---

## 6. Not yet built

- [ ] Spectral repair (mech exp 7) **for full fine-tuning deltas** — the LoRA path is a weaker
      instrument by construction and the module says so; a full-FT delta is what the method was
      designed for
- [ ] `d2t` gate with a walltime that fits two resident models

---

## 7. Known limits to state in the paper

- `relation`'s bootstrap resamples **60 countries**, not 135 rows — detects a large collapse,
  supports **no** equivalence claim
- `mix50 − replay` is an **upper bound**: `replay` carries ~0.70× `sft`'s supervised tokens and
  is therefore the weaker control (Amendment 22)
- `mix50 − flip` came back **inconclusive** at n≈1,000, exactly as the power analysis predicted.
  The equivalence claim needs the 2,500-instance domains, which are in P2/P8
- `mixedtask` confounds mixture with quantity by construction — it carries one fifth the cell's
  data
- **`/scratch` has no confirmed retention policy.** `trials.jsonl` is mirrored to `/work` after
  every eval, but the policy question is still open with the admins
