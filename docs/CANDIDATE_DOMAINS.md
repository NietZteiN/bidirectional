# Which candidate domains this paper should actually run

*Written 2026-09-10, against [`TASK_CATALOGUE.md`](TASK_CATALOGUE.md)'s menu of 80.*

The catalogue is a menu for a course; this is a recommendation for **one paper with a fixed
budget and a decision gate it has not yet passed**. The two are different problems. A paper is
not improved by breadth per se — it is improved by domains that close a specific hole in its
argument, and made worse by domains that add noise, cost, or an unfalsifiable criterion.

So the question for each candidate is not "is this interesting?" but: **what does the paper
currently fail to establish that this would establish?**

---

## What is already covered

| catalogue | this project | label |
|---|---|---|
| **#22** output ↔ input prediction | `exec` (CRUXEval) | `L` |
| **#38** SQL ↔ natural language | `sql` (Spider) | `S` |
| **#65** machine translation | `mt_{en-de,de-en,en-zh,zh-en}` | `S` |
| **#44** JSON ↔ XML ↔ YAML | `fmt` | `B` |
| — data-to-text | `d2t` (WebNLG) | `S` |
| — code obfuscation | `code` (5 transforms) | `L` + renaming |
| — synthetic invertibility ladder | `fmt_det{75,50,25,00}` | constructed |
| — never-had control | `fmt_novel` | constructed |

**Two catalogue entries are already inside `code` and cost nothing to report.** obtune's
condition ladder includes `L2`, sequential minification — which is **#4** (minification ↔
beautification) as a within-domain subtask — and `L1b`/`L1r`, adversarial and random renaming.
The paper can report the structural (`S1`, `S2`) versus renaming (`L1b`, `L1r`, `L2`) contrast
as a within-domain invertibility test **for free**, because those cells are already in the
grid. That contrast is currently listed under RQ3 and should be named in the paper as the
natural-domain counterpart to the synthetic ladder.

---

## The three holes worth filling

Each of these is a domain the paper's argument needs, not a domain that would be nice to have.
Each costs roughly **117 adapter-units ≈ 23 GPU-h** at the small tier (3 models × 3 seeds ×
11 arms), which is the real price and should be weighed against the ~390 GPU-h total.

### 1. A provably hard inverse — **#30, cellular automata** `L`

**The hole.** RQ3 claims the recoverable ceiling tracks invertibility. The evidence is a
*constructed* ladder (`fmt_det*`) and an *empirical* observation (renaming recovers less than
structural transforms). Neither is a case where the inverse is hard **for a reason that can be
stated independently of the measurement**. A reviewer can ask whether the ladder's floor is a
property of information or of the model, and right now the honest answer is "we constructed it
that way".

**What it buys.** Finding a Game of Life predecessor is NP-hard in general, and a large fraction
of states are *Gardens of Eden* — they have no predecessor at all. That is a ceiling derived
from the problem, not from the corpus. The prediction becomes sharp and pre-registerable: no
reverse dose moves the ceiling on Garden-of-Eden states, and the achievable ceiling on the rest
is bounded by predecessor density, which can be computed per instance at build time exactly as
`determinable` already is for `fmt_det*`.

**Cost.** Low. Pure generation, no external data, exact verifier (simulate one step and
compare). The `determinable` flag comes free from a brute-force predecessor search on small
boards. Fits the existing `PairInstance` schema with no new machinery.

**Risk.** The base model may be at floor in *both* directions, which fails the base gate and
makes the cell uninformative. Mitigate by sizing boards small (5×5 to 8×8) and checking the
base rate before committing the grid — the gate protocol already does exactly this.

### 2. A formal-reasoning domain — **#56, expansion ↔ factorization** `L`

**The hole.** The paper claims directional collapse in "paired NLP tasks" and tests translation,
semantic parsing, data-to-text, code and format conversion. It has **no formal or mathematical
domain**, which is the first thing a reviewer will name as a gap in a generality claim.

**What it buys.** Expansion is mechanical and factorization is search — a genuine, well-known
asymmetry that is not about surface form. It also connects the paper to Lample and Charton
(2020) on symbolic mathematics with transformers, which is the closest prior work on
direction-asymmetric formal tasks and is currently uncited.

**Why #56 and not #55 (differentiation ↔ integration).** Integration's failure mode is
contaminated: a great many elementary functions have no elementary antiderivative, so "the model
failed" and "no answer exists" are entangled unless the generator is carefully restricted.
Polynomial factorization over the integers is decidable, the verifier is a single SymPy call,
and every instance has a determined answer. #55 remains the better-known task and is worth a
paragraph as a variant, but #56 is the better *measurement*.

**Cost.** Lowest of the three. SymPy 1.14 is already installed. Generator and verifier are
perhaps a hundred lines; the domain module is the same shape as `fmt`.

### 3. A natural task whose forward direction is trivial — **#66, diacritic restoration** `L`

**The hole.** In every current domain the forward direction is itself a real task the model is
learning. That leaves a confound the paper cannot fully answer: does forward-only training
destroy the inverse, or does it merely *spend capacity* on a hard forward task?

**What it buys.** Stripping diacritics is a deterministic character map — nothing to learn.
So a collapse in the reverse direction cannot be explained by the forward task consuming the
model. This is the cleanest available isolation of the phenomenon, and it is the domain where a
null would be most damaging to the headline claim, which is exactly what makes it worth running.

**Cost.** Low. Unlimited paired data from any Unicode corpus, exact-match verifier, no external
benchmark to license. Vietnamese has the densest diacritics and the strongest published
baselines; French is the safer high-resource choice.

---

## What to leave out, and why

- **Weak verifiers — #12, #72, #73, and #32 in its summarisation direction.** Noisy grading
  blurs the effect being measured. The workshop paper's hardest lesson was that a permissive
  criterion produced a number with near-zero lift over unconditioned output; every domain here
  earns its place by having a criterion that can be wrong.
- **Expensive toolchains — #1 (compilation), #13 (cross-language translation), #21 (hardware),
  #58 (Lean).** Each needs a per-language build-and-test harness. #1 and #13 are the most
  scientifically attractive entries in the whole catalogue and are the right *next* paper; they
  are not affordable inside this one's budget, and a half-built test harness produces a
  criterion that measures the harness.
- **Anything needing a library that is not installed — #74 (RDKit), #78 (music21), #31
  (python-chess).** Not a real barrier, but not free either, and none of them closes a hole the
  three above do not.
- **#69, spelling and string reversal.** Genuinely interesting — it would say whether any of
  this is a tokenization artifact — but it is a different paper's question. Worth one sentence
  in Limitations rather than a grid cell.
- **More `S` domains.** The paper already has four. A fifth adds cost without adding an axis.

---

## Governance: how a domain may be added

The gate has not run. **No domain should be added before it does**, for two reasons: the gate
may send the paper to a boundary-conditions framing in which the domain list changes entirely,
and every domain added now is 23 GPU-h spent against a hypothesis that has not been checked.

When a domain is added, it must be declared in [`../PREREGISTRATION.md`](../PREREGISTRATION.md)
**before its results are seen** — a dated amendment naming the domain, its criterion, its
threshold procedure and its predicted direction. Adding domains and then reporting the ones that
worked is the failure mode this paper exists to criticise in others, and it would be visible in
the git history.

Recommended sequence:

1. Run the gate. Read `50_contrasts.py --gate`.
2. If it passes: build **#56** first — it is the cheapest and closes the most-cited gap.
3. Then **#66**, which is the strongest isolation of the phenomenon.
4. Then **#30**, if RQ3's ceiling claim is drawing reviewer weight and the synthetic ladder
   alone looks thin.
5. Report **#4** and the renaming-versus-structural contrast from the existing `code` cells —
   free, and already in the grid.

Each addition is a domain module implementing the five-function contract in
`src/bidir/domains/__init__.py`, a config, and a `10_build_domain.py` run. The harness — arms,
mixtures, budget matching, evaluation, contrasts, mechanism experiments — needs no changes.
