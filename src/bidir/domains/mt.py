"""Machine translation: en<->de and en<->zh.

Both directions are real tasks the base model can already do, and a reverse pair is free —
the same sentence pair read the other way. This is the domain where the phenomenon already
has a partial observation in the literature: Zhu et al. (2024, EMNLP) report that tuning on a
single direction generalizes EXCEPT when English is on the target side, which damages
translation into non-English. That asymmetry is why every MT cell is run in both training
directions from the start (plan §3 RQ2): if collapse appears for X->en training and not for
en->X, that is the moderator, not a failure.

Train: News Commentary v16 (WMT's own training corpus, ungated).
Eval:  FLORES-200 devtest, 1,012 sentences, via the ungated ALMA mirror `haoranxu/FLORES-200`.
       The canonical `openlanguagedata/flores_plus` is gated (auto-approve on request); the
       mirror carries the same devtest split and needs no access request.

Criteria (plan §4). Forward: target language correct AND COMET-22 >= tau. Reverse: the same,
AND not an echo. tau is frozen from the base model's own COMET distribution before any tuned
model is scored. chrF++, BLEU and raw COMET are reported beside the strict rate, because a
strict rate alone cannot distinguish "slightly worse" from "collapsed".
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from bidir.config import PROJECT_ROOT, load_config
from bidir.domains._common import base_row, is_echo, normalize, threshold_for
from bidir.schema import PairInstance

NAME = "mt"
CELL = "mt_en-de"
CELL_OPTS: dict[str, Any] = {"pair": "en-de", "forward": "en-de"}

LANG_NAME = {"en": "English", "de": "German", "zh": "Chinese"}
SCORE_ENV_PY = Path(os.environ.get("BIDIR_SCORE_ENV", "/work/jvl210002/migration/envs/bidir-score")) / "bin" / "python"


def _dirs() -> tuple[str, str]:
    """(source_lang, target_lang) of this cell's FORWARD direction."""
    src, tgt = CELL_OPTS["forward"].split("-")
    return src, tgt


def langs_for(direction: str) -> tuple[str, str]:
    src, tgt = _dirs()
    return (src, tgt) if direction == "forward" else (tgt, src)


# --------------------------------------------------------------------------- build

def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    """`side_a` is always this cell's forward SOURCE, `side_b` its forward TARGET, so
    mt_en-de and mt_de-en are genuinely different cells over the same sentence pairs."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import random

    src, tgt = _dirs()
    pair = CELL_OPTS["pair"]
    hf_pair = cfg["train_pairs"][pair]           # news_commentary's own config name, e.g. "de-en"
    n_train = int(cfg["n_train"])
    n_val = int(cfg["n_val"])

    ds = load_dataset(cfg["train_hf_id"], hf_pair, split="train")
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    lo, hi = int(cfg["min_chars"]), int(cfg["max_chars"])
    for i, ex in enumerate(ds):
        t = ex["translation"]
        a, b = (t.get(src) or "").strip(), (t.get(tgt) or "").strip()
        if not a or not b:
            continue
        if not (lo <= len(a) <= hi and lo <= len(b) <= hi):
            continue
        key = normalize(a)
        if key in seen:                           # News Commentary repeats sentences across years
            continue
        seen.add(key)
        rows.append((f"{pair}::nc::{i}", a, b))
    rng = random.Random(int(cfg.get("seed", 17)))
    rng.shuffle(rows)
    need = n_train + n_val
    if len(rows) < need:
        raise ValueError(f"{cfg['train_hf_id']}/{hf_pair} yielded {len(rows)} usable rows, need {need}")

    def mk(chunk, split):
        return [PairInstance(pair_id=pid, domain=NAME, subtask=CELL_OPTS["forward"], side_a=a,
                             side_b=b, split=split, meta={"src_lang": src, "tgt_lang": tgt})
                for pid, a, b in chunk]

    out = {"train": mk(rows[:n_train], "train"), "val": mk(rows[n_train:need], "val")}

    # Eval: FLORES-200 devtest. A separate corpus from training by construction, which is
    # what the "never in any training split" rule needs (plan §4 sizing).
    f = hf_hub_download(cfg["eval_hf_id"], f"{src}-{tgt}/test-00000-of-00001.parquet",
                        repo_type="dataset")
    col = f"{src}-{tgt}"
    recs = pq.read_table(f).to_pylist()
    test = [PairInstance(pair_id=f"flores::{src}-{tgt}::{i}", domain=NAME,
                         subtask=CELL_OPTS["forward"], side_a=r[col][src].strip(),
                         side_b=r[col][tgt].strip(), split="test",
                         meta={"src_lang": src, "tgt_lang": tgt})
            for i, r in enumerate(recs)]
    limit = int(cfg.get("n_test", 0)) or len(test)
    out["test"] = test[:limit]
    return out


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    src, tgt = langs_for(direction)
    text = inst["side_a"] if direction == "forward" else inst["side_b"]
    return (f"Translate the following {LANG_NAME[src]} text into {LANG_NAME[tgt]}.\n\n"
            f"{LANG_NAME[src]}:\n{text}")


def augmented_hint(direction: str) -> str:
    _, tgt = langs_for(direction)
    return (f"The output must be written entirely in {LANG_NAME[tgt]}. Do not copy the input, "
            f"and do not answer in any other language.")


# --------------------------------------------------------------------------- scoring

def _detect_lang(text: str, expect: str) -> bool:
    """Script test for zh (unambiguous), langdetect for the Latin-script pair."""
    t = normalize(text)
    if not t:
        return False
    if expect == "zh":
        han = sum(1 for ch in t if "一" <= ch <= "鿿")
        return han / max(1, len(t)) > 0.15
    if any("一" <= ch <= "鿿" for ch in t):
        return False                              # Han characters where none belong
    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        return any(c.lang.split("-")[0] == expect and c.prob > 0.5 for c in detect_langs(t))
    except Exception:
        return True                               # detector failure must not create a false negative


def _comet(srcs: Sequence[str], hyps: Sequence[str], refs: Sequence[str],
           model: str, batch_size: int = 64) -> list[float]:
    """COMET-22 in its own venv (it pins transformers<5, the training stack pins 5.14.1).

    A subprocess, not an import: one process boundary is the whole cost of keeping two
    incompatible transformers versions in one pipeline.

    THE CARD IS ALREADY BUSY. COMET is scored inside the same job that generated, while vLLM
    still holds `gpu_memory_utilization` of the device -- 0.85 here, so wmt22-comet-da (XLM-R
    large) and its batch have to fit in the remaining 15%. That is ~21 GB on an H200 and ~3.6 GB
    on an a30, where batch 64 does not fit. A CUDA OOM here would kill the job *after* all the
    generation work was done, so it falls back: halve the batch, then move to CPU. Slower is
    the right trade for a step that runs once per gate.
    """
    if not srcs:
        return []
    if not SCORE_ENV_PY.exists():
        raise FileNotFoundError(f"{SCORE_ENV_PY} missing — run env/setup_score_env.sh")
    payload = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as td:
        inp, outp = Path(td) / "in.json", Path(td) / "out.json"
        inp.write_text(json.dumps(payload))
        script = f'''
import json, sys, torch
from comet import download_model, load_from_checkpoint
data = json.load(open({str(inp)!r}))
m = load_from_checkpoint(download_model({model!r}))

# (batch, gpus) attempts, in order. vLLM holds most of the card, so OOM here is expected
# rather than exceptional, and the CPU attempt is the one that cannot fail for memory.
plans = [({batch_size}, 1), (max(1, {batch_size} // 4), 1), (16, 0)] if torch.cuda.is_available() else [(16, 0)]
last = None
for bs, gpus in plans:
    try:
        out = m.predict(data, batch_size=bs, gpus=gpus, progress_bar=False)
        print(f"[comet] scored with batch_size={{bs}} gpus={{gpus}}", file=sys.stderr)
        json.dump(list(out["scores"]), open({str(outp)!r}, "w"))
        break
    except torch.cuda.OutOfMemoryError as exc:
        last = exc
        print(f"[comet] OOM at batch_size={{bs}} gpus={{gpus}}, retrying smaller", file=sys.stderr)
        torch.cuda.empty_cache()
else:
    raise SystemExit(f"COMET could not fit anywhere, last OOM: {{last}}")
'''
        r = subprocess.run([str(SCORE_ENV_PY), "-c", script], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"COMET scoring failed:\n{r.stderr[-2000:]}")
        # stderr carries which plan won; surface it so a silent fall to CPU is visible in the log.
        for line in r.stderr.splitlines():
            if line.startswith("[comet]"):
                print(line, flush=True)
        return json.loads(outp.read_text())


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    import sacrebleu

    src_lang, tgt_lang = langs_for(direction)
    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]

    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]
    zh = tgt_lang == "zh"
    chrf = sacrebleu.CHRF(word_order=2)           # chrF++
    bleu = sacrebleu.BLEU(tokenize="zh" if zh else "13a", effective_order=True)
    for row, o, t in zip(rows, outputs, targets):
        row["chrf2"] = chrf.sentence_score(o or "", [t]).score
        row["bleu"] = bleu.sentence_score(o or "", [t]).score
    for row, o in zip(rows, outputs):
        row["off_target"] = int(not _detect_lang(o, tgt_lang))

    comet_model = cfg.get("comet_model")
    if comet_model:
        scores = _comet(sources, [o or "" for o in outputs], targets, comet_model,
                        int(cfg.get("comet_batch_size", 64)))
        for row, sc in zip(rows, scores):
            row["comet"] = float(sc)

    tau = threshold_for(cfg, direction, "comet") if comet_model else threshold_for(cfg, direction, "chrf2")
    key = "comet" if comet_model else "chrf2"
    for row in rows:
        ok = (not row["empty_output"]) and not row["off_target"] and row[key] >= tau
        if direction == "reverse":
            ok = ok and not row["echo"]
        row["strict"] = int(ok)
        row["criterion"] = f"{key}>={tau:.4f} & on-target" + (" & not-echo" if direction == "reverse" else "")
    return rows
