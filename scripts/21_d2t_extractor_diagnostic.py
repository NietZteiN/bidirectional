#!/usr/bin/env python
"""Save raw frozen-extractor oracle outputs; never changes D2T eligibility."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from bidir import engine
from bidir.config import DATA_DIR, RESULTS_DIR, load_config
from bidir.domains.d2t import parse_triples
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=8)
    args = parser.parse_args()
    cfg = load_config('domains/d2t.yaml')
    pairs = read_pairs(DATA_DIR / 'd2t/test.jsonl')[:args.n]
    probes = [('gold', p.side_b, p) for p in pairs]
    probes += [('echo', p.side_a, p) for p in pairs]
    probes += [('empty', '', p) for p in pairs]
    probes += [('garbage', 'The quick brown fox jumps over the lazy dog. This is not an answer to anything.', p)
               for p in pairs]
    messages = [('Extract the RDF triples this text expresses. Write one triple per line in '
                 'the form `subject | predicate | object`.\n\nText:\n' + text)
                for _, text, _ in probes]
    raw = engine.generate_with(cfg['roundtrip_extractor'], messages,
        max_tokens=int(cfg.get('roundtrip_max_tokens', 256)),
        util_override=cfg.get('roundtrip_gpu_memory_utilization'))
    if len(raw) != len(probes):
        raise ValueError('extractor output coverage differs')
    rows = []
    for (probe, text, pair), output in zip(probes, raw):
        got, want = parse_triples(output), parse_triples(pair.side_a)
        rows.append(dict(probe=probe, pair_id=pair.pair_id, text=text,
            expected_triples=sorted(want), raw_extractor_output=output,
            parsed_triples=sorted(got), exact=int(bool(want) and got == want)))
    report = dict(diagnostic_only=True, grants_eligibility=False,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_JOB_ID'), extractor=cfg['roundtrip_extractor'],
        config=cfg, implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rates={probe: sum(r['exact'] for r in rows if r['probe'] == probe) / len(pairs)
               for probe in ('gold', 'echo', 'empty', 'garbage')}, rows=rows)
    out = RESULTS_DIR / 'audit' / f'd2t_extractor_raw_{os.environ.get("SLURM_JOB_ID", "local")}.json'
    atomic_json(out, report)
    print(json.dumps(dict(out=str(out), rates=report['rates'])), flush=True)
    return 0


if __name__ == '__main__':
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    engine.shutdown_and_exit(rc)
