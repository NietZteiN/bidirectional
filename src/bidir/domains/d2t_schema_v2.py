"""Schema-guided frozen extractor diagnostic; original WebNLG result stays intact."""
from functools import lru_cache
import json
import re
import unicodedata

from bidir.config import DATA_DIR
from bidir.domains import d2t as original
from bidir.domains._common import base_row
from bidir.domains.d2t import instruction, augmented_hint

NAME = CELL = 'd2t_schema_v2'
CELL_OPTS = {}


def build_pairs(cfg):
    splits = original.build_pairs(cfg)
    for rows in splits.values():
        for pair in rows:
            pair.domain = NAME
            pair.pair_id = NAME+'::'+pair.pair_id
    return splits


@lru_cache(maxsize=1)
def schema():
    document = json.loads((DATA_DIR/NAME/'extractor_schema.json').read_text())
    return document['relations']


def canonical_term(term):
    return ''.join(c for c in unicodedata.normalize('NFKD', original._norm_term(term))
                   if not unicodedata.combining(c))


def parse_triples(text):
    return {(canonical_term(s), re.sub(r'\s+', '', p), canonical_term(o))
            for s, p, o in original.parse_triples(text)}


def content_key(pair):
    return json.dumps(sorted(parse_triples(pair.side_a)),ensure_ascii=False)


def roundtrip_triples(texts, cfg):
    from bidir.engine import generate_with
    contract = ('Extract EVERY fact stated in the text, including implicit relation wording. '
                'Return only subject | predicate | object lines. Use these exact predicate '
                'labels, choosing their usual meaning; do not invent labels or add facts:\n'
                +', '.join(schema())+'\nPreserve entity names and literal values from the text.\nText:\n')
    outputs = generate_with(cfg['roundtrip_extractor'], [contract+text for text in texts],
                            max_tokens=int(cfg['roundtrip_max_tokens']),
                            util_override=cfg['roundtrip_gpu_memory_utilization'])
    return [parse_triples(output) for output in outputs]


def score_batch(direction, outputs, insts, cfg):
    import sacrebleu
    rows = [base_row(output, inst['side_a' if direction=='forward' else 'side_b'],
                     inst['side_b' if direction=='forward' else 'side_a'])
            for output, inst in zip(outputs, insts)]
    extracted = ([parse_triples(output) for output in outputs] if direction=='reverse'
                 else roundtrip_triples(outputs, cfg))
    chrf = sacrebleu.CHRF(word_order=2)
    for row, output, inst, got in zip(rows, outputs, insts, extracted):
        want = parse_triples(inst['side_a'])
        row['off_target'] = int(not got) if direction=='reverse' else int(bool(original.parse_triples(output)))
        row['exact_triple_set'] = int(bool(want) and got==want)
        row['strict'] = int(row['exact_triple_set'] and not row['echo'] and not row['empty_output'] and not row['off_target'])
        row['criterion'] = 'schema-v2 exact normalized triple set; no echo; frozen train-only schema'
        if direction=='forward':
            row['chrf2'] = chrf.sentence_score(output, (inst.get('meta') or {}).get('all_references') or [inst['side_b']]).score
    return rows
