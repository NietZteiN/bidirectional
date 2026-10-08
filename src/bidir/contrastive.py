"""Validated building blocks for the planned contrastive baseline.

Side representations come from independent model encodings;
teacher-forced pair states are invalid.
"""
import hashlib
import random

import torch
import torch.nn.functional as F


def masked_mean(hidden, attention_mask):
    """Float32 mean pooling over non-padding content tokens, then normalization."""
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError('hidden states and content masks disagree')
    counts = mask.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError('empty content encoding')
    pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts
    return F.normalize(pooled, dim=-1)


def symmetric_nce(anchor_a, anchor_b, candidates_b, candidates_a, temperature=0.1):
    """Positive in candidate slot zero, fixed negatives in subsequent slots.

    Anchors: [batch, dim]; candidates: [batch, 1 + negatives, dim].
    Both independently encoded sides retain their gradients.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if anchor_a.shape != anchor_b.shape or anchor_a.ndim != 2:
        raise ValueError('anchor shape mismatch')
    expected = (anchor_a.shape[0], anchor_a.shape[1])
    for candidates in (candidates_a, candidates_b):
        if candidates.ndim != 3 or (candidates.shape[0], candidates.shape[2]) != expected:
            raise ValueError('candidate shape mismatch')
        if candidates.shape[1] < 2:
            raise ValueError('at least one negative required')
    labels = torch.zeros(anchor_a.shape[0], dtype=torch.long, device=anchor_a.device)
    def loss(anchors, candidates):
        scores = (F.normalize(anchors.float(), dim=-1).unsqueeze(1)
                  * F.normalize(candidates.float(), dim=-1)).sum(-1) / temperature
        return F.cross_entropy(scores, labels)
    return (loss(anchor_a, candidates_b) + loss(anchor_b, candidates_a)) / 2


def negative_schedule(pair_ids, semantic_keys, side_a_keys, side_b_keys, seed=17, count=4):
    """Fixed per-pair negatives independent of batch order, size, and refits.

    Call only on training pairs. Keys must be domain-canonicalized by the caller.
    Exclude equivalent pairs and duplicates on either side. Hash by pair id so
    rearranging the corpus leaves each anchor's negative identities unchanged.
    """
    n = len(pair_ids)
    if count < 1 or len(set(pair_ids)) != n:
        raise ValueError('positive negative count and unique pair ids required')
    if any(len(keys) != n for keys in (semantic_keys, side_a_keys, side_b_keys)):
        raise ValueError('key lengths disagree')
    order = sorted(range(n), key=lambda i: pair_ids[i])
    result = []
    for i, pid in enumerate(pair_ids):
        eligible = [j for j in order if semantic_keys[j] != semantic_keys[i]
                    and side_a_keys[j] != side_a_keys[i] and side_b_keys[j] != side_b_keys[i]]
        if len(eligible) < count:
            raise ValueError(f'insufficient distinct negatives for {pid}')
        digest = hashlib.sha256(f'{seed}:{pid}'.encode()).digest()
        result.append(random.Random(int.from_bytes(digest, 'big')).sample(eligible, count))
    return result


def attach_candidates(examples, rows, domain, seed, shuffled=False):
    """Attach isolated side texts; negatives always come from the training split."""
    import json
    from bidir.config import DATA_DIR
    from bidir.schema import read_pairs
    from bidir.domains.fmt import _parse
    pool = read_pairs(DATA_DIR / domain / 'train.jsonl')
    if domain not in {'fmt', 'units'} and not domain.startswith('mt_'):
        raise ValueError('contrastive semantic filter is only validated for fmt/MT/units')
    def semantic(p):
        if domain == 'fmt':
            return json.dumps(_parse(p.side_a,p.subtask.split('-')[0]),
                              sort_keys=True,separators=(',',':'))
        if domain == 'units':
            from bidir.domains.units import content_key
            return content_key(p)  # exact physical quantity, shared across unit representations
        return (p.side_a.strip(),p.side_b.strip())
    keys = [semantic(p) for p in pool]
    canonical = sorted(range(len(pool)), key=lambda i:pool[i].pair_id)
    result = []
    for ex,row in zip(examples,rows):
        anchor_key = semantic(row)
        rng = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{row.pair_id}'.encode()).digest(),'big'))
        selected = []
        def eligible(j):
            p=pool[j]
            return (j not in selected and keys[j] != anchor_key
                    and p.side_a.strip() != row.side_a.strip()
                    and p.side_b.strip() != row.side_b.strip())
        for _ in range(200):
            j = canonical[rng.randrange(len(canonical))]
            if eligible(j): selected.append(j)
            if len(selected) == 4: break
        if len(selected) != 4:
            for j in canonical:
                if eligible(j): selected.append(j)
                if len(selected) == 4: break
        if len(selected) != 4:
            raise ValueError('insufficient distinct training negatives')
        aa = [row.side_a] + [pool[j].side_a for j in selected]
        bb = [row.side_b] + [pool[j].side_b for j in selected]
        # Fixed non-identity cyclic mapping among this anchor's five candidates.
        if shuffled: bb = bb[1:] + bb[:1]
        result.append({**ex,'contrastive_a':aa,'contrastive_b':bb})
    return result
