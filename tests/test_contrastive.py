import pytest
import torch
from bidir.contrastive import masked_mean, symmetric_nce, negative_schedule


def test_pool_padding_and_gradients():
    h = torch.randn(2,3,5,requires_grad=True)
    mask = torch.tensor([[1,1,0],[1,0,0]])
    pooled = masked_mean(h,mask)
    changed = h.detach().clone(); changed[mask == 0] = 900
    torch.testing.assert_close(pooled,masked_mean(changed,mask))
    with pytest.raises(ValueError):
        masked_mean(h,torch.zeros_like(mask))
    # Positive correspondence should beat a mismatched positive index.
    a = torch.eye(4,requires_grad=True); b = torch.eye(4,requires_grad=True)
    cb = torch.stack([b, b.roll(1,0), b.roll(2,0)],1)
    ca = torch.stack([a, a.roll(1,0), a.roll(2,0)],1)
    good = symmetric_nce(a,b,cb,ca)
    bad = symmetric_nce(a,b,cb.roll(1,1),ca.roll(1,1))
    assert good < bad
    good.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_negative_schedule_filters_and_batch_independence():
    ids = [f'p{i}' for i in range(8)]
    semantic = ['same','same'] + ids[2:]
    aa = ['duplicate','duplicate'] + ids[2:]
    bb = ids[:]
    schedule = negative_schedule(ids,semantic,aa,bb,count=4)
    assert 1 not in schedule[0] and 0 not in schedule[1]
    perm = list(reversed(range(8)))
    reordered = negative_schedule(*[[xs[i] for i in perm] for xs in (ids,semantic,aa,bb)], count=4)
    original_ids = {ids[i]:[ids[j] for j in ns] for i,ns in enumerate(schedule)}
    new_ids = {ids[perm[i]]:[ids[perm[j]] for j in ns] for i,ns in enumerate(reordered)}
    assert original_ids == new_ids
    with pytest.raises(ValueError):
        negative_schedule(['a','b'],['same']*2,['a','b'],['a','b'])


def test_training_candidates_use_training_pool_and_shuffled_ablation():
    from bidir.contrastive import attach_candidates
    from bidir.schema import read_pairs
    from bidir.config import DATA_DIR
    rows = read_pairs(DATA_DIR / 'fmt' / 'val.jsonl')[:3]
    pool = read_pairs(DATA_DIR / 'fmt' / 'train.jsonl')
    pool_a = {p.side_a for p in pool}; pool_b = {p.side_b for p in pool}
    normal = attach_candidates([{}]*3,rows,'fmt',17)
    shuffled = attach_candidates([{}]*3,rows,'fmt',17,shuffled=True)
    for row,n,s in zip(rows,normal,shuffled):
        assert n['contrastive_a'][0] == row.side_a
        assert n['contrastive_b'][0] == row.side_b
        assert all(x in pool_a for x in n['contrastive_a'][1:])
        assert all(x in pool_b for x in n['contrastive_b'][1:])
        assert s['contrastive_a'] == n['contrastive_a']
        assert s['contrastive_b'] == n['contrastive_b'][1:]+n['contrastive_b'][:1]


def test_contrastive_no_silent_fallback_and_zero_lambda(monkeypatch):
    from bidir.losses import ContrastiveTrainer
    from trl import SFTTrainer
    trainer = object.__new__(ContrastiveTrainer)
    trainer.contrastive_lambda = 0
    with pytest.raises(RuntimeError,match='missing'):
        trainer.compute_loss(None,{'labels':torch.tensor([[1]])})
    ce = torch.tensor(1.25,requires_grad=True)
    sentinel = object()
    monkeypatch.setattr(SFTTrainer,'compute_loss',lambda *a,**kw:(ce,sentinel))
    batch = {'labels':torch.tensor([[1]])}
    for side in ('a','b'):
        for field in ('input_ids','attention_mask'):
            batch[f'contrastive_{side}_{field}'] = torch.ones(5,1,dtype=torch.long)
    result,out = trainer.compute_loss(None,batch,return_outputs=True)
    assert result is ce and out is sentinel


def test_nce_microbatch_loss_and_gradient_equivalence():
    torch.manual_seed(5)
    a = torch.randn(4,5,9,requires_grad=True)
    b = torch.randn(4,5,9,requires_grad=True)
    full = symmetric_nce(a[:,0],b[:,0],b,a)
    split = sum(symmetric_nce(a[i:i+1,0],b[i:i+1,0],b[i:i+1],a[i:i+1]) for i in range(4))/4
    torch.testing.assert_close(full,split)
    full_grad = torch.autograd.grad(full,(a,b),retain_graph=True)
    split_grad = torch.autograd.grad(split,(a,b))
    for f,s in zip(full_grad,split_grad):
        torch.testing.assert_close(f,s)


def test_units_candidates_exclude_equivalent_quantities_and_heldout_data(monkeypatch):
    from bidir.contrastive import attach_candidates
    from bidir.domains import units
    from bidir.schema import PairInstance
    splits = units.build_pairs(dict(n_train=24, n_val=4, n_test=4, seed=17))
    pool = splits['train']
    anchor = pool[0]
    value, unit = units.parse(anchor.side_a)
    alias = PairInstance(**{**anchor.__dict__, 'pair_id': 'alias',
                            'side_a': f'{value.numerator}/{value.denominator} {unit}'})
    pool = pool + [alias]
    monkeypatch.setattr('bidir.schema.read_pairs', lambda *args: pool)
    normal = attach_candidates([{}], [anchor], 'units', 17)[0]
    assert len(set(normal['contrastive_a'])) == 5
    assert alias.side_a not in normal['contrastive_a'][1:]
    assert anchor.side_b not in normal['contrastive_b'][1:]
    assert set(normal['contrastive_a'][1:]) <= {p.side_a for p in splits['train']}
    assert not set(normal['contrastive_a'][1:]) & {p.side_a for p in splits['val'] + splits['test']}
    monkeypatch.setattr('bidir.schema.read_pairs', lambda *args: list(reversed(pool)))
    assert normal == attach_candidates([{}], [anchor], 'units', 17)[0]
