from bidir.domains import units

def test_units_gold_and_rejections():
    splits = units.build_pairs({'n_train':20,'n_val':4,'n_test':8,'seed':17})
    for direction in ['forward','reverse']:
        insts = [p.model_dump() for p in splits['test']]
        target = 'side_b' if direction=='forward' else 'side_a'
        source = 'side_a' if direction=='forward' else 'side_b'
        assert all(r['strict'] for r in units.score_batch(direction,[i[target] for i in insts],insts,{}))
        for outputs in [[i[source] for i in insts],['']*8,['garbage']*8,['0 wrong']*8]:
            assert not any(r['strict'] for r in units.score_batch(direction,outputs,insts,{}))

def test_exact_parsing_and_split_identity():
    assert units.parse('1/2 kg') == units.parse('0.500 kg')
    assert units.parse('-32 C')[0] == -32
    assert units.parse('1/0 m') is None
    assert units.parse('1 m extra') is None
    splits = units.build_pairs({'n_train':200,'n_val':20,'n_test':40,'seed':17})
    keys = [units.content_key(p) for rows in splits.values() for p in rows]
    assert len(set(keys)) == len(keys)
