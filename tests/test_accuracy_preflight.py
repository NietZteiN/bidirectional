"""Catch inaccurate result presentation without changing experiments or regenerating scores."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('accuracy_preflight_test',ROOT/'scripts/123_accuracy_preflight.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_replay_excludes_secondary_strategy_and_rejects_duplicates(tmp_path):
    path=tmp_path/'trials.jsonl'
    row=dict(system='base',direction='forward',pair_id='p',strict=1,strategy='simple')
    path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,strategy='cot',strict=0))+'\n')
    means,groups=audit.replay(path)
    assert means['base']['forward']==1 and len(groups[('base','forward')])==1
    with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='duplicate'):audit.replay(path)


def test_replay_rejects_nonbinary_success(tmp_path):
    path=tmp_path/'trials.jsonl'
    path.write_text(json.dumps(dict(system='base',direction='forward',pair_id='p',strict=.5))+'\n')
    with pytest.raises(ValueError,match='nonbinary'):audit.replay(path)


def test_replayed_means_detect_swapped_direction_values():
    with pytest.raises(ValueError,match='incorrect mean'):
        audit.require_means({'base':{'forward':1,'reverse':0}},
                            {'base':{'forward':0,'reverse':1}},'swapped')


def objective_grid(model='L3B',forward='100.0',reverse='0.0',missing=r'\textit{n/a}'):
    return ' & '.join(['SQL',model,'F',forward,missing])+r' \\'+'\n'+ \
           ' & '.join(['','','B',reverse,missing])+r' \\'+'\n'


def test_grid_detects_wrong_model_and_direction_scores():
    cells=[dict(domain='sql',model='llama',run='run')]
    raw={'run':{'base':{'forward':1.,'reverse':0.}}}
    args=(cells,['base','cft'],raw,{'sql':'SQL'},{'llama':'L3B'})
    assert audit.verify_grid_text(objective_grid(),*args)==2
    with pytest.raises(ValueError,match='model label'):
        audit.verify_grid_text(objective_grid(model='L8B'),*args)
    with pytest.raises(ValueError,match='printed score'):
        audit.verify_grid_text(objective_grid(forward='0.0',reverse='100.0'),*args)


def test_grid_does_not_treat_inapplicable_cft_as_a_missing_measurement():
    cells=[dict(domain='sql',model='llama',run='run')]
    raw={'run':{'base':{'forward':1.,'reverse':0.}}}
    with pytest.raises(ValueError,match='not-applicable'):
        audit.verify_grid_text(objective_grid(missing='--'),cells,['base','cft'],raw,
                               {'sql':'SQL'},{'llama':'L3B'})


def test_grid_does_not_confuse_highlight_color_with_the_score():
    cells=[dict(domain='sql',model='llama',run='run')]
    raw={'run':{'base':{'forward':1.,'reverse':0.}}}
    colored=r'\textcolor{blue!60!black}{\textbf{100.0}}'
    assert audit.verify_grid_text(objective_grid(forward=colored),cells,['base','cft'],raw,
                                  {'sql':'SQL'},{'llama':'L3B'})==2
