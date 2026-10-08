import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('followup',Path(__file__).resolve().parents[1]/'scripts/102_followup_analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def rows():
    return [dict(system=s,direction=d,strategy='simple',pair_id=f'p{i}',cluster_id=f'p{i}',strict=i%2)
            for s in ['base','fullft_sft'] for d in ['forward','reverse'] for i in range(4)]

def test_full_weight_coverage_and_cluster_checks():
    data=rows();analysis.validate(data,{'base','fullft_sft'},4)
    with pytest.raises(ValueError,match='duplicate'):analysis.validate(data+[data[0]],{'base','fullft_sft'},4)
    with pytest.raises(ValueError,match='coverage'):analysis.validate(data[:-1],{'base','fullft_sft'},4)
    changed=[dict(r) for r in data];changed[-1]['cluster_id']='foreign'
    with pytest.raises(ValueError,match='cluster'):analysis.validate(changed,{'base','fullft_sft'},4)
