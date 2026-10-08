import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('contrastive_analysis',Path(__file__).parents[1]/'scripts'/'55_contrastive_analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def trial_rows():
    return [dict(system=s,direction=d,strategy='simple',pair_id=f'p{i}',cluster_id=f'p{i}',
                 strict=int(i%2==0)) for s in analysis.REQUIRED for d in ('forward','reverse') for i in range(8)]


def test_same_pass_complete_pair_coverage_required():
    rows=trial_rows();analysis.validate_rows(rows,8)
    with pytest.raises(ValueError,match='incomplete'):
        analysis.validate_rows(rows[:-1],8)
    with pytest.raises(ValueError,match='duplicate'):
        analysis.validate_rows(rows+[rows[0]],8)
    changed=[dict(r) for r in rows];changed[0]['pair_id']='foreign'
    with pytest.raises(ValueError,match='same evaluation'):
        analysis.validate_rows(changed,8)
    changed=[dict(r) for r in rows];changed[0]['cluster_id']='wrong_cluster'
    with pytest.raises(ValueError,match='cluster'):
        analysis.validate_rows(changed,8)


def test_primary_interval_metadata_and_null_result(tmp_path):
    rows=trial_rows()
    (tmp_path/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(n_instances=8,domain='fmt',model='llama32-3b',seed=17,finished_utc='2026-10-03T00:00:00+00:00')
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    result=analysis.analyse_run(tmp_path,n_boot=500)
    primary=[r for r in result['contrasts'] if r['primary']]
    assert len(primary)==5 and len(result['contrasts'])==10
    assert result['same_pass_only'] and result['confidence']==0.99
    assert all(r['delta_pp']==0 and r['ci_lo']==0 and r['ci_hi']==0 for r in primary)
    assert all('verdict' not in r for r in primary)
    assert len(result['trial_sha256'])==64


def test_wider_confidence_uses_same_paired_draws():
    rows=trial_rows()
    for r in rows:
        if r['system']=='cl_fwd': r['strict']=1
    p95=analysis.paired.paired_delta(rows,'cl_fwd','sft','strict','reverse',n_boot=2000,confidence=.95)
    p99=analysis.paired.paired_delta(rows,'cl_fwd','sft','strict','reverse',n_boot=2000,confidence=.99)
    assert p95['delta_pp']==p99['delta_pp']==50
    assert p99['ci_lo']<=p95['ci_lo']<=p95['ci_hi']<=p99['ci_hi']
