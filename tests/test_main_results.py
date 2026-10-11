"""Integrity checks for the compact main tables, not claims from pooled campaigns."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('main_results_test',ROOT/'scripts/117_main_results.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def campaign(tmp_path, base_reverse=1):
    rows = [dict(system=arm,direction=d,pair_id=str(i),strict=(base_reverse if arm=='base' and d=='reverse' else i%2),strategy='simple')
            for arm in ['base','sft','mix1','mix5','fwd2x'] for d in ['forward','reverse'] for i in range(2)]
    trials = tmp_path/'trials.jsonl'
    trials.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary = dict(arms=['base','sft','mix1','mix5','fwd2x'],n_instances=2,
        adapter_effectiveness={a:dict(identical_rate=.1) for a in ['sft','mix1','mix5','fwd2x']})
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    return rows,summary


def test_main_rates_require_complete_pairs_for_every_control(tmp_path):
    rows,summary = campaign(tmp_path)
    rows.pop()  # Missing backward coverage in the longer-forward control must reject all.
    (tmp_path/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='incomplete arm/direction: fwd2x'):
        report.campaign_rates(tmp_path,report.sha(tmp_path/'trials.jsonl'),{})


def test_main_rates_reject_changed_withdrawn_and_ineffective_control(tmp_path):
    rows,summary = campaign(tmp_path)
    digest = report.sha(tmp_path/'trials.jsonl')
    with pytest.raises(ValueError,match='changed trials'):
        report.campaign_rates(tmp_path,'0'*64,{})
    (tmp_path/'WITHDRAWN.txt').write_text('withdrawn')
    with pytest.raises(ValueError,match='withdrawn'):
        report.campaign_rates(tmp_path,digest,{})
    (tmp_path/'WITHDRAWN.txt').unlink()
    summary['adapter_effectiveness']['fwd2x']['identical_rate'] = 1
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='ineffective/unverified arm: fwd2x'):
        report.campaign_rates(tmp_path,digest,{})


def test_main_rates_reject_duplicate_trials(tmp_path):
    rows,_ = campaign(tmp_path)
    with (tmp_path/'trials.jsonl').open('a') as f:
        f.write(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='duplicate'):
        report.campaign_rates(tmp_path,report.sha(tmp_path/'trials.jsonl'),{})


def test_distinct_passes_keep_their_own_base_and_source_hash(tmp_path):
    left = tmp_path/'left';right = tmp_path/'right'
    left.mkdir();right.mkdir()
    campaign(left,1);campaign(right,0)
    sources = {}
    a=report.campaign_rates(left,report.sha(left/'trials.jsonl'),sources)
    b=report.campaign_rates(right,report.sha(right/'trials.jsonl'),sources)
    assert a['base']['reverse']==1 and b['base']['reverse']==0
    assert len(sources)==4
    assert sources[str(left/'trials.jsonl')]!=sources[str(right/'trials.jsonl')]


@pytest.mark.parametrize('change',['trials','withdrawal'])
def test_cached_table_does_not_hide_later_evidence_changes(tmp_path,monkeypatch,change):
    paper=tmp_path/'paper';(paper/'tables').mkdir(parents=True)
    (tmp_path/'runs/status').mkdir(parents=True)
    cells=[];sources={}
    for domain in set(report.TASKS)-{'units'}:
        run=tmp_path/'results'/domain/'small_s17';run.mkdir(parents=True)
        campaign(run)
        cells.append(dict(domain=domain,model='llama32-3b',seed=17,run=str(run)))
        sources[str(run/'trials.jsonl')]=report.sha(run/'trials.jsonl')
    (paper/'EVIDENCE_SNAPSHOT.json').write_text(json.dumps(dict(core_cells=cells,
        contrastive_reports=[],followup_reports=[],sources=sources)))
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'PAPER',paper)
    monkeypatch.setattr(report,'evaluation_succeeded',lambda *args:True)
    assert report.main()==0
    run=Path(cells[0]['run'])
    if change=='withdrawal':
        (run/'WITHDRAWN.txt').write_text('withdrawn after table generation')
    else:
        with (run/'trials.jsonl').open('a') as f:
            f.write('{}\n')
    with pytest.raises(ValueError,match='reference panel incomplete'):
        report.main()
