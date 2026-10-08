import json
from bidir.pipeline_state import atomic_json,evaluation_succeeded


def test_partial_failed_and_stale_eval_never_mark_complete(tmp_path):
    run=tmp_path/'run';run.mkdir();statuses=tmp_path/'status';statuses.mkdir()
    name='ev_fmt_llama32-3b_s42_contrastive_pilot'
    (run/'trials.jsonl').write_text('{"partial":true}\n')
    assert not evaluation_succeeded(run,statuses,name)
    atomic_json(run/'summary.json',{'finished_utc':'2026-10-06T12:00:00+00:00'})
    atomic_json(statuses/(name+'.1.json'),{'exit_code':1,'finished_utc':'2026-10-06T12:01:00+00:00'})
    assert not evaluation_succeeded(run,statuses,name)
    atomic_json(statuses/(name+'.2.json'),{'exit_code':0,'finished_utc':'2026-10-05T12:01:00+00:00'})
    assert not evaluation_succeeded(run,statuses,name)
    atomic_json(statuses/(name+'.3.json'),{'exit_code':0,'finished_utc':'2026-10-06T12:01:00+00:00'})
    (statuses/(name+'.broken.json')).write_text('{broken')
    assert evaluation_succeeded(run,statuses,name)


def test_atomic_state_replacement(tmp_path):
    path=tmp_path/'state.json'
    atomic_json(path,{'pass':1});atomic_json(path,{'pass':2})
    assert json.loads(path.read_text())=={'pass':2}
    assert not path.with_suffix('.json.tmp').exists()


def test_engineering_retry_proof_ignores_failures_and_pre_registration_jobs(tmp_path):
    from bidir.pipeline_state import registered_engineering_job
    entry = dict(job=100, name='cl_scale_smoke')
    for job, rc in [(99,0), (100,1), (101,0), (102,1)]:
        atomic_json(tmp_path / f"{entry['name']}.{job}.json", dict(exit_code=rc))
    (tmp_path / f"{entry['name']}.103.json").write_text('{broken')
    assert registered_engineering_job(tmp_path, entry) == 101
