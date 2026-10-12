"""Scientific budget, split safety, negative validity and bounded admission checks."""
import importlib.util
import json
from pathlib import Path
import pytest
from bidir.schema import PairInstance, TrainRow

ROOT=Path(__file__).resolve().parents[1]


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


worker=module('pj_test_worker','124_paired_judgment_worker.py')
queue=module('pj_test_queue','125_paired_judgment_queue.py')


def pool():
    return [PairInstance(pair_id=str(i),domain='mt',subtask='en-de',side_a='source '+str(i),
                         side_b='target '+str(i),split='train') for i in range(80)]


def rows():
    return [TrainRow.from_pair(p,'rev' if i<20 else 'fwd') for i,p in enumerate(pool())]


def test_replacement_matches_budget_and_shuffling_changes_only_labels():
    normal=worker.judgment_rows(rows(),pool(),'mt_en-de',17)
    shuffled=worker.judgment_rows(rows(),pool(),'mt_en-de',17,True)
    assert len(normal)==len(shuffled)==80
    assert sum(r.task=='fwd' for r in normal)==60
    assert sum(r.task=='pos' for r in normal)==sum(r.task=='neg' for r in normal)==10
    assert sum(r.task=='pos' for r in shuffled)==sum(r.task=='neg' for r in shuffled)==10
    assert not any(r.task=='rev' for r in normal+shuffled)
    assert {r.pair_id for r in normal if r.task!='fwd'}=={r.pair_id for r in rows() if r.task=='rev'}
    assert [r.model_dump(exclude={'task'}) for r in normal]==[r.model_dump(exclude={'task'}) for r in shuffled]
    assert any(a.task!=b.task for a,b in zip(normal,shuffled))


def test_negative_pool_rejects_validation_or_test_examples():
    for split in ['val','test']:
        bad=pool();bad[0]=bad[0].model_copy(update={'split':split})
        with pytest.raises(ValueError,match='training pairs only'):
            worker.judgment_rows(rows(),bad,'mt_en-de',17)


def test_negatives_exclude_known_alternative_targets_for_the_same_source():
    candidates=pool()
    candidates += [p.model_copy(update={'pair_id':'alias-'+p.pair_id,'side_b':'alias target '+p.pair_id}) for p in pool()]
    result=worker.judgment_rows(rows(),candidates,'mt_en-de',17)
    for row in result:
        if row.task=='neg':
            assert row.meta['judgment_candidate'] not in {row.side_b,'alias target '+row.pair_id}
            assert row.meta['judgment_candidate_split']=='train'


def test_sql_negatives_stay_in_context_and_reject_equivalent_queries():
    candidates=[p.model_copy(update={'domain':'sql','subtask':'db'+str(int(p.pair_id)%2),
                                    'meta':{'db_id':'db'+str(int(p.pair_id)%2),'schema':'test'}}) for p in pool()]
    selected=[TrainRow.from_pair(p,'rev' if i<20 else 'fwd') for i,p in enumerate(candidates)]
    checked=[]
    def check(source,candidate,timeout):
        assert source.subtask==candidate.subtask
        checked.append(candidate.pair_id)
        return int(candidate.pair_id)%4 in (0,1)
    result=worker.judgment_rows(selected,candidates,'sql',17,checker=check)
    assert checked
    assert all(int(r.meta['judgment_candidate_pair_id'])%4 in (0,1) for r in result if r.task=='neg')
    with pytest.raises(ValueError,match='no checked negative'):
        worker.judgment_rows(selected,candidates,'sql',17,checker=lambda *args:False)


def test_judgment_prompt_uses_binary_target_and_preserves_quoted_context():
    row=worker.judgment_rows(rows(),pool(),'mt_en-de',17)[0]
    row=row.model_copy(update={'task':'pos','side_a':'a "quoted" sentence'})
    example=worker.judgment_example(row.task,row.model_dump())
    assert example['completion']==[{'role':'assistant','content':'Yes'}]
    data=json.loads(example['prompt'][-1]['content'].split('\n',1)[1])
    assert data['source']==row.side_a
    assert data['candidate']==row.meta['judgment_candidate']


def test_sql_scoring_errors_cannot_create_negatives_and_connections_close(monkeypatch):
    from bidir.domains import sql
    class Connection:
        closed=False
        def execute(self,*args):raise worker.sqlite3.OperationalError('invalid query')
        def close(self):self.closed=True
    con=Connection()
    monkeypatch.setattr(worker.sqlite3,'connect',lambda *args,**kwargs:con)
    monkeypatch.setattr(sql,'db_path',lambda db:'test.sqlite')
    pair=pool()[0].model_copy(update={'meta':{'db_id':'db'}})
    assert worker.sql_negative(pair,pool()[1],3) is False
    assert con.closed


def test_one_phase_per_cell_and_no_retry_of_deterministic_failures(tmp_path,monkeypatch):
    cell=queue.worker.plan()['cells'][0]
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    status=tmp_path/'runs/status';status.mkdir(parents=True)
    monkeypatch.setattr(queue.worker,'ready_phase',lambda c:'train')
    train=queue.worker.job_name(cell,'train')
    assert queue.next_action(cell,{queue.worker.job_name(cell,'smoke'):'1'}, {})=='inflight'
    assert queue.next_action(cell,{}, {train:['1','2','3']}).startswith('blocked: attempts')
    (status/(train+'.1.json')).write_text(json.dumps({'exit_code':1}))
    assert queue.next_action(cell,{},{}).startswith('blocked: failure')


def test_admission_rejects_incomplete_or_stale_preflight(monkeypatch):
    monkeypatch.setattr(queue.worker,'signature',lambda cell:'frozen')
    proof={'preflight_sha256':queue.worker.sha(ROOT/'scripts/127_paired_judgment_preflight.py'),
           'rows':[dict(cell,arm=arm,signature='frozen') for cell in queue.worker.plan()['cells']
                   for arm in queue.worker.plan()['new_arms']]}
    queue.validate_preflight(proof)
    with pytest.raises(ValueError,match='incomplete'):
        queue.validate_preflight({**proof,'rows':proof['rows'][:-1]})
    proof['rows'][0]['signature']='changed'
    with pytest.raises(ValueError,match='stale data'):
        queue.validate_preflight(proof)


def test_production_training_waits_for_current_smoke(monkeypatch):
    cell=worker.plan()['cells'][0]
    class Gate:
        def gate_passed(self,*args):return True
        def adapter_complete(self,*args):return True
    monkeypatch.setattr(worker,'module',lambda *args:Gate())
    monkeypatch.setattr(worker,'completed',lambda c,p:False)
    assert worker.ready_phase(cell)=='smoke'
    monkeypatch.setattr(worker,'completed',lambda c,p:p=='smoke')
    assert worker.ready_phase(cell)=='train'
    monkeypatch.setattr(worker,'completed',lambda c,p:p in ['smoke','train'])
    assert worker.ready_phase(cell)=='eval_smoke'
    monkeypatch.setattr(worker,'completed',lambda c,p:p in ['smoke','train','eval_smoke'])
    assert worker.ready_phase(cell)=='eval'


def test_report_omits_all_unmeasured_values(tmp_path,monkeypatch):
    reporter=module('pj_empty_report','126_paired_judgment_report.py')
    monkeypatch.setattr(reporter,'ROOT',tmp_path)
    (tmp_path/'paper/bidir/tables').mkdir(parents=True)
    monkeypatch.setattr(worker,'completed',lambda *args:False)
    # Other loaded modules resolve against the real repository, while publication uses tmp_path.
    monkeypatch.setattr(reporter,'module',lambda name,filename:worker if filename.startswith('124_') else module(name,filename))
    assert reporter.main()==0
    result=json.loads((tmp_path/'paper/bidir/PAIRED_JUDGMENT_RESULTS.json').read_text())
    assert result['complete']==0 and len(result['pending'])==4
    assert (tmp_path/'paper/bidir/tables/paired_judgment_results.tex').read_text()=='\n'
