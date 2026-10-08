"""A disabled pool-share guard must keep the gate DAG executable."""
import importlib.util
from pathlib import Path


def test_zero_share_does_not_index_an_empty_dependency_chain(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/slurm/pipeline_gate.py'
    spec=importlib.util.spec_from_file_location('gate_share_test',path)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    calls=[]
    def submit(name,argv,**kwargs):
        calls.append((name,kwargs))
        return str(100+len(calls))
    monkeypatch.setenv('BIDIR_JUNO_SHARE','0')
    monkeypatch.setattr(gate,'sub',submit)
    monkeypatch.setattr('sys.argv',['pipeline_gate.py','--dry-run'])
    assert gate.main()==0
    trains=[kwargs for name,kwargs in calls if name.startswith('tr_')]
    assert len(trains)==4
    assert all(kwargs.get('dep') is None for kwargs in trains)
    evals=[kwargs for name,kwargs in calls if name.startswith('ev_')]
    assert len(evals)==4 and all(kwargs['dep'] for kwargs in evals)
