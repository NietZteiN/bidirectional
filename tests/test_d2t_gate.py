import importlib.util
from pathlib import Path
from types import SimpleNamespace
from bidir.domains import d2t


def test_exact_d2t_gate_never_requests_reverse_chrf(monkeypatch,tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'15_base_gate.py'
    spec=importlib.util.spec_from_file_location('d2t_gate',path)
    gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    monkeypatch.setattr(gate,'RESULTS_DIR',tmp_path)
    monkeypatch.setattr(gate,'load_config',lambda _: {})
    inst=dict(pair_id='p',side_a='A | relation | B',side_b='A relates to B.',meta={},subtask='x')
    monkeypatch.setattr(gate,'load_pairs',lambda *_:[SimpleNamespace(model_dump=lambda:inst)])
    monkeypatch.setattr(gate.eng,'generate',lambda *a: (['answer'],[1]))
    monkeypatch.setattr(d2t,'score_batch',lambda direction,*a:[dict(strict=1,echo=0,off_target=0)])
    monkeypatch.setattr(d2t,'roundtrip_triples',lambda *a:[d2t.parse_triples(inst['side_a'])])
    args=SimpleNamespace(model='llama32-3b',limit=1,seed=17,tau_quantile=.25,
        dump_failures=False,min_base_rate=.1,max_format_fail=.15,max_echo_probe_strict=.02,write=False)
    report=gate.gate_one(args,'d2t',object())
    assert report['passes_gate'] and report['tau_metric'] is None
    assert report['roundtrip_ceiling']==1
