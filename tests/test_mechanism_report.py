import importlib.util
import json
from pathlib import Path

def module(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('mechanism_report',Path(__file__).resolve().parents[1]/'scripts/53_mech_report.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'RESULTS_DIR',tmp_path)
    return m

def test_relearning_controls_match_model_and_seed(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch)
    for model,seed,control in [('llama32-3b',17,.1),('gemma3-4b',42,.8)]:
        for domain,value in [('fmt_novel',control),('sql',.5)]:
            p=tmp_path/'date'/domain/model/f'relearn_s{seed}'/'trials.jsonl';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(direction='reverse',strategy='simple',system='relearn10',strict=value))+'\n')
    out=m.read_relearn()
    assert out['sql/llama32-3b/s17']['control']['relearn10']==.1
    assert out['sql/gemma3-4b/s42']['control']['relearn10']==.8
    assert out['sql/llama32-3b/s17']['outruns_never_had_at_k']==[10]
    assert out['sql/gemma3-4b/s42']['outruns_never_had_at_k']==[]

def test_forward_progress_is_undefined_without_gain(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch)
    p=tmp_path/'mech/alpha_scale/cell/curve.json';p.parent.mkdir(parents=True)
    rows=[dict(alpha=a,direction=d,strict=v) for a,d,v in
          [(0.,'forward',.7),(1.,'forward',.6),(0.,'reverse',.7),(1.,'reverse',.1)]]
    p.write_text(json.dumps(dict(domain='sql',model='llama32-3b',seed=17,rows=rows)))
    assert m.read_alpha()['sql/llama32-3b/s17']['forward_progress_there'] is None
