import json
from pathlib import Path
import pytest

from bidir.checkpoint_eval import generate_campaign


def test_full_weights_are_separate_from_lora_requests(monkeypatch):
    calls = []
    def worker(argv, check):
        task = json.loads(Path(argv[argv.index('--input')+1]).read_text())
        calls.append(task)
        Path(argv[argv.index('--output')+1]).write_text(json.dumps(dict(
            raw=[f"{task['checkpoint']}:{m[0]['content']}" for m in task['messages']],
            tokens=[2]*len(task['messages']), engine_version='test', rendered_sha256='hash')))
    monkeypatch.setattr('bidir.checkpoint_eval.subprocess.run', worker)
    systems = {'base':None, 'sft':'/lora', 'fullft_sft':'/full'}
    reqs = [dict(system=n, adapter=p, messages=[dict(role='user',content=n)])
            for n,p in systems.items()]
    raw, tokens, reports = generate_campaign('llama32-3b', systems, reqs, {}, {})
    assert len(calls)==2
    assert calls[0]['adapters']==[None,'/lora']
    assert calls[1]['checkpoint']=='/full' and calls[1]['adapters']==[None]
    assert raw[2]=='/full:fullft_sft' and tokens==[2,2,2]
    assert reports[1]['systems']==['fullft_sft']


def test_worker_failure_propagates(monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr('bidir.checkpoint_eval.subprocess.run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        generate_campaign('llama32-3b', {'base':None},
                          [dict(system='base',adapter=None,messages=[])], {}, {})
