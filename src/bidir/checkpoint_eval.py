"""Isolated-checkpoint generation for a single matched evaluation campaign."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from bidir import arms
from bidir.config import resolve_model


def generate_campaign(model, systems, reqs, engine_cfg, sampling, full_systems=None):
    base = resolve_model(model)['hf_id']
    groups = {base: []}
    for index, req in enumerate(reqs):
        name = req['system']
        full = name in full_systems if full_systems is not None else (name != 'base' and arms.resolve(name).full_ft)
        groups.setdefault(systems[name] if full else base, []).append(index)
    raw, tokens = [None] * len(reqs), [None] * len(reqs)
    reports = []
    for checkpoint, indices in groups.items():
        if not indices:
            continue
        full = checkpoint != base
        task = dict(checkpoint=checkpoint, model=model, engine=engine_cfg,
                    sampling={'temperature': 0., 'top_p': 1., 'max_tokens': 512,
                              'seed': 17, **sampling},
                    messages=[reqs[i]['messages'] for i in indices],
                    adapters=[None if full else reqs[i]['adapter'] for i in indices])
        with tempfile.TemporaryDirectory(prefix='bidir_checkpoint_') as tmp:
            inp, out = Path(tmp)/'input.json', Path(tmp)/'output.json'
            inp.write_text(json.dumps(task))
            subprocess.run([sys.executable, '-m', 'bidir.checkpoint_worker',
                            '--input', str(inp), '--output', str(out)], check=True)
            result = json.loads(out.read_text())
        if len(result['raw']) != len(indices) or len(result['tokens']) != len(indices):
            raise RuntimeError('checkpoint worker returned incomplete request coverage')
        for i, text, count in zip(indices, result['raw'], result['tokens']):
            raw[i], tokens[i] = text, count
        reports.append(dict(checkpoint=checkpoint, n_requests=len(indices),
                            systems=sorted({reqs[i]['system'] for i in indices}),
                            rendered_sha256=result['rendered_sha256'],
                            engine_version=result['engine_version']))
    return raw, tokens, reports
