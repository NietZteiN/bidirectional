#!/usr/bin/env python
"""Build a local anonymous evidence bundle and verify all packaged file hashes.

No publication occurs. Dataset/weight redistribution and historical-source gaps remain
recorded review items; model weights and authentication material are never copied.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import paper_suite as suite
from bidir.config import RESULTS_DIR,RUNS_DIR
from bidir.pipeline_state import atomic_json

BUNDLE=ROOT/'paper/artifact'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def destination(path):
    for prefix,namespace in ((RESULTS_DIR,'evidence/results'),(RUNS_DIR,'evidence/runs'),(ROOT,'source')):
        if path.is_relative_to(prefix):return Path(namespace)/path.relative_to(prefix)
    raise ValueError('unexpected artifact outside registered roots')


def redact(value):
    replacements=[(str(RESULTS_DIR),'evidence/results'),(str(RUNS_DIR),'evidence/runs'),
        (str(ROOT),'source'),(str(ROOT.parent),'<WORKSPACE>'),('jvl210002','<ANONYMOUS_ACCOUNT>')]
    for old,new in replacements:value=value.replace(old,new)
    return value


def selected_paths():
    paths=set()
    for name,key in [('EVIDENCE_SNAPSHOT.json','sources'),('PUBLICATION_ANALYSIS.json','source_sha256')]:
        file=ROOT/'paper'/name
        if file.exists():
            document=json.loads(file.read_text());paths.add(file)
            paths.update(Path(p) for p in document.get(key,{}))
    paths.update(suite.DATA.glob('*.json*'))
    paths.update((ROOT/'configs').rglob('*.yaml'));paths.update((ROOT/'configs').glob('*.json'))
    paths.update((ROOT/'src/bidir').rglob('*.py'));paths.update((ROOT/'scripts').rglob('*.py'))
    paths.update((ROOT/'tests').glob('*.py'))
    for name in ['CLAUDE.md','PREREGISTRATION.md','RUN_PLAN.md','TASKS.md','scripts/env.sh','pytest.ini',
                 'docs/PAPER_FINISH_PLAN.md','docs/AUTONOMOUS_PIPELINE.md','paper/README.md','paper/ARGUMENT.md',
                 'docs/GATE_REPAIR_PLAN.md',
                 'env/extras.txt','paper/main.tex','paper/planned_evaluations.tex','paper/refs.bib',
                 'paper/numbers.tex','paper/numbers_provenance.json','paper/check_arr.py','paper/Makefile',
                 'paper/acl.sty','paper/acl_natbib.bst','paper/page_limit.txt','paper/evidence_snapshot.tex',
                 'paper/PROVENANCE.json','paper/PUBLICATION_PROVENANCE.json',
                 'paper/MECHANISM_REVIEW.json',
                 'runs/feeder/paper_finish_preflight.json','runs/feeder/paper_finish_status.json',
                 'runs/feeder/h100_overflow_admission.json']:
        paths.add(ROOT/name)
    paths.update((ROOT/'paper/tables').glob('*.tex'))
    paths.update((ROOT/'paper/figures').glob('*.pdf'))
    paths.update((ROOT/'paper/figures').glob('*.png'))
    paths.update((ROOT/'runs/status').glob('*paper*.json'))
    paths.update((ROOT/'runs/feeder').glob('paper_finish*.json'))
    # Include failed original gates and withdrawal records even when they do not support a gain.
    paths.update(RESULTS_DIR.glob('base_gates/*/*.json'))
    paths.update(RESULTS_DIR.glob('audit/*.json'))
    paths.update(RESULTS_DIR.glob('*/*/*/*/WITHDRAWN.txt'))
    paths.update(suite.OUT.rglob('*.json'))
    paths.update(suite.OUT.rglob('trials.jsonl'))
    paths.update((RESULTS_DIR/'gate_repair_v1').rglob('*.json'))
    paths.update((RESULTS_DIR/'gate_repair_v1').rglob('trials.jsonl'))
    paths.update((ROOT/'data').glob('*_v2/*.json'))
    paths.update((ROOT/'data').glob('*_v2/*.jsonl'))
    paths.update((ROOT/'runs/feeder').glob('gate_repair*.json'))
    paths.update((ROOT/'runs/status').glob('gr_*.json'))
    paths.update(RUNS_DIR.glob('adapters/*_v2/*/*/repair_training_protocol.json'))
    for filename in ['run_manifest.json','training_summary.json']:
        paths.update(RUNS_DIR.glob(f'adapters/*_v2/*/*/{filename}'))
    for filename in ['run_manifest.json','training_summary.json','resolved_recipe.json']:
        paths.update((RUNS_DIR/'paper_finish_v1').rglob(filename))
    return sorted(p for p in paths if p.is_file())


def build():
    paths=selected_paths();fingerprints=[]
    for p in paths:
        if p.name in {'paper_finish_status.json','gate_repair_status.json'}:
            status=json.loads(p.read_text())
            fingerprints.append((str(p),json.dumps(status.get('items',[]),sort_keys=True)))
        else:fingerprints.append((str(p),p.stat().st_mtime_ns,p.stat().st_size))
    signature=hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
    manifest_path=BUNDLE/'MANIFEST.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('signature')==signature:
        print('local artifact bundle unchanged');return 0
    BUNDLE.mkdir(parents=True,exist_ok=True);records=[]
    for path in paths:
        relative=destination(path);target=BUNDLE/relative;target.parent.mkdir(parents=True,exist_ok=True)
        raw=path.read_bytes()
        if path.suffix in {'.pdf','.png'}:public=raw
        else:public=redact(raw.decode('utf-8')).encode('utf-8')
        target.write_bytes(public)
        records.append(dict(path=str(relative),source_sha256=hashlib.sha256(raw).hexdigest(),
            packaged_sha256=hashlib.sha256(public).hexdigest(),bytes=len(public),redacted=public!=raw))
    metadata=dict(built_utc=datetime.now(timezone.utc).isoformat(),signature=signature,
        files=records,source_to_packaged_hash_mapping=True,
        limitations=['Local review bundle; no publication or redistribution has occurred.',
            'Raw upstream dataset reconstruction and license audit remain required for external release.',
            'Model/adapter weights excluded; training manifests record their identities.',
            'Captured historical script hashes may not match current code; exact historical training reproduction is not guaranteed.',
            'Path redaction changes bytes; original and packaged hashes are recorded separately.'])
    atomic_json(manifest_path,metadata)
    (BUNDLE/'README.txt').write_text('Anonymous local evidence bundle\n\n'
        'MANIFEST.json records original and packaged hashes, including path-redaction changes.\n'
        'Verify packaged bytes: python verify_bundle.py\n'
        'Replay recorded strict-score means: python replay_means.py\n'
        'The source/paper folder builds with Tectonic and Python/pypdf: make paper.\n'
        'The source/configs/paper_finish.json and source/data/paper_finish/manifest.json fix new protocols and dataset revisions.\n'
        'Generation and training need separately installed project environments, model access, and reconstructed original data.\n'
        'No model weights, credentials, or authentication cache is bundled. Review dataset licenses and historical code gaps before release.\n')
    (BUNDLE/'verify_bundle.py').write_text('from pathlib import Path\nimport hashlib,json\n'
        'root=Path(__file__).resolve().parent\n'
        'manifest=json.loads((root/"MANIFEST.json").read_text())\n'
        'bad=[f["path"] for f in manifest["files"] if hashlib.sha256((root/f["path"]).read_bytes()).hexdigest()!=f["packaged_sha256"]]\n'
        'assert not bad, bad\nprint("Verified",len(manifest["files"]),"packaged artifacts")\n')
    (BUNDLE/'replay_means.py').write_text('from pathlib import Path\nfrom collections import defaultdict\nimport json\n'
        'root=Path(__file__).resolve().parent\n'
        'for path in sorted((root/"evidence/results").rglob("trials.jsonl")):\n'
        '    values=defaultdict(list)\n'
        '    for line in path.open():\n'
        '        r=json.loads(line)\n'
        '        if r.get("strategy","simple")!="simple": continue\n'
        '        values[(r["system"],r.get("endpoint",r.get("direction")))].append(r["strict"])\n'
        '    print(json.dumps({"trial_file":str(path.relative_to(root)),"means":{str(k):sum(v)/len(v) for k,v in values.items()}},sort_keys=True))\n')
    print('Built local evidence bundle:',len(records),'files;',sum(r['bytes'] for r in records),'bytes')
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:raise SystemExit(build())
    print('Run --build to create the local anonymous evidence bundle.')
