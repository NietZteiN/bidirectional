#!/usr/bin/env python
"""Isolated trainer/pack for the versioned fmt contrastive experiment."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir import contrastive, gate_repair as repair
from bidir.config import DATA_DIR
from bidir.pipeline_state import atomic_json
from bidir.schema import read_pairs
from bidir.train import adapter_dir


def install_candidates():
    """Reuse the validated fmt filter with a checked identical training-only pool."""
    cell=repair.plan()['contrastive_cell']
    def content(path):
        return [(p.side_a,p.side_b,p.subtask) for p in read_pairs(path)]
    if content(DATA_DIR/cell/'train.jsonl')!=content(DATA_DIR/'fmt/train.jsonl'):
        raise ValueError('repaired fmt negative pool does not match original training content')
    original=contrastive.attach_candidates
    def attach(examples,rows,domain,seed,shuffled=False):
        if domain!=cell:raise ValueError('repair trainer is restricted to '+cell)
        return original(examples,rows,'fmt',seed,shuffled)
    contrastive.attach_candidates=attach


def single(argv):
    from bidir.train import main as train
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--domain',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--rank',type=int,default=32)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--out',type=Path)
    args,_=parser.parse_known_args(argv)
    if args.domain!=repair.plan()['contrastive_cell']:raise ValueError('unregistered repair cell')
    install_candidates()
    frozen_hash=repair.code_hash()
    rc=train(argv)
    out=args.out or adapter_dir(args.domain,args.model,args.arm,args.rank,args.seed)
    if rc==0 and (out/'run_manifest.json').exists():
        atomic_json(out/'repair_training_protocol.json',dict(protocol=repair.plan()['protocol'],
            worker_sha256=frozen_hash,data_sha256=repair.data_hash(args.domain),
            negative_pool_sha256=repair.sha(DATA_DIR/'fmt/train.jsonl'),
            manifest_sha256=repair.sha(out/'run_manifest.json'),
            finished_utc=datetime.now(timezone.utc).isoformat()))
    return rc


def main():
    if '--single' in sys.argv:
        return single([arg for arg in sys.argv[1:] if arg!='--single'])
    spec=importlib.util.spec_from_file_location('repair_pack',ROOT/'scripts/20_train_pack.py')
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    # Only rewrite the pack's child trainer entrypoint; all packing, ordering,
    # completion checks and arm accounting remain the established implementation.
    original=subprocess.run
    def run(argv,*args,**kwargs):
        if isinstance(argv,list) and argv[1:3]==['-m','bidir.train']:
            argv=[argv[0],str(Path(__file__).resolve()),'--single',*argv[3:]]
        return original(argv,*args,**kwargs)
    pack.subprocess.run=run
    return pack.main()


if __name__=='__main__':raise SystemExit(main())
