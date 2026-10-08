#!/usr/bin/env python
"""Two-step actual trainer smoke, isolated from experimental adapter paths."""
import argparse
import os
import json
import time
import torch
from bidir.train import main
from bidir.config import RESULTS_DIR

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--arm',default='cl_fwd')
    parser.add_argument('--train-config',default='train/contrastive_smoke.yaml')
    args=parser.parse_args()
    out=RESULTS_DIR / 'engineering_smokes' / f'{args.model}_{args.arm}_{os.environ.get("SLURM_JOB_ID","local")}'
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    rc=main(['--domain','fmt','--model',args.model,'--arm',args.arm,
                          '--train-config',args.train_config,'--max-steps','2',
                          '--out',str(out)])
    report={'engineering_smoke':True,'model':args.model,'arm':args.arm,'elapsed_s':time.monotonic()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'exit_code':rc}
    (out / 'profile.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    raise SystemExit(rc)
