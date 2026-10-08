#!/usr/bin/env python
"""Small allocation-local CUDA diagnostic; saves no scientific model outputs."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.config import RESULTS_DIR
from bidir.pipeline_state import atomic_json

def main():
    report=dict(host=os.uname().nodename,job_id=os.environ.get('SLURM_JOB_ID'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                ld_library_path=os.environ.get('LD_LIBRARY_PATH'))
    for key,cmd in [('gpu',['nvidia-smi','--query-gpu=name,driver_version,compute_mode,memory.total','--format=csv,noheader']),
                    ('driver',['nvidia-smi'])]:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        report[key]=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    try:
        driver=ctypes.CDLL('libcuda.so.1')
        rc=driver.cuInit(0);name=ctypes.c_char_p()
        driver.cuGetErrorName(rc,ctypes.byref(name))
        report['cuInit']=dict(code=rc,name=name.value.decode() if name.value else None)
    except Exception as exc:report['driver_error']=repr(exc)
    import torch
    report['torch_version']=torch.__version__;report['cuda_build']=torch.version.cuda
    report['cuda_available']=torch.cuda.is_available()
    try:
        report['tensor_result']=float((torch.ones(4,device='cuda')*2).sum().item())
        report['device']=torch.cuda.get_device_name()
        report['passed']=report['tensor_result']==8
    except Exception as exc:
        report['cuda_error']=repr(exc);report['passed']=False
    from datetime import datetime,timezone
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    out=RESULTS_DIR/'engineering_smokes/gpu_environment'/f"{report['job_id']}.json"
    atomic_json(out,report)
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
