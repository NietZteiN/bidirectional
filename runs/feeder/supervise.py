#!/usr/bin/env python
"""Restart a dead/stalled controller, retaining Slurm jobs and on-disk progress."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
from daemon import kill_group
F=ROOT/'runs'/'feeder'
shutdown=False


def terminate(*_):
    global shutdown
    shutdown=True


def read_heartbeat():
    try: return json.loads((F/'heartbeat.json').read_text())
    except (OSError,ValueError): return {}


def main():
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    with (F/'.supervisor.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 0
        # A supervisor crash leaves its separately-sessioned worker alive. Reclaim
        # only this allocation's verified local daemon before starting a replacement.
        orphan=read_heartbeat()
        pid=orphan.get('pid')
        if (pid and orphan.get('host')==os.uname().nodename
            and orphan.get('slurm_job_id')==os.environ.get('SLURM_JOB_ID')):
            try:
                process=Path(f'/proc/{pid}')
                owned=process.stat().st_uid==os.getuid()
                argv=(process/'cmdline').read_bytes().decode(errors='replace')
                if owned and str(F/'daemon.py') in argv:
                    if orphan.get('stage_pid'): kill_group(orphan['stage_pid'])
                    kill_group(pid)
            except (OSError,ValueError): pass
        restarts=0
        with (F/'supervisor.log').open('a',buffering=1) as log:
            while not shutdown and not (F/'STOP').exists():
                child=subprocess.Popen([sys.executable,str(F/'daemon.py')],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.time()
                log.write(f'start controller {child.pid}; restart={restarts}\n')
                while child.poll() is None and not shutdown and not (F/'STOP').exists():
                    atomic_json(F/'supervisor.json',dict(pid=os.getpid(),controller_pid=child.pid,
                        host=os.uname().nodename,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                        updated_epoch=time.time(),restarts=restarts))
                    h=read_heartbeat()
                    if h.get('pid')==child.pid and time.time()-h.get('updated_epoch',started)>90:
                        log.write('stale heartbeat; restarting controller\n');break
                    if time.time()-started>90 and h.get('pid')!=child.pid:
                        log.write('controller never wrote heartbeat; restarting\n');break
                    time.sleep(2)
                h=read_heartbeat()
                if h.get('pid')==child.pid and h.get('stage_pid'):
                    kill_group(h['stage_pid'])
                kill_group(child.pid);child.wait()
                if shutdown or (F/'STOP').exists(): break
                restarts+=1
                time.sleep(3)
    return 0


if __name__=='__main__': raise SystemExit(main())
