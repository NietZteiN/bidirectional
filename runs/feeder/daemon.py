#!/usr/bin/env python
"""One bounded controller pass per minute; Slurm owns all training/evaluation jobs."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
F=ROOT/'runs'/'feeder'
STOP=F/'STOP'
HEARTBEAT=F/'heartbeat.json'
shutdown=False


def terminate(*_):
    global shutdown
    shutdown=True


def heartbeat(phase, **extra):
    atomic_json(HEARTBEAT,dict(pid=os.getpid(),host=os.uname().nodename,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),updated_epoch=time.time(),
        updated_utc=datetime.now(timezone.utc).isoformat(),phase=phase,**extra))


def kill_group(pid):
    try: os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError: return
    time.sleep(.5)
    try: os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError: pass


def stage(name,cmd,timeout,log,shell=False):
    process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,start_new_session=True,shell=shell)
    started=time.monotonic()
    while process.poll() is None:
        heartbeat(name,stage_pid=process.pid,stage_started_epoch=time.time()-(time.monotonic()-started))
        if shutdown or STOP.exists() or time.monotonic()-started>timeout:
            kill_group(process.pid);process.wait()
            heartbeat(name,stage_pid=None,last_exit_code=124)
            return 124
        time.sleep(.5)
    rc=process.returncode
    log.write(f'[{name}] exit={rc}\n');log.flush()
    heartbeat(name,stage_pid=None,last_exit_code=rc)
    return rc


def queue_pass(log):
    queue=F/'queue.txt';queue.touch(exist_ok=True)
    tries=len(queue.read_text().splitlines())
    for _ in range(tries):
        if shutdown or STOP.exists(): break
        lines=queue.read_text().splitlines()
        if not lines: break
        line=lines[0]
        if not line.strip():
            queue.write_text('\n'.join(lines[1:])+('\n' if lines[1:] else ''));continue
        count=subprocess.run(['squeue','-h','-u',os.environ['USER']],capture_output=True,text=True,timeout=30,check=True)
        if len(count.stdout.splitlines())>=int(os.environ.get('CAP','90')): break
        journal=F/'queue_transaction.json'
        # An interrupted shell command may have submitted a partial train/eval pair.
        # Preserve it for review instead of guessing and duplicating expensive jobs.
        if journal.exists():
            transaction=json.loads(journal.read_text())
            if transaction.get('state') in ('launching','interrupted'):
                log.write('[queue] interrupted transaction preserved; see queue_transaction.json\n');log.flush();break
        atomic_json(journal,dict(state='launching',command=line,started_utc=datetime.now(timezone.utc).isoformat()))
        output=F/'queue_command.out'
        with output.open('w') as command_log:
            rc=stage('queue',line,180,command_log,shell=True)
        content=output.read_text()
        log.write(content);log.flush()
        if rc==124:
            atomic_json(journal,dict(state='interrupted',command=line,exit_code=rc,output=content))
            break
        atomic_json(journal,dict(state='finished',command=line,exit_code=rc,output=content))
        current=queue.read_text().splitlines()
        if current and current[0]==line:
            current=current[1:]
            if rc!=0: current.append(line)
            temporary=queue.with_suffix('.txt.tmp');temporary.write_text('\n'.join(current)+('\n' if current else ''));temporary.replace(queue)
        if rc==0:
            with (F/'submitted.tsv').open('a') as submitted:
                for jid,name in re.findall(r'^submitted\s+(\d+)\s+(\S+)',content,re.M):
                    submitted.write(f'{jid}\t{name}\t{line}\n')
        elif 'REFUSED' in content: break


def main():
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    os.chdir(ROOT)
    with (F/'.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 0
        interval=float(os.environ.get('BIDIR_FEEDER_INTERVAL','60'))
        with (F/'feeder.log').open('a',buffering=1) as log:
            while not shutdown and not STOP.exists():
                results={}
                log.write(f'--- {datetime.now(timezone.utc).isoformat()} controller {os.getpid()} ---\n')
                for name,cmd,timeout in [
                    ('watchdog',[sys.executable,'runs/feeder/watchdog.py'],120),
                    ('capacity',[sys.executable,'scripts/96_gpu_capacity.py'],90),
                    ('a30_admission',[sys.executable,'scripts/103_a30_admission.py'],90),
                    ('runner',[sys.executable,'scripts/95_runner.py','--max-inflight','24'],600),
                    ('followup_analysis',[sys.executable,'scripts/102_followup_analysis.py'],600),
                    ('paper_snapshot',[sys.executable,'scripts/99_paper_evidence.py','--if-changed','--build'],600)]:
                    if shutdown or STOP.exists(): break
                    try: results[name]=stage(name,cmd,timeout,log)
                    except Exception as exc:
                        results[name]=1;log.write(f'[{name}] {type(exc).__name__}: {exc}\n')
                if not shutdown and not STOP.exists():
                    try: queue_pass(log)
                    except Exception as exc: log.write(f'[queue] {type(exc).__name__}: {exc}\n');results['queue']=1
                heartbeat('idle',stage_pid=None,last_pass=results)
                deadline=time.monotonic()+interval
                while not shutdown and not STOP.exists() and time.monotonic()<deadline:
                    heartbeat('idle',stage_pid=None,last_pass=results);time.sleep(min(5,interval))
            heartbeat('stopped',stage_pid=None)
    return 0


if __name__=='__main__': raise SystemExit(main())
