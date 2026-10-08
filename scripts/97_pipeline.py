#!/usr/bin/env python
"""Start/status/stop/debug the detached Slurm-hosted pipeline."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bidir.pipeline_state import atomic_json
F=ROOT/'runs'/'feeder'
HOST=ROOT.parent/'node'/'hold.sbatch'


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return {}


def command(args):
    return subprocess.run(args,capture_output=True,text=True,timeout=30,check=True).stdout.strip()


def host_jobs():
    return command(['squeue','-h','-u',os.environ['USER'],'-n','hold-node','-o','%i|%T|%N|%E'])


def status():
    h=read(F/'heartbeat.json');s=read(F/'supervisor.json')
    return dict(supervisor=s,controller=h,
                heartbeat_age_s=round(time.time()-h['updated_epoch'],1) if h.get('updated_epoch') else None,
                stop_requested=(F/'STOP').exists(),host_jobs=host_jobs())


def start():
    (F/'STOP').unlink(missing_ok=True)
    state=status()
    if state['heartbeat_age_s'] is not None and state['heartbeat_age_s']<90 and state['controller'].get('phase')!='stopped':
        print('Pipeline already active; '+str(state['controller']['pid']));return 0
    active=[line.split('|')[0] for line in state['host_jobs'].splitlines() if '|RUNNING|' in line]
    if not active:
        print(command(['sbatch',str(HOST)]));return 0
    with (F/'startup.log').open('a') as log:
        process=subprocess.Popen(['srun','--jobid='+active[0],'--overlap','--ntasks=1',
            '--cpus-per-task=2','bash',str(F/'feeder.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Detached Slurm controller step started: client pid {process.pid}, host job {active[0]}')
    return 0


def debug(fault=False,supervisor_fault=False):
    state=status();h=state['controller'];s=state['supervisor'];checks={}
    # Stage heartbeats omit last_pass while a healthy pass is running. Observe a
    # completed pass before judging stage health instead of reporting a false failure.
    deadline=time.monotonic()+25
    while 'last_pass' not in h and time.monotonic()<deadline:
        time.sleep(.5);h=read(F/'heartbeat.json')
    s=read(F/'supervisor.json')
    checks['fresh_heartbeat']=state['heartbeat_age_s'] is not None and state['heartbeat_age_s']<90
    checks['supervisor_matches_controller']=s.get('controller_pid')==h.get('pid')
    checks['stages_ok']=bool(h.get('last_pass')) and all(v==0 for v in h.get('last_pass',{}).values())
    jobs=state['host_jobs'].splitlines()
    running=[line.split('|')[0] for line in jobs if '|RUNNING|' in line]
    checks['renewal_successor']=any('|PENDING|' in line and any('afterany:'+job in line for job in running) for line in jobs)
    checks['logout_detached']=False
    if h.get('host')==os.uname().nodename and h.get('pid'):
        try:
            stdin=os.readlink(f"/proc/{h['pid']}/fd/0")
            checks['logout_detached']=stdin=='/dev/null' and os.getsid(h['pid'])==h['pid']
        except OSError: pass
    if fault or supervisor_fault:
        if not all(checks.values()):
            print(json.dumps(checks,indent=2));raise RuntimeError('health checks must pass before fault injection')
        deadline=time.monotonic()+25
        while h.get('phase')!='idle' and time.monotonic()<deadline:
            time.sleep(1);h=read(F/'heartbeat.json')
        if h.get('phase')!='idle': raise RuntimeError('controller busy; retry fault test when idle')
        if h['host']!=os.uname().nodename: raise RuntimeError('run fault test on the controller host')
        target='supervisor_crash_recovery' if supervisor_fault else 'crash_recovery'
        old=h['pid'];old_supervisor=read(F/'supervisor.json').get('pid')
        os.kill(old_supervisor if supervisor_fault else old,signal.SIGKILL)
        deadline=time.monotonic()+45
        checks[target]=False
        while time.monotonic()<deadline:
            time.sleep(1);new=read(F/'heartbeat.json');supervisor=read(F/'supervisor.json')
            if new.get('pid')!=old and new.get('phase')=='idle' and all(v==0 for v in new.get('last_pass',{}).values()):
                checks[target]=supervisor.get('controller_pid')==new.get('pid') and (not supervisor_fault or supervisor.get('pid')!=old_supervisor)
                break
    report=dict(finished_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
                fault_test=fault,supervisor_fault_test=supervisor_fault,
                host=os.uname().nodename,passed=all(checks.values()))
    atomic_json(F/'debug_latest.json',report)
    with (F/'debug_history.jsonl').open('a') as history:
        history.write(json.dumps(report)+'\n')
    print(json.dumps(checks,indent=2))
    return 0 if all(checks.values()) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop','debug'])
    parser.add_argument('--fault-test',action='store_true')
    parser.add_argument('--supervisor-fault-test',action='store_true')
    args=parser.parse_args()
    if args.action=='start': return start()
    if args.action=='status': print(json.dumps(status(),indent=2));return 0
    if args.action=='stop':
        (F/'STOP').touch();print('Scheduling stop requested; submitted Slurm jobs continue.');return 0
    return debug(args.fault_test,args.supervisor_fault_test)


if __name__=='__main__': raise SystemExit(main())
