#!/usr/bin/env python3
"""Admit at most BIDIR_GPU_CAP GPU jobs, reserving account capacity for probing.

Own holds are marked so the runner counts them and never submits duplicates. Existing
running jobs are never interrupted. Only jobs in this repository are changed.
"""
import fcntl
import getpass
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MARK = 'bidir_capacity_hold'

def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout

def main():
    cap = int(os.environ.get('BIDIR_GPU_CAP', '6'))
    if cap < 0:
        raise ValueError('GPU cap must be nonnegative; zero disables admission limits')
    lock = ROOT / 'runs/feeder/capacity.lock'
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        rows = command('squeue', '-u', getpass.getuser(), '-h', '-o',
                       '%i|%T|%r|%E|%k|%Z|%b').splitlines()
        ours = []
        for row in rows:
            fields = row.split('|')
            if len(fields) != 7:
                continue
            jid, state, reason, dep, comment, wd, tres = fields
            if Path(wd) != ROOT or 'gpu' not in tres:
                continue
            ours.append((jid, state, reason, dep, comment))
        if cap == 0:
            released = 0
            for jid, state, reason, dep, comment in ours:
                if state == 'PENDING' and comment == MARK:
                    command('scontrol', 'release', jid)
                    command('scontrol', 'update', f'JobId={jid}', 'Comment=')
                    released += 1
            print(f'[capacity] disabled; released {released} capacity holds')
            return
        running = sum(state in ('RUNNING', 'COMPLETING', 'CONFIGURING')
                      for _, state, *_ in ours)
        pending = [j for j in ours if j[1] == 'PENDING']
        candidates = [j for j in pending if j[3] in ('', '(null)')
                      and (j[2] not in ('JobHeldUser', 'JobHeldAdmin', 'DependencyNeverSatisfied')
                           or j[4] == MARK)]
        # Keep already released jobs admitted; prefer older job IDs for new admissions.
        candidates.sort(key=lambda j: (j[4] == MARK, int(j[0])))
        admitted = {j[0] for j in candidates[:max(0, cap-running)]}
        changes = 0
        # Hold excess before releasing replacements.
        for jid, state, reason, dep, comment in pending:
            if jid in admitted or comment == MARK or reason in ('JobHeldUser', 'JobHeldAdmin'):
                continue
            if comment not in ('', '(null)'):
                continue  # Preserve comments/holds belonging to other workflows.
            command('scontrol', 'update', f'JobId={jid}', f'Comment={MARK}')
            command('scontrol', 'hold', jid)
            changes += 1
        for jid, state, reason, dep, comment in pending:
            if jid in admitted and comment == MARK:
                command('scontrol', 'release', jid)
                command('scontrol', 'update', f'JobId={jid}', 'Comment=')
                changes += 1
        print(f'[capacity] {running} running, {len(admitted)} admitted pending, '
              f'cap={cap}; {changes} changes')

if __name__ == '__main__':
    main()
