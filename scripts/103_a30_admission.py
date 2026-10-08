#!/usr/bin/env python
"""Widen specific pending jobs after allocation-local A30 workload validation."""
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bidir.config import RESULTS_DIR
from bidir.pipeline_state import atomic_json, evaluation_succeeded

STATUS = ROOT / 'runs/status'
JOURNAL = ROOT / 'runs/feeder/a30_admission.json'
A30_EXCLUSIONS = {'g-01-01', 'g-02-01', 'g-03-01'}
ARMS = {'base', 'sft', 'mix5', 'replay', 'rev', 'cl_fwd', 'cl_mix5',
        'cl_shuffled', 'sft_extra_ce', 'mix5_extra_ce'}
TARGETS = {
    '447359': ('fullft_projection_repair_mt_de-en_s17', 'svd'),
    '447366': ('gate_explicit_units_gemma3-4b', 'evaluation'),
    '446428': ('ev_fmt_gemma3-4b_s42_contrastive_pilot', 'evaluation'),
    '446430': ('ev_fmt_gemma3-4b_s1234_contrastive_pilot', 'evaluation'),
}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def success(name, jid):
    report = read(STATUS / f'{name}.{jid}.json')
    return (report.get('exit_code') == 0 and report.get('job_id') == jid
            and report.get('node') == 'g-04-01' and report.get('partition') == 'a30')


def valid_campaign(summary):
    if (summary.get('domain') != 'fmt' or summary.get('model') != 'gemma3-4b'
            or summary.get('n_instances') != 20 or set(summary.get('systems', {})) != ARMS):
        return False
    effects = summary.get('adapter_effectiveness', {})
    if set(effects) != ARMS - {'base'}:
        return False
    if any(e.get('n_compared') != 40 or not 0 <= e.get('identical_rate', 1) < .999
           for e in effects.values()):
        return False
    coverage = {(c.get('system'), c.get('direction')): c.get('n')
                for c in summary.get('cells', []) if c.get('subtask') == 'ALL'}
    return coverage == {(a, d): 20 for a in ARMS for d in ('forward', 'reverse')}


def valid_svd(report):
    return (report.get('engineering_smoke') is True and report.get('exit_code') == 0
            and report.get('node') == 'g-04-01' and report.get('gpu') == 'NVIDIA A30'
            and 0 <= report.get('synthetic_relative_residual', 1) < 1e-4
            and 0 < report.get('peak_cuda_bytes', 0) < .7 * report.get('total_cuda_bytes', 0)
            and report.get('spectrum', {}).get('rank_considered', 0) >= 3072)


def pending_target(jid, name, state, partitions, proofs):
    target = TARGETS.get(jid)
    return (target is not None and target[0] == name and state == 'PENDING'
            and 'a30' not in partitions.split(',') and proofs.get(target[1]) is True)


def main():
    environment = read(RESULTS_DIR / 'engineering_smokes/gpu_environment/447401.json')
    if not (environment.get('passed') is True and environment.get('host') == 'g-04-01'
            and success('debug_a30_cuda_environment', '447401')):
        print('A30 admission: environment proof pending'); return 0
    eval_proof = False
    if success('a30_gemma_full_eval_smoke', '447997'):
        for run in RESULTS_DIR.glob('*/fmt/gemma3-4b/a30_full_engineering_smoke_s17'):
            if (evaluation_succeeded(run, STATUS, 'a30_gemma_full_eval_smoke')
                    and valid_campaign(read(run / 'summary.json'))):
                eval_proof = True
    proofs = dict(evaluation=eval_proof, svd=(success('a30_fullft_svd_smoke', '447993')
        and valid_svd(read(RESULTS_DIR / 'engineering_smokes/fullft_svd/447993.json'))))
    if not any(proofs.values()):
        print('A30 admission: workload proofs pending'); return 0
    queue = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T|%P'],
                           capture_output=True, text=True, check=True, timeout=30).stdout
    journal = read(JOURNAL)
    for line in queue.splitlines():
        fields = line.strip().split('|')
        if len(fields) != 4 or not pending_target(*fields, proofs):
            continue
        jid, name, _, _ = fields
        history = journal.setdefault(jid, [])
        if len(history) >= 3:
            continue
        before = subprocess.run(['scontrol', 'show', 'job', jid, '-o'], capture_output=True,
                                text=True, check=True, timeout=30).stdout
        # Recheck after reading squeue. Never alter a job already running.
        if 'JobState=PENDING ' not in before or f'JobName={name} ' not in before:
            continue
        match = re.search(r'\bExcNodeList=(\S+)', before)
        previous = set(match[1].split(',')) if match and match[1] != '(null)' else set()
        excluded = ','.join(sorted(previous | A30_EXCLUSIONS))
        command = ['scontrol', 'update', f'JobId={jid}', 'Partition=h100,h200,a30',
                   f'ExcNodeList={excluded}']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        history.append(dict(updated_utc=datetime.now(timezone.utc).isoformat(), proofs=proofs,
            before=before, command=command, exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        atomic_json(JOURNAL, journal)
        print(f'A30 admission {jid}: exit={result.returncode} {result.stderr.strip()}')
        if result.returncode:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
