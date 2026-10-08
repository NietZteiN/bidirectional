"""Persistent controller state and proof of completed evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def evaluation_succeeded(run, status_dir, name):
    run=Path(run)
    try:
        if not (run/'trials.jsonl').is_file(): return False
        summary=json.loads((run/'summary.json').read_text())
        finished=datetime.fromisoformat(summary['finished_utc'])
        if finished.tzinfo is None: finished=finished.replace(tzinfo=timezone.utc)
        for path in Path(status_dir).glob(name+'.*.json'):
            try:
                status=json.loads(path.read_text())
                ended=datetime.fromisoformat(status['finished_utc'])
                if ended.tzinfo is None: ended=ended.replace(tzinfo=timezone.utc)
                if status['exit_code']==0 and ended>=finished:
                    return True
            except (OSError,KeyError,ValueError):
                continue
    except (OSError,KeyError,ValueError):
        return False
    return False


def registered_engineering_job(status_dir, entry):
    """Resolve successful retries of a registered job name, preserving its ID floor.

    Workload-specific callers still validate profiles, exposures and manifests.
    A failed or malformed status never supplies an admission proof.
    """
    jobs = [int(entry['job'])]
    for path in Path(status_dir).glob(entry['name'] + '.*.json'):
        try:
            job = int(path.name.rsplit('.', 2)[1])
            status = json.loads(path.read_text())
            if job >= int(entry['job']) and status.get('exit_code') == 0:
                jobs.append(job)
        except (OSError, ValueError):
            continue
    return max(jobs)
