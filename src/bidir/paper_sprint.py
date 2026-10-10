"""Explicit user-directed admission budget; original scientific protocols stay frozen."""
from datetime import datetime, timezone
import json
from pathlib import Path

POLICY = Path(__file__).resolve().parents[2] / 'configs/paper_sprint.json'


def plan():
    if not POLICY.exists():
        return None
    config = json.loads(POLICY.read_text())
    if not config['enabled']:
        return None
    deadline = datetime.fromisoformat(config['finish_by_utc'])
    if deadline.tzinfo is None or not set(config['production_jobs']) <= set(config['allowed_jobs']):
        raise ValueError('invalid paper sprint policy')
    return config


def deferred_reason(name, now=None):
    config = plan()
    if config is None:
        return None
    if name not in config['allowed_jobs']:
        return 'deferred by user 24-hour scope'
    if (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(config['finish_by_utc']):
        return 'deferred at 24-hour admission deadline'
    return None


def allowed(name, now=None):
    return deferred_reason(name, now) is None
