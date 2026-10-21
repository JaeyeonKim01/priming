"""Coordinated checkpoint requests without interrupting distributed updates."""
import os
from pathlib import Path
import time


def stop_reason(settings, session_start, *, now=None, monotonic=None):
    now = time.time() if now is None else now
    monotonic = time.monotonic() if monotonic is None else monotonic
    stop_file = settings.get('stop_file')
    if stop_file and Path(stop_file).exists():
        return 'stop_file'
    deadline = os.environ.get('WORLD_MODEL_DEADLINE')
    margin = float(settings.get('deadline_margin_seconds', 180))
    if deadline and now >= float(deadline) - margin:
        return 'wall_time'
    maximum = settings.get('max_session_seconds')
    if maximum is not None and monotonic - session_start >= float(maximum):
        return 'session_time'
    return None
