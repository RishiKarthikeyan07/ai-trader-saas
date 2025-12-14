from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass
class RuntimeState:
    kill_switch: bool = False
    elite_auto_enabled: bool = False


_state = RuntimeState()
_lock = Lock()


def set_kill_switch(value: bool) -> RuntimeState:
    with _lock:
        _state.kill_switch = value
        return _state


def set_elite_auto(value: bool) -> RuntimeState:
    with _lock:
        _state.elite_auto_enabled = value
        return _state


def get_state() -> RuntimeState:
    return _state
