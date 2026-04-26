"""
state_manager.py — Controla quais steps do pipeline já foram executados.

Melhoria: usa FileLock para evitar corrida entre processos ao ler/escrever
o arquivo pipeline_state.json. Fallback gracioso se filelock não estiver
instalado (comportamento anterior, sem trava).
"""

from __future__ import annotations

import json
from pathlib import Path

STATE_FILE = "pipeline_state.json"
LOCK_FILE  = STATE_FILE + ".lock"


def _load_unsafe() -> dict:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_unsafe(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _with_lock(fn):
    """Executa fn() dentro de um FileLock, se disponível."""
    try:
        from filelock import FileLock
        with FileLock(LOCK_FILE, timeout=10):
            return fn()
    except ImportError:
        # filelock não instalado: comportamento legado sem trava
        return fn()
    except Exception as e:
        print(f"[state_manager] Aviso de lock: {e} — continuando sem trava.")
        return fn()


def load_state() -> dict:
    return _with_lock(_load_unsafe)


def save_state(state: dict) -> None:
    _with_lock(lambda: _save_unsafe(state))


def mark_done(step: str) -> None:
    def _op():
        state = _load_unsafe()
        state[step] = True
        _save_unsafe(state)
    _with_lock(_op)


def is_done(step: str) -> bool:
    def _op():
        return _load_unsafe().get(step, False)
    return _with_lock(_op)


def reset(step: str | None = None) -> None:
    """Remove um step específico (ou todos) do estado salvo."""
    def _op():
        state = _load_unsafe()
        if step is None:
            state.clear()
        else:
            state.pop(step, None)
        _save_unsafe(state)
    _with_lock(_op)