import json
from pathlib import Path

STATE_FILE = "pipeline_state.json"


def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_done(step):
    state = load_state()
    state[step] = True
    save_state(state)


def is_done(step):
    state = load_state()
    return state.get(step, False)