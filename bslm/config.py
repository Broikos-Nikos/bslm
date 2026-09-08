"""The small set of variables the assistant needs that are not in the prompt:
the home place (used when a weather question names none), the soul fact the
model is told about the user, and the assistant's name. Editable in the
showcase window's left panel, saved to data/showcase_config.json, and read by
the tool coverage test so both use the same place."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "data" / "showcase_config.json"
DEFAULTS = {"home": "Athens", "soul": "rides a scooter", "name": "Bee"}


def load():
    d = dict(DEFAULTS)
    try:
        d.update({k: v for k, v in json.loads(CFG.read_text(encoding="utf-8")).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return d


def save(d):
    CFG.parent.mkdir(exist_ok=True)
    CFG.write_text(json.dumps({k: d.get(k, DEFAULTS[k]) for k in DEFAULTS}, indent=2, ensure_ascii=False),
                   encoding="utf-8", newline="\n")
    return d
