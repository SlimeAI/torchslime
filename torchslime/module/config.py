import json


def load_json(path: str):
    with open(path) as f:
        cfg = json.load(f)
    return cfg
