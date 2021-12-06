import json

def write_json(e, file:str):
    with open(file, 'w') as f:
        json.dump(e, f, indent=4)

def load_json(file: str):
    with open(file, 'r') as f:
        e = json.load(f)
    return e
