import json


def load_configs(path: str = 'config.json') -> json:
    with open(path, 'r', encoding='UTF8') as f:
        return json.loads(f.read())
