import yaml
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.insert(0, project_root)

from functions.config import resolve_init_path


def load_config(load_all: list = None):
    if load_all is None:
        load_all = ['config', 'config_pipe', 'config_model']

    config_files = {
        'config': 'config.yaml',
        'config_pipe': 'pipeline.yaml',
        'config_model': 'model.yaml',
    }

    invalid_configs = [config_name for config_name in load_all if config_name not in config_files]
    if invalid_configs:
        valid_configs = ', '.join(config_files.keys())
        raise ValueError(f"Config(s) invalida(s): {invalid_configs}. Opcoes validas: {valid_configs}")

    loaded_configs = []
    for config_name in load_all:
        config_path = os.path.join(
            project_root,
            "Regression/house_prices/config",
            config_files[config_name],
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config_name == 'config':
                config = resolve_init_path(config, project_root)
            loaded_configs.append(config)

    if len(loaded_configs) == 1:
        return loaded_configs[0]

    return tuple(loaded_configs)
