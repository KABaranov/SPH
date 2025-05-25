from .config_class import Config
import yaml
import os


def get_config(name: str = "common", print_param: bool = False) -> Config:
    start_param = dict()
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    # чтение
    with open(f'{BASE_DIR}/configs/{name}.yml', 'r', encoding='utf-8') as f1:
        common_cfg = yaml.safe_load(f1)
    for key in common_cfg:
        start_param[key] = common_cfg[key]

    if not ("profile" in common_cfg):
        raise ValueError("Нужно указать профиль в конфигурации (пример: profile: 'water')")
    with open(f'{BASE_DIR}/configs/profiles/{common_cfg["profile"]}.yml', 'r', encoding='utf-8') as f2:
        profile_cfg = yaml.safe_load(f2)
    for key in profile_cfg:
        start_param[key] = profile_cfg[key]

    if not ("scenario" in common_cfg):
        raise ValueError("Нужно указать сценарий в конфигурации (пример: scenario: 'dam_break')")
    with open(f'{BASE_DIR}/configs/scenario/{common_cfg["scenario"]}.yml', 'r', encoding='utf-8') as f3:
        scenario_cfg = yaml.safe_load(f3)
    for key in scenario_cfg:
        start_param[key] = scenario_cfg[key]

    if print_param:
        print(f"Конфигурация: {name}")
        for key in start_param:
            print(f"\t{key}: {start_param[key]}")

    cfg = Config(start_param=start_param)
    return cfg
