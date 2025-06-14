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

    if not ("profile" in start_param.keys()):
        raise ValueError("Нужно указать профиль в конфигурации (пример: profile: 'water')")
    with open(f'{BASE_DIR}/configs/profiles/{start_param["profile"]}.yml', 'r', encoding='utf-8') as f2:
        profile_cfg = yaml.safe_load(f2)
    for key in profile_cfg:
        start_param[key] = profile_cfg[key]

    if not ("scenario" in start_param.keys()):
        raise ValueError("Нужно указать сценарий в конфигурации (пример: scenario: 'dam_break')")
    with open(f'{BASE_DIR}/configs/scenario/{start_param["scenario"]}.yml', 'r', encoding='utf-8') as f3:
        scenario_cfg = yaml.safe_load(f3)
    start_param["scenario_param"] = dict()
    for key in scenario_cfg:
        start_param["scenario_param"][key] = scenario_cfg[key]

    if not ("kernel" in start_param.keys()):
        raise ValueError("Нужно указать ядро в конфигурации (пример: kernel: 'wendland_c2')")
    if not ("dim" in start_param["scenario_param"].keys()):
        raise ValueError("Нужно указать количество измерений в конфигурации (пример: dim: 1)")
    kernel_name = "Не указано ядро"
    if start_param["kernel"].lower() in ["spline", "cubic_spline"]:
        kernel_name = "cubic_spline"
    elif start_param["kernel"].lower() in ["wendland", "wendlandc2", "wendland_c2"]:
        kernel_name = "wendland_c2"
    elif start_param["kernel"].lower() in ["gaussian", "gauss"]:
        kernel_name = "gauss"
    with open(f'{BASE_DIR}/configs/kernels/{kernel_name}/{start_param["scenario_param"]["dim"]}d.yml', 'r',
              encoding='utf-8') as f3:
        kernel_cfg = yaml.safe_load(f3)
    start_param["kernel_param"] = dict()
    for key in kernel_cfg:
        start_param["kernel_param"][key] = kernel_cfg[key]

    if not ("pst" in start_param.keys()) or start_param["pst"].lower() == "none":
        start_param["pst"] = "none"
    else:
        with open(f'{BASE_DIR}/configs/pst/{start_param["pst"]}.yml', 'r', encoding='utf-8') as f3:
            pst_cfg = yaml.safe_load(f3)
        for key in pst_cfg:
            start_param[key] = pst_cfg[key]

    if not ("viscosity" in start_param.keys()):
        start_param["viscosity"] = "none"
    else:
        with open(f'{BASE_DIR}/configs/viscosity/{start_param["viscosity"]}.yml', 'r', encoding='utf-8') as f3:
            viscosity_cfg = yaml.safe_load(f3)
        for key in viscosity_cfg:
            start_param[key] = viscosity_cfg[key]

    if print_param:
        print(f"Конфигурация: {name}")
        for key in start_param:
            print(f"\t{key}: {start_param[key]}")

    cfg = Config(start_param=start_param)
    return cfg
