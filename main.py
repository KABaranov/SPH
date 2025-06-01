from SPH.configs.get_config import get_config
from SPH.scenario.get_scenario import get_scenario


if __name__ == "__main__":
    cfg = get_config(print_param=True)
    get_scenario(cfg.scenario_name)(cfg, out_plot=True)
