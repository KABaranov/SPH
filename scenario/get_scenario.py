from SPH.scenario.density_test1 import density_test1
from SPH.scenario.eos_test import test_eos_all

from typing import Callable


def get_scenario(name: str) -> Callable:
    scenario_list = {
        "density_test1": density_test1,
        "eos_test": test_eos_all
    }
    return scenario_list[name]
