from SPH.configs.config_class import Config
from SPH.core.equations.eos import eos

import numpy as np
import matplotlib.pyplot as plt


# ===========================================
# Проверка eos
# ===========================================

# Проверка нулевого давления rho=rho_0
# Для классической формы уравнения Тейта (см. функцию eos)
# при rho=rho_0 давление строго равно 0 (если p_floor <= 0)
def test_eos_at_rho0(cfg: Config) -> bool:
    return eos(cfg.rho0, cfg) == max(0.0, cfg.p_floor)


# Проверка монотонности (2 функции)
# Функция должна быть строго возрастающей
# Если rho_2 > rho_1, то p(rho_2) > p(rho_1) (при любом gamma > 1)
def test_eos_monotonic(cfg: Config, k1: float, k2: float) -> bool:
    p1 = eos(k1*cfg.rho0, cfg)
    p2 = eos(k2*cfg.rho0, cfg)
    return p2 > p1


# Функция построения графика монотонности
def plot_eos_monotonic(cfg: Config, a: float = 0.0, b: float = 2.0) -> None:
    rho = np.linspace(a * cfg.rho0, b * cfg.rho0, int(cfg.rho0 * 2.0))
    p = [eos(rho_i, cfg) for rho_i in rho]
    plt.plot(rho, p)
    plt.title("Монотонность функции p(rho)")
    plt.show()


# Правильная скорость звука
# Скорость звука для Тейта - это корень частной производной dp/drho
# Функция проверяет численную производную
# (Может не выдавать True из-за p_floor)
def sound_speed(rho: float, cfg: Config) -> float:
    """Аналитический c(ρ) без учёта p_floor."""
    return np.sqrt(cfg.gamma * cfg.B / cfg.rho0) * (rho/cfg.rho0)**((cfg.gamma-1)/2)


def test_sound_speed(cfg: Config, delta: float = 1e-6) -> bool:
    rho = cfg.rho0 * 1.02            # на 2 % выше, чтобы точно > p_floor
    dpdrho_num = (eos(rho*(1+delta), cfg) - eos(rho*(1-delta), cfg)) / (2*rho*delta)
    c_num = np.sqrt(dpdrho_num)
    c_theory = sound_speed(rho, cfg)
    return abs(c_num - c_theory)/c_theory < 1e-4


# Функция проверки eos на математическую корректность
# запускает 3 теста
def test_eos_all(cfg: Config) -> None:
    for param in ["k1", "k2", "delta"]:
        if param not in cfg.scenario_param.keys():
            raise ValueError(f"Необходимо указать {param} в параметрах сценария (eos_test)")
    k1, k2, delta = cfg.scenario_param["k1"], cfg.scenario_param["k2"], cfg.scenario_param["delta"]
    print("Проверка eos на математическую корректность:")
    r1 = test_eos_at_rho0(cfg)
    print(f"\tПроверка нулевого давления{'' if r1 else ' не'} пройдена")
    r2 = test_eos_monotonic(cfg, k1, k2)
    print(f"\tПроверка монотонности{'' if r2 else ' не'} пройдена")
    r3 = test_sound_speed(cfg, delta)
    print(f"\tПроверка правильной скорости звука{'' if r3 else ' не'} пройдена")
    print("\t============================")
    result = r1 * r2 * r3
    print(f"\tПроверка{'' if result else ' не'} пройдена")
