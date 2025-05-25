from SPH.configs.config_class import Config
from eos import eos
import matplotlib.pyplot as plt
import numpy as np

from SPH.configs.get_config import get_config


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
def test_sound_speed(cfg: Config, delta: float = 1e-6):
    rho = cfg.rho0
    p_plus = eos(rho*(1+delta), cfg)
    p_minus = eos(rho*(1-delta), cfg)
    dpdrho_num = (p_plus - p_minus) / (2*rho*delta)
    c_num = np.sqrt(dpdrho_num)
    c_theory = np.sqrt(cfg.gamma * cfg.B / cfg.rho0)
    print(c_num, c_theory)
    return abs(c_num - c_theory)/c_theory < 1e-4


if __name__ == "__main__":
    config = get_config("common", print_param=True)
    print(test_sound_speed(config))
