from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Уравнение состояния
# ------------------------------------------------------------------
def eos(rho: float, cfg: Config) -> float:
    """Tait EOS: p = B * ((rho / rho0)^gamma − 1)  c отсечкой p ≥ p_floor"""
    p = cfg.B * ((rho / cfg.rho0) ** cfg.gamma - 1.0)
    return max(p, cfg.p_floor)
