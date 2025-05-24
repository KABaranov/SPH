import numpy as np


class Config:
    def __init__(
        self,
        start_param: dict,
    ) -> None:
        self.rho0 = start_param["rho0"] if "rho0" in start_param.keys() else 1000.0
        self.gamma = start_param["gamma"] if "gamma" in start_param.keys() else 7.0
        if "B" in start_param.keys():
            self.B = start_param["B"]
            self.c0 = np.sqrt(self.B * self.gamma / self.rho0)
        else:
            if "c0" in start_param.keys():
                self.c0 = start_param["c0"]
                self.B = self.rho0 * self.c0 ** 2.0 / self.gamma
            else:
                raise ValueError("Нужно указать B или c0 для EOS")
        self.alpha = start_param["alpha"] if "alpha" in start_param.keys() else 1.0
        self.beta = start_param["beta"] if "beta" in start_param.keys() else 2.0
        self.epsilon = start_param["epsilon"] if "epsilon" in start_param.keys() else 0.01
        self.p_floor = start_param["p_floor"] if "p_floor" in start_param.keys() else 0.0
