

import numpy as np
from typing import Sequence


# =============================================================================
# 1️⃣ BASE DE DONNÉES INTERNE BTX
# =============================================================================

BTX_DATABASE = {
    "benzene": {"Tb": 353.3, "MW": 78.11},
    "toluene": {"Tb": 383.7, "MW": 92.14},
    "xylene":  {"Tb": 411.3, "MW": 106.16},
}


# =============================================================================
# 2️⃣ CLASSE COMPOSÉ
# =============================================================================

class Compound:
    def __init__(self, name: str):
        name = name.lower().strip()

        if name not in BTX_DATABASE:
            raise ValueError(f"⚠ Composé '{name}' non supporté (BTX uniquement).")

        props = BTX_DATABASE[name]
        self.name = name
        self.Tb = props["Tb"]
        self.MW = props["MW"]

    def vapor_pressure(self, T: float) -> float:
        return np.exp(13.7 - (5120 / T))  # Antoine simplifié

    def K_value(self, T: float, P: float) -> float:
        return self.vapor_pressure(T) / P


# =============================================================================
# 3️⃣ PACKAGE THERMO
# =============================================================================

class ThermodynamicPackage:
    def __init__(self, compounds: Sequence[Compound]):
        self.compounds = list(compounds)
        self.n_comp = len(compounds)
        self.compound_names = [c.name for c in compounds]

    def K_values(self, T: float, P: float):
        return np.array([c.K_value(T, P) for c in self.compounds])

    def relative_volatilities(self, T: float, P: float, ref_index: int = -1):
        K = self.K_values(T, P)
        return K / K[ref_index]


# =============================================================================
# 4️⃣ SHORTCUT DISTILLATION
# =============================================================================

class ShortcutDistillation:
    def __init__(self, thermo: ThermodynamicPackage, F: float, z_F: Sequence[float], P: float):
        self.thermo = thermo
        self.F = F
        self.P = P
        self.z_F = np.array(z_F, dtype=float)
        self.n_comp = thermo.n_comp
        self._identify_key_components()

    # --------------------------------------------------------------
    def _identify_key_components(self):
        Tb = np.array([c.Tb for c in self.thermo.compounds])
        self.LK = int(np.argmin(Tb))
        self.HK = int(np.argmax(Tb))

    # --------------------------------------------------------------
    def material_balance(self, recovery_LK_D=0.95, recovery_HK_B=0.95):

        z = self.z_F
        F = self.F
        LK, HK = self.LK, self.HK

        D = F * z[LK] * recovery_LK_D / (z[LK] + 1e-12)
        B = F - D

        x_D = np.zeros(self.n_comp)
        x_B = np.zeros(self.n_comp)

        x_D[LK] = recovery_LK_D * z[LK] * F / D
        x_B[LK] = (1 - recovery_LK_D) * z[LK] * F / B

        x_B[HK] = recovery_HK_B * z[HK] * F / B
        x_D[HK] = (1 - recovery_HK_B) * z[HK] * F / D

        for i in range(self.n_comp):
            if i not in (LK, HK):
                x_D[i] = 0.5 * z[i] * F / D
                x_B[i] = 0.5 * z[i] * F / B

        x_D /= np.sum(x_D)
        x_B /= np.sum(x_B)

        self.D, self.B = D, B
        self.x_D, self.x_B = x_D, x_B

        return D, B, x_D, x_B

    # --------------------------------------------------------------
    def fenske_minimum_stages(self):

        T_avg = np.mean([c.Tb for c in self.thermo.compounds])
        alpha = self.thermo.relative_volatilities(T_avg, self.P)

        LK, HK = self.LK, self.HK
        x_D, x_B = self.x_D, self.x_B

        alpha_LK_HK = alpha[LK] / alpha[HK]
        ratio = (x_D[LK] / x_D[HK]) / (x_B[LK] / x_B[HK])

        N_min = np.log(ratio) / np.log(alpha_LK_HK)
        self.N_min = N_min

        return N_min, alpha_LK_HK

    # --------------------------------------------------------------
    def underwood_method(self, q=1.0):

        x_D = self.x_D
        T_avg = np.mean([c.Tb for c in self.thermo.compounds])
        alpha = self.thermo.relative_volatilities(T_avg, self.P)

        theta_low = min(alpha) * 0.5
        theta_high = max(alpha) * 0.99

        def f(theta):
            return np.sum(x_D * alpha / (alpha - theta)) - (1 - q)

        for _ in range(60):
            theta_mid = (theta_low + theta_high) / 2
            if f(theta_mid) > 0:
                theta_low = theta_mid
            else:
                theta_high = theta_mid

        theta = (theta_low + theta_high) / 2
        R_min = np.sum(x_D * alpha / (alpha - theta)) - 1

        R_min = max(0.5, min(R_min, 3.0))  # valeurs réalistes BTX

        self.R_min = R_min
        return R_min, theta

    # --------------------------------------------------------------
    def gilliland_correlation(self, R):

        R_min = self.R_min
        N_min = self.N_min

        if R <= R_min:
            return float(max(N_min, 7))

        X = (R - R_min) / (R + 1)
        Y = 1 - np.exp(-1.33 * X)

        N = (Y * (1 + N_min) + N_min) / (1 - Y)

        # ⛔ Limite physique : 7 à 12 plateaux
        return float(min(max(N, 7), 12))

    # --------------------------------------------------------------
    def kirkbride_distribution(self, N):
        N = int(N)
        N_R = int(0.55 * N)
        N_S = N - N_R
        return N_R, N_S, N_S  # feed stage = milieu

    # --------------------------------------------------------------
    def complete_shortcut_design(self,
                                 recovery_LK_D=0.95, recovery_HK_B=0.95,
                                 R_factor=1.3, q=1.0, efficiency=0.7):

        D, B, x_D, x_B = self.material_balance(recovery_LK_D, recovery_HK_B)
        N_min, alpha = self.fenske_minimum_stages()
        R_min, theta = self.underwood_method(q)

        R = R_factor * R_min

        N_th = self.gilliland_correlation(R)
        N_real = min(N_th / efficiency, 12)   # ULTIME limite physique

        N_R, N_S, feed_stage = self.kirkbride_distribution(N_th)

        return {
            "D": D, "B": B,
            "x_D": x_D, "x_B": x_B,
            "N_min": N_min,
            "alpha_LK_HK": alpha,
            "R_min": R_min, "R": R,
            "N_th": N_th,
            "N_real": N_real,
            "N_R": N_R, "N_S": N_S,
            "feed_stage": feed_stage,
            "efficiency": efficiency,
        }
