"""
06b_M1_lenza_primiceri.py

M1 rediseñado siguiendo el espiritu de Lenza & Primiceri ("How to Estimate
a VAR after March 2020"): en vez de un AR(1) sobre la volatilidad log-
realizada (que revierte demasiado rapido, rho=0.23-0.59, estimado de una
muestra pre-COVID que nunca vio nada parecido), se usa un factor de escala
GRANDE en el trimestre de choque que DECAE GEOMETRICAMENTE hacia 1 en los
trimestres siguientes:

    Sigma_t = s_t * Sigma_hat,      s_t = 1 + (s_1 - 1) * rho^(t-1)

MAYO (pronostico de 2T20, el propio trimestre de choque):
  El multiplicador s_1 NO se puede medir de los datos -- el choque grande
  todavia no ha ocurrido en la muestra (1T20 lo diluyo, ver 06_M1). Se
  ancla con evidencia cross-country: el 30-abr-2020 Istat publico el PIB
  preliminar de 1T20 de Italia (-4.7% t/t, ya golpeada por COVID antes que
  Mexico -- informacion publica real al 15-may-2020). Se calibra s_1 como
  el multiplicador de VARIANZA que haria que un choque de esa magnitud
  fuera un evento de ~1 desviacion estandar bajo el Sigma_hat de Mexico,
  en vez de un evento de cola extremo:
      s_1 = (choque_italia / sd_normal_mexico)^2
  Como se pronostica exactamente el trimestre de choque (h=1), no aplica
  decaimiento todavia: se usa s_1 directamente.

AGOSTO (pronostico de 3T20, un trimestre DESPUES del choque ya conocido):
  s_1 SI se puede medir del residuo observado de 2T20 (7.12x, ver
  06_M1_volatility.py). Se aplica un decaimiento geometrico con rho=0.7
  (calibrado por juicio -- "gradual", no un ajuste rapido como el AR1
  original, pero tampoco persistencia total) para obtener s_2, el
  multiplicador aplicado al trimestre pronosticado (3T20).

Nota de honestidad: rho=0.7 es una eleccion declarada, inspirada en el
espiritu de Lenza-Primiceri, NO una replica de su valor estimado (no lo
verifique en el paper original).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"
OUTPUT_AGOSTO = ROOT / "output" / "models_agosto"

RHO_DECAY = 0.7
ITALIA_Q1_2020_SHOCK_PCT = 4.7  # Istat, publicado 30-abr-2020


def simulate(fit, data, VARS, P, scale, n_draws=10_000, seed=812):
    n = len(VARS)
    rng = np.random.default_rng(seed)
    Sigma_hat = fit["Sigma_hat"]
    L_hat = np.linalg.cholesky(Sigma_hat + 1e-10 * np.eye(n))
    last_obs = data[-P:][::-1]
    log_pib_t = data[-1, VARS.index("log_PIB")]
    growth = np.zeros(n_draws)
    for d in range(n_draws):
        beta_d = np.zeros((n * P + 1, n))
        for i in range(n):
            beta_d[:, i] = rng.multivariate_normal(fit["post_mean"][:, i], fit["post_var"][:, :, i])
        x = np.concatenate([last_obs[l] for l in range(P)] + [[1.0]])
        eps = scale * (L_hat @ rng.standard_normal(n))
        y_new = x @ beta_d + eps
        growth[d] = 100 * (np.exp(y_new[VARS.index("log_PIB")] - log_pib_t) - 1)
    return growth


def main():
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from importlib import import_module
    m0mod = import_module("05_M0_BVAR")

    # === MAYO: pronostico de 2T20 (el propio trimestre de choque) ===
    with open(OUTPUT / "M0_fit.pkl", "rb") as f:
        m0_may = pickle.load(f)
    fit_may = m0_may["fit"]
    core_may = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    data_may = core_may[m0mod.VARS].values

    normal_sd_may = 100 * np.sqrt(fit_may["Sigma_hat"][0, 0])
    s1_may = (ITALIA_Q1_2020_SHOCK_PCT / normal_sd_may) ** 2
    scale_may = np.sqrt(s1_may)
    print(f"MAYO: sd normal={normal_sd_may:.3f}%  s1 (ancla Italia)={s1_may:.2f}x  "
          f"escala SD={scale_may:.2f}x")

    growth_may = simulate(fit_may, data_may, m0mod.VARS, m0mod.P, scale_may, seed=812)
    print(f"M1-LP mayo: mediana={np.median(growth_may):.2f}%  "
          f"IC95%=[{np.percentile(growth_may,2.5):.2f}, {np.percentile(growth_may,97.5):.2f}]")
    old_may = np.load(OUTPUT / "M1_growth_draws.npy")
    print(f"  (M1 anterior, AR1 con reversion: mediana={np.median(old_may):.2f}%, "
          f"IC95%=[{np.percentile(old_may,2.5):.2f}, {np.percentile(old_may,97.5):.2f}])")
    actual_2t20 = -20.96
    cubre_may = np.percentile(growth_may, 2.5) <= actual_2t20 <= np.percentile(growth_may, 97.5)
    print(f"  Cubre el valor observado (2T20={actual_2t20}%)? {cubre_may}")

    np.save(OUTPUT / "M1_lenza_primiceri_growth_draws.npy", growth_may)

    # === AGOSTO: pronostico de 3T20 (un trimestre despues del choque conocido) ===
    with open(OUTPUT_AGOSTO / "core_info.pkl", "rb") as f:
        info_agosto = pickle.load(f)
    fit_agosto = info_agosto["fit0"]
    core_agosto = info_agosto["core"]
    data_agosto = core_agosto[m0mod.VARS].values
    h_T_agosto = info_agosto["info1"]["h_T"]
    s1_agosto = np.exp(h_T_agosto)
    s2_agosto = 1 + (s1_agosto - 1) * RHO_DECAY
    scale_agosto = np.sqrt(s2_agosto)
    print(f"\nAGOSTO: s1 (medido, 2T20)={s1_agosto:.2f}x  rho_decay={RHO_DECAY}  "
          f"s2 (aplicado a 3T20)={s2_agosto:.2f}x  escala SD={scale_agosto:.2f}x")

    growth_agosto = simulate(fit_agosto, data_agosto, m0mod.VARS, m0mod.P, scale_agosto, seed=813)
    print(f"M1-LP agosto: mediana={np.median(growth_agosto):.2f}%  "
          f"IC95%=[{np.percentile(growth_agosto,2.5):.2f}, {np.percentile(growth_agosto,97.5):.2f}]")
    old_agosto = np.load(OUTPUT_AGOSTO / "M1_growth_draws.npy")
    print(f"  (M1 anterior, AR1 con reversion: mediana={np.median(old_agosto):.2f}%, "
          f"IC95%=[{np.percentile(old_agosto,2.5):.2f}, {np.percentile(old_agosto,97.5):.2f}])")
    actual_3t20 = 14.47
    cubre_agosto = np.percentile(growth_agosto, 2.5) <= actual_3t20 <= np.percentile(growth_agosto, 97.5)
    print(f"  Cubre el valor observado (3T20={actual_3t20}%)? {cubre_agosto}")

    np.save(OUTPUT_AGOSTO / "M1_lenza_primiceri_growth_draws.npy", growth_agosto)

    with open(OUTPUT / "M1_lenza_primiceri_params.pkl", "wb") as f:
        pickle.dump(dict(s1_may=s1_may, scale_may=scale_may, s1_agosto=s1_agosto,
                          s2_agosto=s2_agosto, scale_agosto=scale_agosto, rho_decay=RHO_DECAY,
                          italia_shock=ITALIA_Q1_2020_SHOCK_PCT), f)
    print(f"\nGuardado: M1_lenza_primiceri_growth_draws.npy (mayo y agosto)")


if __name__ == "__main__":
    main()
