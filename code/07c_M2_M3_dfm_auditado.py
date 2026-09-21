"""
07c_M2_M3_dfm_auditado.py

M2 y M3 re-auditados con el mismo principio de Lenza-Primiceri aplicado a
M1: (a) estimar los parametros del DFM (cargas, phi, R) SOLO con datos
limpios (hasta dic-2019, sin ningun mes afectado por COVID), para que las
lecturas extremas de 2020 no distorsionen las cargas -- esto es
probablemente la causa de un resultado raro que encontre en 07b/08b (IMSS
con R~0.04, tratado casi sin ruido, mientras ANTAD/AUTOS con R~15-440,
tratados como puro ruido, a pesar de ser las series que mas informacion
dan sobre el choque); y (b) al FILTRAR hacia adelante (pronosticar
mayo/junio), aplicar un factor de escala tipo Lenza-Primiceri sobre Q (la
varianza de innovacion del estado) en vez de usar el Q historico
(calibrado en tiempos normales, que subestima la incertidumbre de
extrapolar un choque sin precedente).

Metodologia identica a 06c (M1 auditado):
  - parametros base: SOLO datos limpios (hasta dic-2019).
  - filtrado: sobre TODOS los datos reales disponibles (incluye ene-abr
    2020), usando los parametros limpios -- esto SI usa la informacion de
    2020 para actualizar el ESTADO, solo no para estimar los parametros
    estructurales.
  - pronostico: Q escalado por un factor s (ancla via Italia para mayo,
    igual que en 06c) que decae geometricamente con rho=0.8^3=0.512
    (mismo valor, tomado del prior del paper, no inventado de nuevo).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m2mod = import_module("07b_M2_dfm_robusto")
m3mod = import_module("08b_M3_dfm_robusto")

ITALIA_Q1_2020_SHOCK_PCT = 4.7
RHO_QUARTERLY_EQUIV = 0.8 ** 3  # mismo valor usado en 06c, no re-inventado
CLEAN_CUTOFF = pd.Timestamp("2019-12-01")
N_DRAWS = 10_000


def em_dfm_clean(panel, monthly_cols, em_func, a0=3.0, needs_cols=True):
    """Estima los parametros SOLO con panel hasta CLEAN_CUTOFF, pero
    despues filtra sobre el panel COMPLETO (incluyendo 2020) con esos
    parametros fijos, para obtener el estado actualizado sin que 2020
    haya distorsionado las cargas."""
    panel_clean = panel.loc[:CLEAN_CUTOFF]
    if needs_cols:
        fit_clean_params = em_func(panel_clean, monthly_cols, a0=a0, verbose=False)
    else:
        fit_clean_params = em_func(panel_clean, a0=a0, verbose=False)

    # Reconstruir F, Q, H, R con los parametros limpios y correr el filtro
    # sobre el panel COMPLETO (incluye 2020).
    phi, lam, R = fit_clean_params["phi"], fit_clean_params["lam"], fit_clean_params["R"]
    F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])
    Q = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    H_dict = {c: np.array([lam[c], 0, 0]) for c in monthly_cols}
    H_dict["PIB_q"] = np.array([1 / 3, 1 / 3, 1 / 3])
    R_dict = dict(R)

    cols_available = []
    for t in range(len(panel)):
        row = panel.iloc[t]
        cols_available.append([c for c in panel.columns if not np.isnan(row[c])])

    x_filt, P_filt, x_pred, P_pred, ll = m2mod.kalman_filter_missing(
        panel, cols_available, H_dict, R_dict, F, Q)

    return dict(phi=phi, lam=lam, R=R, Q=Q, F=F, x_filt=x_filt, P_filt=P_filt, loglik=ll)


def forecast_with_lp_scale(fit, panel_index, target_last_month, s0, rho, n_draws=N_DRAWS, seed=1101):
    """Pronostica desde el ultimo mes disponible hasta completar el
    trimestre objetivo, escalando Q por s_t = 1+(s0-1)*rho^(j-1) en vez de
    Q=1 fijo (j=1 en el primer mes pronosticado)."""
    phi = fit["phi"]
    last_i = len(panel_index) - 1
    s_last, P_last = fit["x_filt"][last_i], fit["P_filt"][last_i]
    h_months = (target_last_month.to_period("M").ordinal - panel_index[-1].to_period("M").ordinal)

    rng = np.random.default_rng(seed)
    xs_draws = np.zeros((n_draws, h_months + 1))
    for d in range(n_draws):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        xs = [s_d[0]]
        for j in range(1, h_months + 1):
            s_j = 1 + (s0 - 1) * rho ** (j - 1)
            xs.append(phi * xs[-1] + np.sqrt(s_j) * rng.standard_normal())
        xs_draws[d] = xs
    return xs_draws


def bridge_and_growth(xs_draws, panel_index, target_quarter, pib_qoq, factor_hist, n_draws=N_DRAWS, seed=1102):
    factor_q = factor_hist.resample("QS").mean()
    factor_q_n = factor_hist.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)

    rng = np.random.default_rng(seed)
    n_cols = xs_draws.shape[1]
    f_q = xs_draws[:, -min(3, n_cols):].mean(axis=1)
    coef_draws = np.array([rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
                            for _ in range(n_draws)])
    growth = coef_draws[:, 0] + coef_draws[:, 1] * f_q + rng.normal(0, sigma_b, n_draws)
    return growth, coef_b, sigma_b


def main():
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)

    with open(OUTPUT / "M0_fit.pkl", "rb") as f:
        m0 = pickle.load(f)
    normal_sd = 100 * np.sqrt(m0["fit"]["Sigma_hat"][0, 0])
    s0_may = (ITALIA_Q1_2020_SHOCK_PCT / normal_sd) ** 2  # multiplicador de VARIANZA (Q ya esta en esa escala)
    print(f"s0 (ancla Italia, multiplicador de VARIANZA para el estado) = {s0_may:.2f}")

    # === M2 auditado ===
    print("\n=== M2 auditado (parametros limpios + escala LP en pronostico) ===")
    panel_m2 = m2mod.build_data()
    fit_m2 = em_dfm_clean(panel_m2, ["IMSS", "ANTAD", "AUTOS"], m2mod.em_dfm, needs_cols=False)
    print("Cargas limpias:", {k: round(v, 3) for k, v in fit_m2["lam"].items()})
    print("R limpias:", {k: round(v, 3) for k, v in fit_m2["R"].items()})
    xs_m2 = forecast_with_lp_scale(fit_m2, panel_m2.index, pd.Timestamp("2020-06-01"), s0_may, RHO_QUARTERLY_EQUIV)
    factor_hist_m2 = pd.Series(fit_m2["x_filt"][:, 0], index=panel_m2.index)
    growth_m2, coef_m2, sigma_m2 = bridge_and_growth(xs_m2, panel_m2.index, pd.Timestamp("2020-04-01"),
                                                       pib_qoq, factor_hist_m2, seed=1102)
    print(f"M2 auditado mayo: mediana={np.median(growth_m2):.2f}%  "
          f"IC95%=[{np.percentile(growth_m2,2.5):.2f}, {np.percentile(growth_m2,97.5):.2f}]")

    # === M3 auditado (sin IGAE) ===
    print("\n=== M3 auditado (parametros limpios + escala LP en pronostico) ===")
    panel_m3_full = m3mod.build_panel_no_igae()
    monthly_cols_m3 = [c for c in panel_m3_full.columns if c != "PIB_q"]
    fit_m3 = em_dfm_clean(panel_m3_full, monthly_cols_m3, m3mod.em_dfm_general)
    print("Cargas limpias (ordenadas):",
          sorted({k: round(v, 3) for k, v in fit_m3["lam"].items()}.items(), key=lambda kv: -abs(kv[1])))
    xs_m3 = forecast_with_lp_scale(fit_m3, panel_m3_full.index, pd.Timestamp("2020-06-01"), s0_may, RHO_QUARTERLY_EQUIV)
    factor_hist_m3 = pd.Series(fit_m3["x_filt"][:, 0], index=panel_m3_full.index)
    growth_m3, coef_m3, sigma_m3 = bridge_and_growth(xs_m3, panel_m3_full.index, pd.Timestamp("2020-04-01"),
                                                       pib_qoq, factor_hist_m3, seed=1103)
    print(f"M3 auditado mayo: mediana={np.median(growth_m3):.2f}%  "
          f"IC95%=[{np.percentile(growth_m3,2.5):.2f}, {np.percentile(growth_m3,97.5):.2f}]")

    print("\n=== Comparacion con versiones anteriores ===")
    old_m2 = np.load(OUTPUT / "M2_dfm_robusto_growth_draws.npy")
    print(f"M2 (DFM robusto, contaminado + Q=1 fijo): mediana={np.median(old_m2):.2f}  "
          f"IC95=[{np.percentile(old_m2,2.5):.2f}, {np.percentile(old_m2,97.5):.2f}]")
    old_m3 = np.load(OUTPUT / "M3_dfm_robusto_growth_draws.npy")
    print(f"M3 (DFM robusto, contaminado + Q=1 fijo): mediana={np.median(old_m3):.2f}  "
          f"IC95=[{np.percentile(old_m3,2.5):.2f}, {np.percentile(old_m3,97.5):.2f}]")
    actual = -20.96
    print(f"\nCubre M2 auditado? {np.percentile(growth_m2,2.5)<=actual<=np.percentile(growth_m2,97.5)}")
    print(f"Cubre M3 auditado? {np.percentile(growth_m3,2.5)<=actual<=np.percentile(growth_m3,97.5)}")

    np.save(OUTPUT / "M2_auditado_growth_draws.npy", growth_m2)
    np.save(OUTPUT / "M3_auditado_growth_draws.npy", growth_m3)


if __name__ == "__main__":
    main()
