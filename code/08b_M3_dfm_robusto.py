"""
08b_M3_dfm_robusto.py

Rehace M3 con el mismo DFM via EM regularizado de 07b, generalizado a un
panel de 9 series (en vez de PCA), EXCLUYENDO IGAE del panel de factores.

Motivo de excluir IGAE (senalado por el usuario): IGAE es el indicador
mensual OFICIAL de actividad economica de INEGI, construido para ser un
proxy casi directo del PIB -- no es una serie "independiente" que aporte
informacion nueva sobre el estado de la economia, es casi el propio target
medido con mas frecuencia. Incluirlo en un panel de factores junto a
indicadores genuinamente independientes (IMSS, ANTAD, AUTOS, comercio
exterior) corre el riesgo de que el factor dependa casi enteramente de
IGAE (por su correlacion casi perfecta con PIB, 0.97) en vez de aprender
una senal comun genuina de multiples fuentes independientes -- ademas de
que IGAE se apaga en feb-2020, exactamente cuando mas se necesita
informacion.

Panel resultante (9 series): ActividadIndustrial, FBCF, IMCP,
Exportaciones, Importaciones, IMSS, ANTAD, AUTOS, INDPRO_EEUU -- cada una
enmascarada a su propio rezago real-time (igual que en 08_M3).
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
m3mod = import_module("08_M3_high_frequency_state")
m2b = import_module("07b_M2_dfm_robusto")

N_DRAWS = 10_000
RNG_SEED = 72345
MAX_EM_ITER = 100
TOL = 1e-5


def build_panel_no_igae():
    panel = m3mod.build_panel()
    panel = panel.drop(columns=["IGAE"])
    return panel


def em_dfm_general(panel, monthly_cols, a0=3.0, verbose=True):
    prior_scale = {c: m2b.ar1_variance(panel[c].dropna().values) for c in monthly_cols}
    prior_scale["PIB_q"] = panel["PIB_q"].dropna().var() if "PIB_q" in panel.columns else 1.0

    cols_available = []
    for t in range(len(panel)):
        row = panel.iloc[t]
        cols_available.append([c for c in panel.columns if not np.isnan(row[c])])

    phi = 0.3
    lam = {c: 1.0 for c in monthly_cols}
    R = {c: prior_scale[c] for c in monthly_cols}
    R["PIB_q"] = prior_scale["PIB_q"]
    Q = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])
    H_PIB = np.array([1 / 3, 1 / 3, 1 / 3])
    prev_ll = -np.inf
    it = 0

    for it in range(MAX_EM_ITER):
        H_dict = {c: np.array([lam[c], 0, 0]) for c in monthly_cols}
        H_dict["PIB_q"] = H_PIB
        R_dict = dict(R)

        x_filt, P_filt, x_pred, P_pred, ll = m2b.kalman_filter_missing(
            panel, cols_available, H_dict, R_dict, F, Q)
        x_s, P_s = m2b.rts_smoother(x_filt, P_filt, x_pred, P_pred, F)

        new_lam, new_R = {}, {}
        for c in monthly_cols:
            obs_t = [t for t in range(len(panel)) if c in cols_available[t]]
            Sxx = sum(P_s[t, 0, 0] + x_s[t, 0] ** 2 for t in obs_t)
            Sxy = sum(x_s[t, 0] * panel.iloc[t][c] for t in obs_t)
            lam_new = Sxy / Sxx
            ss = sum((panel.iloc[t][c] - lam_new * x_s[t, 0]) ** 2 + (lam_new ** 2) * P_s[t, 0, 0]
                     for t in obs_t)
            n_i = len(obs_t)
            b0 = (a0 + 1) * prior_scale[c]
            R_new = (ss + 2 * b0) / (n_i + 2 * a0 + 2)
            new_lam[c], new_R[c] = lam_new, R_new

        obs_q = [t for t in range(len(panel)) if "PIB_q" in cols_available[t]]
        ss_q = sum((panel.iloc[t]["PIB_q"] - H_PIB @ x_s[t]) ** 2 + H_PIB @ P_s[t] @ H_PIB
                   for t in obs_q)
        b0_q = (a0 + 1) * prior_scale["PIB_q"]
        R_q_new = (ss_q + 2 * b0_q) / (len(obs_q) + 2 * a0 + 2)

        num = sum(P_s[t, 0, 1] + x_s[t, 0] * x_s[t, 1] for t in range(1, len(panel)))
        den = sum(P_s[t, 1, 1] + x_s[t, 1] ** 2 for t in range(1, len(panel)))
        phi_new = float(np.clip(num / den, -0.995, 0.995))

        lam, R, phi = new_lam, dict(new_R, PIB_q=R_q_new), phi_new
        F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])

        if verbose and (it % 20 == 0 or it == MAX_EM_ITER - 1):
            print(f"  EM iter {it}: loglik={ll:.2f}  phi={phi:.3f}")

        if abs(ll - prev_ll) < TOL * abs(prev_ll if prev_ll != 0 else 1):
            break
        prev_ll = ll

    return dict(phi=phi, lam=lam, R=R, Q=Q, F=F, x_filt=x_filt, P_filt=P_filt,
                x_smooth=x_s, P_smooth=P_s, loglik=ll, n_iter=it + 1)


def main():
    panel = build_panel_no_igae()
    monthly_cols = [c for c in panel.columns if c != "PIB_q"]
    print(f"Panel M3 (sin IGAE): {len(monthly_cols)} series -> {monthly_cols}")
    print(f"Rango: {panel.index.min().date()} -> {panel.index.max().date()} (n={len(panel)})")

    print("\nAjustando DFM via EM regularizado...")
    fit = em_dfm_general(panel, monthly_cols)

    print(f"\nConvergencia en {fit['n_iter']} iteraciones, log-verosimilitud final={fit['loglik']:.2f}")
    print(f"phi={fit['phi']:.3f}")
    print("Cargas (lambda), ordenadas por magnitud:")
    for k, v in sorted(fit["lam"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v:.3f}  (R={fit['R'][k]:.3f})")

    x_filt = fit["x_filt"][:, 0]
    print("\nEstado FILTRADO (ultimos 6 meses):")
    for d, x in list(zip(panel.index, x_filt))[-6:]:
        print(f"  {d.date()}: {x:+.2f}")

    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)
    factor_m = pd.Series(x_filt, index=panel.index)
    factor_q = factor_m.resample("QS").mean()
    factor_q_n = factor_m.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)
    print(f"\nEcuacion puente: PIB_q(%) = {coef_b[0]:.3f} + {coef_b[1]:.3f}*factor (sigma={sigma_b:.3f})")

    h_months = pd.Timestamp("2020-06-01").to_period("M").ordinal - panel.index[-1].to_period("M").ordinal
    rng = np.random.default_rng(RNG_SEED)
    phi = fit["phi"]
    s_last, P_last = fit["x_filt"][-1], fit["P_filt"][-1]
    growth_draws = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        xs = [s_d[0]]
        for _ in range(h_months):
            xs.append(phi * xs[-1] + rng.standard_normal())
        f_q2 = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth_draws[d] = coef_d[0] + coef_d[1] * f_q2 + rng.normal(0, sigma_b)

    summary = dict(
        modelo="M3_dfm_robusto", phi=float(phi),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)), p97_5=float(np.percentile(growth_draws, 97.5)),
        p10=float(np.percentile(growth_draws, 10)), p90=float(np.percentile(growth_draws, 90)),
    )
    print("\n=== M3 (DFM robusto, sin IGAE): pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    old = np.load(OUTPUT / "M3_growth_draws.npy")
    print(f"\nComparacion: M3 (PCA con IGAE) mediana={np.median(old):.2f}  "
          f"vs  M3 (DFM robusto sin IGAE) mediana={summary['mediana_pct']:.2f}")

    np.save(OUTPUT / "M3_dfm_robusto_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M3_dfm_robusto_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, fit=fit, bridge_coef=coef_b, bridge_sigma=sigma_b,
                          factor=factor_m, cols=monthly_cols), f)
    print(f"\nGuardado: {OUTPUT / 'M3_dfm_robusto_growth_draws.npy'}")


if __name__ == "__main__":
    main()
