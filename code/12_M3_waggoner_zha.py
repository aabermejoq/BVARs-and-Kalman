"""
M3 = M2 + pronostico condicional (Waggoner y Zha, 1999, "Conditional
Forecasting in Dynamic Multivariate Models", Review of Economics and
Statistics) sobre escenarios epidemiologicos, mezclados con los pesos
bayesianos de backcast -- la formula del paper de referencia:

    p(Y | I_t) = sum_s w_s * p(Y | z_s, I_t)

IMPORTANTE: esta version construye sobre las PARTICULAS YA CORREGIDAS de
M2 (abril/mayo, nowcast con nowcast via indice de choque + correccion no
lineal + filtro de particulas de Gordon) -- no reinicia desde la linea
base ciega del VAR. Junio-diciembre se obtiene condicionando, PARTICULA
POR PARTICULA, la trayectoria de IMSS_mom (variable real del VAR, con
dinamica propia estimada) a la meta de cada escenario epidemiologico +
GDELT, resolviendo la distribucion de choques estructurales consistente
con esa restriccion (formula cerrada de condicionamiento gaussiano) y
propagando el efecto a IGAE via la dinamica cruzada YA ESTIMADA del VAR
(no un ajuste aditivo directo sobre IGAE).

R[j][k] = dY_{t+j}/du_{t+k}: representacion de promedio movil (impulse
response) del VAR, obtenida iterando su forma companera.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
m1 = import_module("04_M1_kalman_sv")
m2 = import_module("05_M2_particle_lp")
epi = import_module("06_M3_epidemic_scenarios")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

TILT_COEF = 0.30
GDELT_TILT_COEF = 0.30
WZ_START_MONTH = pd.Timestamp("2020-06-01")  # abril/mayo ya cubiertos por el nowcast de M2


def companion_matrix(fit0):
    names = fit0["names"]
    n, p = len(names), fit0["p"]
    A_blocks = []
    for lag in range(1, p + 1):
        A_lag = np.zeros((n, n))
        for i, v in enumerate(names):
            b = fit0["equations"][v]["b_post"]
            col0 = 1 + (lag - 1) * n
            A_lag[i, :] = b[col0:col0 + n]
        A_blocks.append(A_lag)
    Cmp = np.zeros((n * p, n * p))
    Cmp[:n, :] = np.hstack(A_blocks)
    if p > 1:
        Cmp[n:, :-n] = np.eye(n * (p - 1))
    return Cmp, n, p


def irf_matrices(Cmp, n, p, h):
    """R[j][k], j,k=0..h-1 (0-indexado), R[j][k]=0 si k>j."""
    Sel = np.zeros((n, n * p))
    Sel[:, :n] = np.eye(n)
    powers = [np.eye(n * p)]
    for _ in range(h):
        powers.append(powers[-1] @ Cmp)
    R = [[np.zeros((n, n)) for _ in range(h)] for _ in range(h)]
    for j in range(h):
        for k in range(j + 1):
            R[j][k] = Sel @ powers[j - k] @ Sel.T
    return R


def wz_setup(R, n, h, Sigma_normal, cond_idx):
    """Precomputa K = Sigma_big A' (A Sigma_big A')^-1 (mapea r -> media de
    u_stack) y cov_u/cov_path (compartidos por todas las particulas y
    escenarios: solo dependen de la estructura del VAR, no de r)."""
    A = np.zeros((h, h * n))
    for j in range(h):
        for k in range(j + 1):
            A[j, k * n:(k + 1) * n] = R[j][k][cond_idx, :]
    Sigma_big = np.kron(np.eye(h), Sigma_normal)
    AS = A @ Sigma_big
    M = AS @ A.T
    M_inv = np.linalg.inv(M + 1e-12 * np.eye(h))
    K = Sigma_big @ A.T @ M_inv  # (h*n) x h
    cov_u = Sigma_big - K @ A @ Sigma_big

    cov_path = np.zeros((h, n, n))
    for j in range(h):
        Rj = np.hstack([R[j][k] for k in range(j + 1)] + [np.zeros((n, n * (h - j - 1)))])
        cov_path[j] = Rj @ cov_u @ Rj.T
    return K, cov_path


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]
    fit0 = m0.fit_bvar_minnesota(est_df)
    i_igae = m0.VARS.index("IGAE_mom")
    i_imss = m0.VARS.index("IMSS_mom")

    sv_fit = m1.fit_sv(None)
    Sigma_normal = sv_fit["Sigma_normal"]
    s0_barra = sv_fit["s0_barra"]

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)

    print("Corriendo M2 (nowcast abril/mayo + filtro de particulas) como base de M3...")
    m2_summary, particles, m2_months = m2.run()  # particles: N x horizon x n
    N = particles.shape[0]

    wz_months = [m for m in horizon_months if m >= WZ_START_MONTH]
    h_wz = len(wz_months)
    wz_start_h = list(horizon_months).index(wz_months[0])

    Cmp, n, p = companion_matrix(fit0)
    R = irf_matrices(Cmp, n, p, h_wz)
    K, cov_path = wz_setup(R, n, h_wz, Sigma_normal, i_imss)

    # --- escenarios epidemiologicos (backcast bayesiano real) ---
    weights_s, paths_df, g0 = epi.build_scenarios()
    weekly_new, _ = epi.observed_weekly_growth(n_weeks=1)
    last_level = weekly_new.iloc[-1]
    monthly_rel = epi.monthly_case_index(paths_df, last_level)
    pressure = monthly_rel.reindex(wz_months, method="nearest").fillna(0.0)
    scenario_names = list(weights_s.index)

    # --- GDELT (forzado, documentado) ---
    gdelt_path = PROC / "gdelt_monthly.csv"
    gdelt_z = pd.Series(0.0, index=wz_months)
    gdelt_r2_mean = gdelt_r2_var = None
    if gdelt_path.exists():
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        baseline_tone = gdelt.loc["2017-01-01":"2019-12-31", "tono_promedio"]
        mu_g, sd_g = baseline_tone.mean(), baseline_tone.std()
        gdelt_z = ((gdelt["tono_promedio"] - mu_g) / sd_g).reindex(wz_months).fillna(0.0)

    s0_imss = np.sqrt(Sigma_normal[i_imss, i_imss])

    target_imss = {}
    for sc in scenario_names:
        tp = np.zeros(h_wz)
        for j, month in enumerate(wz_months):
            tilt_epi = -TILT_COEF * s0_imss * np.tanh(pressure.loc[month, sc])
            tilt_gdelt = -GDELT_TILT_COEF * s0_imss * np.tanh(gdelt_z.loc[month])
            tp[j] = tilt_epi + tilt_gdelt  # DESVIACION respecto al baseline de cada particula
        target_imss[sc] = tp
        print(f"Escenario {sc}: desviacion meta IMSS jun-dic = {np.round(tp, 4)}")

    betas_mean = {v: fit0["equations"][v]["b_post"] for v in m0.VARS}
    rng = np.random.default_rng(m0.SEED + 7)
    scenario_labels = rng.choice(len(scenario_names), size=N, p=weights_s.values)

    m3_draws = particles.copy()
    for i in range(N):
        # historia de la particula i al llegar a mayo (P=3 ultimos meses: marzo,abril,mayo)
        hist = est_df[m0.VARS].values[-m0.P:].tolist()
        for hh in range(wz_start_h):
            hist.append(particles[i, hh, :].tolist())
        hist = hist[-m0.P:]

        baseline_i = np.zeros((h_wz, n))
        for j in range(h_wz):
            row = [1.0]
            for lag in range(1, m0.P + 1):
                row.extend(hist[-lag])
            row = np.array(row)
            mean = np.array([betas_mean[v] @ row for v in m0.VARS])
            baseline_i[j] = mean
            hist.append(mean.tolist())

        sc = scenario_names[scenario_labels[i]]
        r_i = target_imss[sc]  # ya es una desviacion (r = target - baseline_cond_var), ver arriba
        mean_u = K @ r_i  # (h_wz*n)-vector
        mean_path_i = baseline_i.copy()
        for j in range(h_wz):
            Rj = np.hstack([R[j][k] for k in range(j + 1)] + [np.zeros((n, n * (h_wz - j - 1)))])
            mean_path_i[j] += Rj @ mean_u

        for j in range(h_wz):
            var_igae = max(cov_path[j][i_igae, i_igae], 1e-10)
            m3_draws[i, wz_start_h + j, i_igae] = rng.normal(mean_path_i[j, i_igae], np.sqrt(var_igae))

    igae_draws = m3_draws[:, :, i_igae]
    out = pd.DataFrame(igae_draws.T, index=horizon_months)
    out.to_csv(PROC / "M3_igae_mom_draws.csv")

    summary = pd.DataFrame(
        {
            "p05": np.percentile(igae_draws, 5, axis=0),
            "p25": np.percentile(igae_draws, 25, axis=0),
            "mediana": np.percentile(igae_draws, 50, axis=0),
            "p75": np.percentile(igae_draws, 75, axis=0),
            "p95": np.percentile(igae_draws, 95, axis=0),
        },
        index=horizon_months,
    )
    summary.to_csv(PROC / "M3_summary.csv")
    print("\n=== M3 (M2 + Waggoner-Zha sobre IMSS, mezcla de escenarios) ===")
    print(summary.round(4))

    meta = dict(
        metodo="M2 (nowcast+particulas) + Waggoner-Zha (1999) condicionando IMSS_mom jun-dic",
        tilt_coef=TILT_COEF, gdelt_tilt_coef=GDELT_TILT_COEF,
        s0_barra_lenza_primiceri=float(s0_barra),
        scenario_weights=weights_s.to_dict(),
    )
    with open(PROC / "M3_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return summary, igae_draws, horizon_months


if __name__ == "__main__":
    run()
