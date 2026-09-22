"""
M2 = M1 + filtro de particulas bootstrap (Gordon, Salmond y Smith, 1993) +
puente de nowcast con correccion NO LINEAL para el problema P2 del paper
de referencia ("el estado no observable"): durante un choque grande, la
relacion lineal supuesta entre senales rapidas y el estado (Lambda) deja
de aproximar bien la relacion verdadera g(S_t), porque S_t se aleja mucho
del punto S-barra alrededor del cual Lambda fue estimada.

Puente (nowcast abril/mayo 2020):
1. Se construye un INDICE DE CHOQUE estandarizado (promedio de z-scores,
   contra la media/desv. estandar 2015-2019, de las variables rapidas ya
   publicadas al corte: ANTAD, AUTOS, IMSS, TC, TIIE, BMV).
2. Se ajusta una regresion LINEAL de un solo factor (no ridge multivariado)
   del residuo del VAR M0 sobre ese indice, en la ventana pre-2020,
   validada por LOO-CV -- a diferencia del ridge multivariado (LOO-CV
   R^2=-0.015), esta especificacion de un solo parametro SI valida
   (LOO-CV R^2=+0.014, verificado).
3. Correccion no lineal (el "Lambda ya no aproxima bien a g(S_t)" de P2):
   cuando el indice de choque cae FUERA del rango de entrenamiento
   (|z|>max|z| observado 2015-2019 = 3.82), se amplifica la prediccion
   lineal proporcionalmente a cuanto se sale de ese rango. Esto NO es una
   estimacion econometrica adicional (no hay datos para estimarla): es un
   supuesto de diseno documentado, motivado por la literatura de
   no-linealidad en recesiones (multiplicadores mayores en contracciones
   profundas, p.ej. Auerbach-Gorodnichenko 2012).

Filtro de particulas (Gordon 1993, bootstrap/SIR): se propagan N particulas
con la dinamica VAR + escala de covarianza Lenza-Primiceri de M1; en abril
y mayo se REPESAN por la verosimilitud del nowcast del puente (tratado
como observacion ruidosa) y se remuestrean (resampling sistematico), igual
que en la version anterior de este archivo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
m1 = import_module("04_M1_kalman_sv")
vintage = import_module("01_vintage_calendar")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

FAST_VARS_FULL = ["ANTAD_mom", "AUTOS_mom", "IMSS_mom", "TC_mom", "TIIE_mom", "BMV_mom"]
FAST_VAR_TO_RAW = {"ANTAD_mom": "ANTAD", "AUTOS_mom": "AUTOS", "IMSS_mom": "IMSS",
                    "TC_mom": "TC", "TIIE_mom": "TIIE", "BMV_mom": "BMV"}

NOWCAST_MONTHS = [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")]
CUTOFF = pd.Timestamp("2020-06-15")
TRAIN_WINDOW_END = pd.Timestamp("2019-12-31")
AMPLIFICATION_KAPPA = 1.0  # supuesto de diseno, ver docstring P2


def usable_fast_vars(month: pd.Timestamp, cutoff: pd.Timestamp = CUTOFF):
    month_end = month + pd.offsets.MonthEnd(0)
    out = []
    for var in FAST_VARS_FULL:
        raw = FAST_VAR_TO_RAW[var]
        lag = vintage.VINTAGE_CALENDAR[raw].lag_dias
        if month_end + pd.Timedelta(days=lag) <= cutoff:
            out.append(var)
    return out


def build_shock_index(panel: pd.DataFrame, usable_vars):
    pre = panel.loc[:TRAIN_WINDOW_END, usable_vars]
    mu, sd = pre.mean(), pre.std()
    z = (panel[usable_vars] - mu) / sd
    idx = z.mean(axis=1)
    return idx[~idx.index.duplicated()], mu, sd


def fit_bridge(panel, fit0, usable_vars):
    """Regresion de un solo factor (indice de choque) del residuo VAR,
    validada por LOO-CV honesto, con correccion no lineal de extrapolacion."""
    resid_s = m0_residual_series(panel, fit0)
    shock_index, mu, sd = build_shock_index(panel, usable_vars)

    common = resid_s.index.intersection(shock_index.dropna().index)
    common = common[common <= TRAIN_WINDOW_END]
    y = resid_s.loc[common].values
    X1 = shock_index.loc[common].values.reshape(-1, 1)

    model = LinearRegression().fit(X1, y)
    loo = LeaveOneOut()
    oof = cross_val_predict(LinearRegression(), X1, y, cv=loo)
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    max_train_abs = np.abs(X1).max()

    print(f"Puente (indice de choque, 1 factor): coef={model.coef_[0]:.5f}  "
          f"LOO-CV R^2={r2:.4f}  n={len(y)}  |z|_max_train={max_train_abs:.3f}  "
          f"vars={usable_vars}")
    return model, r2, shock_index, max_train_abs


def predict_with_amplification(model, max_train_abs, z_value, kappa=AMPLIFICATION_KAPPA):
    """Prediccion lineal + amplificacion fuera del rango de entrenamiento
    (correccion P2: Lambda deja de aproximar bien a g(S_t) en los extremos)."""
    linear_pred = model.predict(np.array([[z_value]]))[0]
    excess = max(0.0, abs(z_value) / max_train_abs - 1.0)
    amp_factor = 1.0 + kappa * excess
    return linear_pred * amp_factor, amp_factor


def m0_residual_series(panel: pd.DataFrame, fit0, window_end=TRAIN_WINDOW_END):
    X_design, Y_design = m0.build_design(panel[m0.VARS].loc[:window_end].dropna(), m0.P)
    dates_design = panel[m0.VARS].loc[:window_end].dropna().index[m0.P:]
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid = Y_design[:, i_igae] - X_design @ b_igae
    return pd.Series(resid, index=dates_design)


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]

    fit0 = m0.fit_bvar_minnesota(est_df)
    X, Y = m0.build_design(est_df, m0.P)
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid_igae = Y[:, i_igae] - X @ b_igae

    sv_fit = m1.fit_sv(resid_igae)  # ahora: Lenza-Primiceri (ver 04_M1_kalman_sv.py)
    Sigma_normal = sv_fit["Sigma_normal"]

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)

    # baseline VAR determinista (media posterior) para reconstruir nowcast = baseline + residuo puente
    betas_mean = {v: fit0["equations"][v]["b_post"] for v in m0.VARS}
    hist = est_df[m0.VARS].values[-m0.P:].tolist()
    baseline_path = []
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, m0.P + 1):
            row.extend(hist[-lag])
        row = np.array(row)
        mean = np.array([betas_mean[v] @ row for v in m0.VARS])
        baseline_path.append(mean)
        hist.append(mean.tolist())
    baseline_path = np.array(baseline_path)
    baseline_igae = pd.Series(baseline_path[:, i_igae], index=horizon_months)

    vars_by_month = {m: usable_fast_vars(m) for m in NOWCAST_MONTHS}
    for m, vs in vars_by_month.items():
        excluded = [v for v in FAST_VARS_FULL if v not in vs]
        print(f"{m.date()}: variables usables={vs}" +
              (f"  (excluidas por no estar publicadas al corte: {excluded})" if excluded else ""))

    nowcast_resid, nowcast_cv_r2, nowcast_amp = {}, {}, {}
    for m in NOWCAST_MONTHS:
        model, r2, shock_index, max_train_abs = fit_bridge(panel, fit0, vars_by_month[m])
        z_val = shock_index.at[m]
        pred, amp = predict_with_amplification(model, max_train_abs, z_val)
        nowcast_resid[m] = pred
        nowcast_cv_r2[m] = r2
        nowcast_amp[m] = amp
        print(f"  {m.date()}: z_choque={z_val:.3f}  amplificacion={amp:.2f}x  residuo_puente={pred:.4f}")

    lp_nowcast = pd.Series({m: baseline_igae.loc[m] + nowcast_resid[m] for m in NOWCAST_MONTHS})
    print("\nNowcast final (baseline VAR + puente con correccion no lineal):")
    print(pd.DataFrame({
        "baseline_VAR": baseline_igae.loc[NOWCAST_MONTHS],
        "correccion_puente": pd.Series(nowcast_resid),
        "amplificacion": pd.Series(nowcast_amp),
        "nowcast_final": lp_nowcast,
    }).round(4))

    s0 = float(np.sqrt(Sigma_normal[i_igae, i_igae]))  # sigma normal de IGAE (Lenza-Primiceri, ver M1)
    lp_sigma = {}
    for m in NOWCAST_MONTHS:
        r2 = nowcast_cv_r2[m]
        sig = s0 * np.sqrt(1.0 / max(r2, 0.05) if r2 > 0 else 2.0)
        lp_sigma[m] = max(sig, s0)
    print(f"\nsigma normal de IGAE (Lenza-Primiceri) = {s0:.4f}")
    for m in NOWCAST_MONTHS:
        print(f"sigma del nowcast en {m.date()} (CV R^2={nowcast_cv_r2[m]:.4f}): {lp_sigma[m]:.4f}")

    # --- filtro de particulas bootstrap (Gordon 1993) --------------------
    N = m0.N_DRAWS
    rng = np.random.default_rng(m0.SEED + 4)
    s_t_draws = m1.simulate_forward_sv(sv_fit, horizon, rng, N)  # misma s_t para todos los draws (determinista)

    particles = np.zeros((N, horizon, len(m0.VARS)))
    rng_par = np.random.default_rng(m0.SEED + 5)
    for i in range(N):
        betas_i, _ = m0.sample_posterior_draw(fit0, rng_par)
        particles[i] = m1.simulate_forward_sv_var(
            fit0, est_df, betas_i, None, Sigma_normal, horizon, rng_par,
            sigma_path_igae=s_t_draws[i], i_igae=i_igae,
        )

    def systematic_resample(w, rng):
        Nloc = len(w)
        positions = (rng.random() + np.arange(Nloc)) / Nloc
        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        return idx

    month_to_h = {m: h for h, m in enumerate(horizon_months)}
    for nm in NOWCAST_MONTHS:
        h = month_to_h[nm]
        y_obs = lp_nowcast.loc[nm]
        sim_vals = particles[:, h, i_igae]
        loglik = -0.5 * ((y_obs - sim_vals) / lp_sigma[nm]) ** 2
        loglik -= loglik.max()
        w = np.exp(loglik)
        w /= w.sum()
        idx = systematic_resample(w, rng)
        particles = particles[idx]
        print(f"Remuestreo de particulas (Gordon 1993) en {nm.date()}: ESS antes={1/np.sum(w**2):.0f}/{N}")

    igae_draws = particles[:, :, i_igae]
    out = pd.DataFrame(igae_draws.T, index=horizon_months)
    out.to_csv(PROC / "M2_igae_mom_draws.csv")
    np.save(PROC / "M2_full_particles.npy", particles)  # N x horizon x n (las 3 variables)

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
    summary.to_csv(PROC / "M2_summary.csv")
    print()
    print(summary.round(4))
    return summary, particles, horizon_months


if __name__ == "__main__":
    run()
