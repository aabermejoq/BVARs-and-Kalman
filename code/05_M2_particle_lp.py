"""
M2 = M1 + puente de Local Projections (LP) + filtro de particulas.

Puente LP: en vez de regresar IGAE_mom directamente sobre las variables
rapidas (senal univariada casi nula, ver 02_variable_selection.py), se
regresa el RESIDUO del VAR M0 (lo que la dinamica autorregresiva NO
explica) sobre el conjunto de indicadores rapidos disponibles al corte
(ANTAD, AUTOS, IMSS, TC, TIIE, BMV, Balanza, INDPRO; todas con rezago de
publicacion <= 26 dias, ver 01_vintage_calendar.py -- las lentas
FBCF/IMCP con 67 dias y ActividadIndustrial con el mismo rezago que IGAE se
excluyen del puente por no estar disponibles antes que el propio IGAE).
Ridge regularizado (lesson #6: nada de meter variables debiles sin
criterio) y validado por LOO-CV antes de usarse.

Filtro de particulas: se simulan N particulas hacia adelante con la
dinamica VAR+SV de M1: en abril y mayo de 2020 (unicos meses donde hay
nowcast real del puente LP, porque son los que ya tienen indicadores
rapidos publicados al corte) se REPESAN las particulas por su
verosimilitud frente al nowcast LP (tratado como observacion ruidosa de la
IGAE_mom real de ese mes) y se remuestrean (bootstrap particle filter,
resampling sistematico). El sigma de la observacion LP se ancla al
residuo REAL mas grande ya observado al corte (marzo 2020), escalado por
sqrt(s0), en vez de usar el error estandar in-sample del propio puente
(que subestimaria la incertidumbre en un choque sin precedente).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
m1 = import_module("04_M1_kalman_sv")
vintage = import_module("01_vintage_calendar")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

FAST_VARS = ["ANTAD_mom", "AUTOS_mom", "IMSS_mom", "TC_mom", "TIIE_mom",
             "BMV_mom", "Balanza_mom", "INDPRO_mom"]
FAST_VAR_TO_RAW = {  # nombre base en VINTAGE_CALENDAR (sin sufijo _mom)
    "ANTAD_mom": "ANTAD", "AUTOS_mom": "AUTOS", "IMSS_mom": "IMSS",
    "TC_mom": "TC", "TIIE_mom": "TIIE", "BMV_mom": "BMV",
    "Balanza_mom": "Balanza", "INDPRO_mom": "INDPRO",
}

NOWCAST_MONTHS = [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")]
CUTOFF = pd.Timestamp("2020-06-15")


def usable_fast_vars(month: pd.Timestamp, cutoff: pd.Timestamp = CUTOFF):
    """Que variables rapidas tienen el dato PROPIO de `month` ya publicado en
    `cutoff`, segun el calendario de vintage real (evita fuga de informacion:
    p.ej. Balanza e INDPRO de mayo 2020 NO estan publicadas para un corte del
    15 de junio 2020, aunque para abril si lo estan)."""
    month_end = month + pd.offsets.MonthEnd(0)
    out = []
    for var in FAST_VARS:
        raw = FAST_VAR_TO_RAW[var]
        lag = vintage.VINTAGE_CALENDAR[raw].lag_dias
        if month_end + pd.Timedelta(days=lag) <= cutoff:
            out.append(var)
    return out


def fit_lp_bridge(panel: pd.DataFrame, fit0, usable_vars, window_end=pd.Timestamp("2019-12-31")):
    """Ajusta ridge del residuo-VAR sobre el subconjunto de indicadores rapidos
    REALMENTE disponibles para ese horizonte (ver usable_fast_vars), ventana
    pre-corte (excluye el propio episodio 2020 que se va a pronosticar, para
    no hacer trampa)."""
    df_core = panel[m0.VARS].dropna()
    df_fast = panel[usable_vars].dropna()
    common = df_core.index.intersection(df_fast.index)
    common = common[common <= window_end]

    X_design, Y_design = m0.build_design(panel[m0.VARS].loc[:window_end].dropna(), m0.P)
    dates_design = panel[m0.VARS].loc[:window_end].dropna().index[m0.P:]
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    baseline = X_design @ b_igae
    resid = Y_design[:, i_igae] - baseline
    resid_s = pd.Series(resid, index=dates_design)

    common2 = resid_s.index.intersection(df_fast.index)
    y = resid_s.loc[common2].values
    Xf = df_fast.loc[common2].values

    # elegir alpha de ridge por LOO-CV (MSE; R^2 individual por fold no esta
    # definido con 1 sola observacion de prueba, ver UndefinedMetricWarning)
    alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    loo = LeaveOneOut()
    best_alpha, best_mse = alphas[0], np.inf
    for a in alphas:
        scores = cross_val_score(Ridge(alpha=a), Xf, y, cv=loo,
                                  scoring="neg_mean_squared_error")
        mse = -scores.mean()
        if mse < best_mse:
            best_mse, best_alpha = mse, a

    # R^2 agregado honesto: SS_res/SS_tot sobre TODAS las predicciones LOO juntas
    oof_pred = cross_val_predict(Ridge(alpha=best_alpha), Xf, y, cv=loo)
    ss_res = np.sum((y - oof_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    best_score = 1 - ss_res / ss_tot

    model = Ridge(alpha=best_alpha).fit(Xf, y)
    print(f"LP bridge: alpha={best_alpha}  LOO-CV R^2={best_score:.4f}  "
          f"(n={len(y)}; R^2<=0 significa que el puente NO mejora sobre la "
          f"media del residuo -- se reporta con honestidad)")
    return model, best_score, resid_s


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]

    fit0 = m0.fit_bvar_minnesota(est_df)
    X, Y = m0.build_design(est_df, m0.P)
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid_igae = Y[:, i_igae] - X @ b_igae
    sv_fit = m1.fit_sv(resid_igae)

    # conjunto de variables rapidas REALMENTE publicadas al corte, por mes
    # (distinto para abril y mayo -- ver docstring de usable_fast_vars)
    vars_by_month = {m: usable_fast_vars(m) for m in NOWCAST_MONTHS}
    for m, vs in vars_by_month.items():
        excluded = [v for v in FAST_VARS if v not in vs]
        print(f"{m.date()}: variables usables={vs}"
              + (f"  (excluidas por no estar publicadas al corte: {excluded})" if excluded else ""))

    lp_models = {}
    lp_cv_r2s = {}
    for m in NOWCAST_MONTHS:
        model_m, r2_m, _ = fit_lp_bridge(panel, fit0, vars_by_month[m])
        lp_models[m] = model_m
        lp_cv_r2s[m] = r2_m

    # baseline VAR (misma dinamica que M0/M1) para abril y mayo, para reconstruir
    # el nowcast LP = baseline_VAR + residuo_predicho_por_LP
    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)
    rng_base = np.random.default_rng(m0.SEED + 3)
    betas_point, _ = m0.sample_posterior_draw(fit0, np.random.default_rng(0))
    # usamos la MEDIA posterior (b_post) como baseline puntual, no un draw ruidoso
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
        hist.append(mean.tolist())  # baseline determinista: propaga la media
    baseline_path = np.array(baseline_path)
    baseline_igae = pd.Series(baseline_path[:, i_igae], index=horizon_months)

    lp_pred_resid = pd.Series(
        {m: lp_models[m].predict(panel.loc[[m], vars_by_month[m]].values)[0]
         for m in NOWCAST_MONTHS}
    )
    lp_nowcast = baseline_igae.loc[NOWCAST_MONTHS] + lp_pred_resid
    print("\nNowcast LP (baseline VAR + correccion puente):")
    print(pd.DataFrame({"baseline_VAR": baseline_igae.loc[NOWCAST_MONTHS],
                         "correccion_LP": lp_pred_resid,
                         "nowcast_LP": lp_nowcast}).round(4))

    # ancla de incertidumbre: mayor |residuo| REAL ya observado al corte (marzo 2020),
    # escalado por sqrt(s0) -- lesson #4 y #5. Sigma propio por mes segun el CV
    # R^2 de SU modelo (abril y mayo usan conjuntos de variables distintos).
    s0 = resid_igae[-1] ** 2  # varianza del residuo mas reciente observado (marzo 2020)
    lp_sigma = {}
    for m in NOWCAST_MONTHS:
        r2 = lp_cv_r2s[m]
        sig = np.sqrt(s0) * np.sqrt(1.0 / max(r2, 0.05) if r2 > 0 else 2.0)
        lp_sigma[m] = max(sig, resid_igae.std())
    print(f"\ns0 (var. del residuo mas reciente, marzo 2020) = {s0:.6f}")
    for m in NOWCAST_MONTHS:
        print(f"sigma del nowcast LP en {m.date()} (CV R^2={lp_cv_r2s[m]:.4f}): {lp_sigma[m]:.4f}")

    # --- filtro de particulas -------------------------------------------------
    N = m0.N_DRAWS
    rng = np.random.default_rng(m0.SEED + 4)

    # propone trayectorias con la dinamica VAR+SV de M1 (misma funcion que M1)
    sigma_draws = m1.simulate_forward_sv(sv_fit, horizon, rng, N)

    particles = np.zeros((N, horizon, len(m0.VARS)))
    weights = np.ones(N) / N
    betas_draws = []
    rng_par = np.random.default_rng(m0.SEED + 5)
    for i in range(N):
        betas_i, sigmas_i = m0.sample_posterior_draw(fit0, rng_par)
        betas_draws.append(betas_i)
        particles[i] = m1.simulate_forward_sv_var(
            fit0, est_df, betas_i, sigmas_i, fit0["corr"], horizon, rng_par,
            sigma_path_igae=sigma_draws[i], i_igae=i_igae,
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
        print(f"Remuestreo de particulas en {nm.date()}: ESS antes="
              f"{1/np.sum(w**2):.0f}/{N}")

    igae_draws = particles[:, :, i_igae]
    out = pd.DataFrame(igae_draws.T, index=horizon_months)
    out.to_csv(PROC / "M2_igae_mom_draws.csv")

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
    return summary, igae_draws, horizon_months


if __name__ == "__main__":
    run()
