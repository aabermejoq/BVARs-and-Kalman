"""
26_backtest_igae_15dias.py

Backtest recursivo, real-time, del nowcast de IGAE (m/m) con UN SOLO corte
de informacion por mes objetivo: 15 dias despues de cerrado el mes (a
peticion del usuario, tras ver los 3 cortes -- se queda solo con este).
Corre para TODOS los meses de 2019 a 2021 (36 meses) y compara cada
pronostico contra el IGAE realmente publicado despues.

Diseno (para que sea computacionalmente viable con 36 iteraciones):
  - Parametros del DFM (phi, lam, R) y el puente ML se ajustan UNA SOLA
    VEZ, con datos limpios hasta dic-2018 (seguro antes de CUALQUIER mes
    objetivo del rango 2019-2021) -- no se re-estiman por mes.
  - Por cada mes objetivo T: se construye el panel con el corte real-time
    T-fin-de-mes + 15 dias (mismo calendario de publicacion VERIFICADO de
    24_panel_mensual_igae.py), se filtra con los parametros fijos, se
    ancla LP al mayor residuo de estado observado en los ultimos 12 meses
    YA disponibles en ese corte (se adapta solo: grande en 2020, chico en
    tiempos normales), y se aplica el puente YA AJUSTADO.

Requirio extender BMV, ICSA, Google Trends y la tasa larga (antes solo
llegaban a agosto-2020) hasta principios de 2022.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "output" / "models_mensual_igae"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
vintage = import_module("02_real_time_vintage")
m2mod = import_module("19_M2_particulas_lp")
m3mod = import_module("08b_M3_dfm_robusto")

N_DRAWS = 5_000
RHO_MONTHLY = 0.8
TRAIN_CUTOFF = pd.Timestamp("2018-12-01")  # seguro antes de todo el rango 2019-2021
LAG_WINDOW_LP = 12  # meses hacia atras (ya disponibles en el corte) para anclar LP

NIVEL1_COLS = ["ActividadIndustrial", "FBCF", "IMCP", "Exportaciones", "Importaciones",
               "IMSS", "ANTAD", "AUTOS", "INDPRO_EEUU", "TIIE", "TC_dep", "TasaDesempleo_chg",
               "ICSA", "Trends_reapertura", "Trends_despidos", "BMV_yoy", "Spread_tasas"]
ALL_COLS = NIVEL1_COLS + ["IGAE_target"]

EXACT_DATES = {  # ver 24_panel_mensual_igae.py -- solo cubre 2020, resto usa el rezago generico verificado
    "IGAE_ActInd": {
        "2020-02-01": "2020-04-24", "2020-03-01": "2020-05-26", "2020-04-01": "2020-06-26",
        "2020-05-01": "2020-07-24", "2020-06-01": "2020-08-26", "2020-07-01": "2020-09-25",
    },
    "FBCF_IMCP": {
        "2020-02-01": "2020-05-06", "2020-03-01": "2020-06-05", "2020-04-01": "2020-07-06",
        "2020-05-01": "2020-08-06", "2020-06-01": "2020-09-07", "2020-07-01": "2020-10-06",
    },
    "Balanza": {"2020-04-01": "2020-05-26", "2020-05-01": "2020-06-26", "2020-06-01": "2020-07-27"},
    "ANTAD": {"2020-05-01": "2020-06-11"}, "AUTOS": {"2020-05-01": "2020-06-05"},
    "IMSS": {"2020-05-01": "2020-06-12"},
}


def _exact_or_registry(group, ref_month, variable, asof):
    exact = EXACT_DATES.get(group, {}).get(ref_month.strftime("%Y-%m-01"))
    if exact is not None:
        return pd.Timestamp(exact) <= asof
    return vintage.is_available(variable, ref_month, asof=asof)


_SERIES_CACHE = {}


def build_monthly_panel(cutoff_date, panel_end):
    if not _SERIES_CACHE:
        with open(INTERIM / "series_raw.pkl", "rb") as f:
            _SERIES_CACHE["series"] = pickle.load(f)
        _SERIES_CACHE["bmv"] = pd.read_csv(EXTERNAL / "bmv_2019_2021.csv", index_col=0, parse_dates=True).iloc[:, 0]
        _SERIES_CACHE["icsa"] = pd.read_csv(EXTERNAL / "icsa_2019_2021.csv", parse_dates=["date"]).set_index("date")["ICSA"]
        _SERIES_CACHE["lt"] = pd.read_csv(EXTERNAL / "mx_ltrate_2019_2021.csv",
                                           parse_dates=["observation_date"]).set_index("observation_date")["IRLTLT01MXM156N"]
        _SERIES_CACHE["trends"] = pd.read_csv(INTERIM / "trends_reapertura_despidos_2018_2022.csv",
                                               index_col=0, parse_dates=True)
        _SERIES_CACHE["indpro"] = pd.read_csv(EXTERNAL / "indpro_full_2022.csv", parse_dates=["date"]).set_index("date")["INDPRO"]

    series = _SERIES_CACHE["series"]

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    idx = pd.date_range("1993-01-01", panel_end, freq="MS")
    panel = pd.DataFrame(index=idx)

    def apply_cutoff(raw, group, variable):
        s = raw.reindex(idx)
        mask = pd.Series([_exact_or_registry(group, d, variable, cutoff_date) for d in idx], index=idx)
        return s.where(mask)

    panel["ActividadIndustrial"] = apply_cutoff(yoy(series["otros"]["ActividadIndustrial"]), "IGAE_ActInd", "ActividadIndustrial")
    panel["FBCF"] = apply_cutoff(yoy(series["otros"]["FBCF"]), "FBCF_IMCP", "FBCF")
    panel["IMCP"] = apply_cutoff(yoy(series["Consumo"]["IMCP"]), "FBCF_IMCP", "IMCP")
    panel["Exportaciones"] = apply_cutoff(yoy(series["Balanza"]["Exportaciones"]), "Balanza", "Exportaciones")
    panel["Importaciones"] = apply_cutoff(yoy(series["Balanza"]["Importaciones"]), "Balanza", "Importaciones")
    panel["IMSS"] = apply_cutoff(yoy(series["IMSS"]["IMSS_empleos"]), "IMSS", "IMSS_empleos")
    panel["ANTAD"] = apply_cutoff(yoy(series["Consumo"]["ANTAD"]), "ANTAD", "ANTAD")
    panel["AUTOS"] = apply_cutoff(yoy(series["Consumo"]["AUTOS"]), "AUTOS", "AUTOS")

    indpro = _SERIES_CACHE["indpro"].dropna().resample("MS").mean()
    indpro_ok = pd.Series(idx, index=idx).apply(lambda d: (d + pd.Timedelta(days=45)) <= cutoff_date)
    panel["INDPRO_EEUU"] = yoy(indpro).reindex(idx).where(indpro_ok)

    within = lambda days: pd.Series(idx, index=idx).apply(lambda d: (d + pd.Timedelta(days=days)) <= cutoff_date)
    panel["TIIE"] = series["TIIE"]["TIIE"].resample("MS").mean().reindex(idx).where(within(2))
    tc_m = series["TC"]["TC"].resample("MS").mean()
    panel["TC_dep"] = yoy(tc_m).reindex(idx).where(within(2))
    desem = series["desempleo"]["TasaDesempleo"].diff()
    panel["TasaDesempleo_chg"] = desem.reindex(idx).where(within(24))

    icsa_m = _SERIES_CACHE["icsa"].resample("MS").mean()
    panel["ICSA"] = yoy(icsa_m).reindex(idx).where(within(10))

    trends_m = _SERIES_CACHE["trends"].resample("MS").mean()
    panel["Trends_reapertura"] = trends_m["reapertura"].reindex(idx).where(within(3))
    panel["Trends_despidos"] = trends_m["despidos"].reindex(idx).where(within(3))

    bmv_m = _SERIES_CACHE["bmv"].resample("MS").mean()
    panel["BMV_yoy"] = yoy(bmv_m).reindex(idx).where(within(2))

    lt_m = _SERIES_CACHE["lt"].resample("MS").mean().reindex(idx)
    panel["Spread_tasas"] = lt_m.where(within(30)) - panel["TIIE"]

    igae_mom = 100 * np.log(series["otros"]["IGAE"] / series["otros"]["IGAE"].shift(1))
    panel["IGAE_target"] = apply_cutoff(igae_mom, "IGAE_ActInd", "IGAE")

    return panel


def fit_clean_params_and_bridge():
    """Se ajusta UNA SOLA VEZ (parametros DFM + puente ML), con datos
    limpios hasta TRAIN_CUTOFF -- reutilizado para los 36 meses."""
    panel_full = build_monthly_panel(cutoff_date=pd.Timestamp("2022-02-01"), panel_end=TRAIN_CUTOFF)
    panel_clean = panel_full.loc[:TRAIN_CUTOFF, ALL_COLS]
    print(f"Ajustando parametros limpios del DFM (hasta {TRAIN_CUTOFF.date()}, n={len(panel_clean)})...")
    fit = m3mod.em_dfm_general(panel_clean, ALL_COLS, verbose=False)
    phi, lam, R = fit["phi"], fit["lam"], fit["R"]
    print(f"phi={phi:.3f}  cargas principales: " +
          ", ".join(f"{k}={v:+.2f}" for k, v in sorted(lam.items(), key=lambda kv: -abs(kv[1]))[:5]))

    factor_clean = pd.Series(fit["x_smooth"][:, 0], index=panel_clean.index)
    bmv_lag1 = panel_full["BMV_yoy"].shift(1).reindex(panel_clean.index)
    df = pd.concat([panel_clean["IGAE_target"].rename("igae"), factor_clean.rename("f"),
                     bmv_lag1.rename("bmv_l1")], axis=1).dropna()
    y = df["igae"].values
    X_f, X_b, X_fb = df[["f"]].values, df[["bmv_l1"]].values, df[["f", "bmv_l1"]].values

    def cv_r2(X, model_fn, seed=0):
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        preds = np.zeros(len(y))
        for tr, te in kf.split(X):
            preds[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2), preds

    candidates = {
        ("factor", "OLS"): (X_f, lambda: LinearRegression()),
        ("BMV", "OLS"): (X_b, lambda: LinearRegression()),
        ("factor+BMV", "OLS"): (X_fb, lambda: LinearRegression()),
        ("factor+BMV", "ElasticNet"): (X_fb, lambda: ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=5000)),
        ("factor+BMV", "RandomForest"): (X_fb, lambda: RandomForestRegressor(n_estimators=150, max_depth=3, random_state=0)),
        ("factor+BMV", "GradientBoosting"): (X_fb, lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
    }
    results, preds_cache = {}, {}
    for key, (X, model_fn) in candidates.items():
        r2, preds = cv_r2(X, model_fn)
        results[key], preds_cache[key] = r2, preds
        print(f"  Puente [{key[0]:12s} / {key[1]:16s}]  n={len(df)}  CV10 R2={r2:+.3f}")
    best_key = max(results, key=results.get)
    Xb, model_fn = candidates[best_key]
    model = model_fn().fit(Xb, y)
    sigma = (y - preds_cache[best_key]).std(ddof=1)
    print(f"  -> Puente elegido: {best_key[0]} / {best_key[1]}  (CV10 R2={results[best_key]:.3f}, sigma={sigma:.2f}pp)\n")

    return dict(phi=phi, lam=lam, R=R, bridge_model=model, bridge_features=best_key[0], sigma=sigma)


def run_month(target_month, params):
    cutoff = target_month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15)
    panel = build_monthly_panel(cutoff, panel_end=target_month)
    phi, lam, R = params["phi"], params["lam"], params["R"]
    cols_available = [[c for c in NIVEL1_COLS if not np.isnan(panel.iloc[t][c])] for t in range(len(panel))]

    Q_flat = np.ones(len(panel))
    pf_proxy = m2mod.particle_filter(panel, cols_available, lam, R, phi, m2mod.g_tanh, 1e6, Q_flat,
                                      n_particles=300, seed=1)
    fm = pf_proxy["filtered_mean"]
    window = list(range(max(1, len(panel) - LAG_WINDOW_LP), len(panel)))
    resids = {i: fm[i] - phi * fm[i - 1] for i in window if not np.isnan(fm[i])}
    if resids:
        i_anchor = max(resids, key=lambda i: abs(resids[i]))
        s0 = max(abs(resids[i_anchor]) / 1.0, 1.0) ** 2
    else:
        i_anchor, s0 = len(panel) - 1, 1.0

    Q_series = np.ones(len(panel))
    for j, t in enumerate(range(i_anchor, len(panel))):
        Q_series[t] = 1 + (s0 - 1) * RHO_MONTHLY ** j

    pre_2020 = panel.loc[:"2019-12-01", NIVEL1_COLS]
    c = max(1.5 * pre_2020.std().mean() / np.mean(list(lam.values())), 1.0)
    res = m2mod.particle_filter(panel, cols_available, lam, R, phi, m2mod.g_asinh, c, Q_series, seed=31)
    final_particles, final_weights = res["final_particles"], res["final_weights"]

    bmv_input = panel["BMV_yoy"].dropna().iloc[-1] if panel["BMV_yoy"].notna().any() else 0.0
    rng = np.random.default_rng(1000 + target_month.year * 100 + target_month.month)
    idx_particles = rng.choice(len(final_particles), size=N_DRAWS, p=final_weights)
    growth = np.zeros(N_DRAWS)
    model, feats, sigma = params["bridge_model"], params["bridge_features"], params["sigma"]
    for d in range(N_DRAWS):
        x = final_particles[idx_particles[d]]
        x_in = np.array([[x]]) if feats == "factor" else (np.array([[bmv_input]]) if feats == "BMV" else np.array([[x, bmv_input]]))
        pred = model.predict(x_in)[0]
        growth[d] = pred + rng.normal(0, sigma)

    return dict(target_month=target_month, cutoff=cutoff, mediana=np.median(growth),
                p025=np.percentile(growth, 2.5), p975=np.percentile(growth, 97.5), s0=s0)


def main():
    params = fit_clean_params_and_bridge()

    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    igae_real = 100 * np.log(series["otros"]["IGAE"] / series["otros"]["IGAE"].shift(1))

    months = pd.date_range("2019-01-01", "2021-12-01", freq="MS")
    rows = []
    for i, m in enumerate(months):
        r = run_month(m, params)
        r["real"] = igae_real.get(m, np.nan)
        rows.append(r)
        print(f"[{i+1}/{len(months)}] {m.date()} (corte {r['cutoff'].date()}): "
              f"pronostico={r['mediana']:+.2f}% [{r['p025']:+.2f},{r['p975']:+.2f}]  real={r['real']:+.2f}%  s0={r['s0']:.1f}")

    df = pd.DataFrame(rows).set_index("target_month")
    df.to_csv(OUTPUT / "backtest_15dias_2019_2021.csv")
    print(f"\nGuardado: {OUTPUT / 'backtest_15dias_2019_2021.csv'}")

    cobertura = ((df["real"] >= df["p025"]) & (df["real"] <= df["p975"])).mean()
    mae = (df["mediana"] - df["real"]).abs().mean()
    print(f"Cobertura IC95% (2019-2021): {cobertura:.1%}   MAE mediana vs. real: {mae:.2f}pp")


if __name__ == "__main__":
    main()
