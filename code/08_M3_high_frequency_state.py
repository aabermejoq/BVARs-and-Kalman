"""
08_M3_high_frequency_state.py

FASE 9: M3 resuelve P3 (mucha informacion de alta frecuencia, no toda
comparable directamente con el PIB).

Panel (9 series mensuales, estandarizadas z-score con media/sd pre-2020):
  "Lentas" (rezago ~8-10 semanas, ultimo dato real-time = feb-2020):
    IGAE, ActividadIndustrial, FBCF, IMCP
  "Intermedias" (rezago ~25 dias, ultimo dato real-time = mar-2020):
    Exportaciones, Importaciones, INDPRO_EEUU (ver correccion abajo)
  "Rapidas" (rezago ~1-2 semanas, ultimo dato real-time = abr-2020):
    IMSS, ANTAD, AUTOS

CORRECCION DE UN SEGUNDO BUG DE VINTAGE (transparencia): el archivo
descargado de FRED en la Fase 1b trae una observacion fechada 2020-05-01
para INDPRO, que NUNCA pudo conocerse el 15-may-2020 (el mes de mayo no
habia terminado). Ademas, la observacion de abril-2020 se publica ~15 de
mayo (el mismo dia del corte): demasiado en el limite para asumir
disponibilidad con confianza. Se trunca conservadoramente INDPRO_EEUU a
marzo-2020 como ultima referencia segura.

Metodo (comparacion explicita, no ML "porque si"):
  A. FACTOR DINAMICO: primer componente principal (PCA) del panel
     BALANCEADO (donde estan las 9 series simultaneamente, feb-1993 a
     feb-2020), aplicado a los meses de "ragged edge" (mar,abr-2020) via
     proyeccion por minimos cuadrados usando solo las series disponibles
     ese mes. Se extrapola el factor a mayo-junio con un AR(1) (mismo
     enfoque que M2) y se traduce a PIB via ecuacion puente MCO.
  B. ML: ElasticNetCV sobre el panel trimestralizado (regularizacion L1+L2,
     alpha por validacion cruzada), entrenado con todas las variables
     simultaneamente (sin reducir a un factor).

Se comparan A vs B con una validacion pseudo-real-time (origen movil,
2010T1-2019T4, un paso adelante) y se USA el metodo con menor RMSE fuera de
muestra para el pronostico final de 2T20. La regla explicita del ejercicio
es que el ML no reemplaza automaticamente al factor tradicional: se reporta
el ganador y la magnitud de la diferencia.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNetCV

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "output" / "models"

N_DRAWS = 10_000
RNG_SEED = 42345
STD_CUTOFF = pd.Timestamp("2019-12-01")

SLOW_END = pd.Timestamp("2020-02-01")
MED_END = pd.Timestamp("2020-03-01")
FAST_END = pd.Timestamp("2020-04-01")


def build_panel():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    idx = pd.date_range("1993-01-01", "2020-04-01", freq="MS")
    panel = pd.DataFrame(index=idx)

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    panel["IGAE"] = yoy(series["otros"]["IGAE"]).reindex(idx)
    panel["ActividadIndustrial"] = yoy(series["otros"]["ActividadIndustrial"]).reindex(idx)
    panel["FBCF"] = yoy(series["otros"]["FBCF"]).reindex(idx)
    panel["IMCP"] = yoy(series["Consumo"]["IMCP"]).reindex(idx)
    panel["Exportaciones"] = yoy(series["Balanza"]["Exportaciones"]).reindex(idx)
    panel["Importaciones"] = yoy(series["Balanza"]["Importaciones"]).reindex(idx)
    panel["IMSS"] = yoy(series["IMSS"]["IMSS_empleos"]).reindex(idx)
    panel["ANTAD"] = yoy(series["Consumo"]["ANTAD"]).reindex(idx)
    panel["AUTOS"] = yoy(series["Consumo"]["AUTOS"]).reindex(idx)

    fred = pd.read_csv(EXTERNAL / "fred_us_benchmarks.csv", parse_dates=["date"]).set_index("date")
    indpro = fred["INDPRO"].dropna()
    indpro = indpro[indpro.index <= MED_END]  # correccion de vintage: truncar a mar-2020
    panel["INDPRO_EEUU"] = yoy(indpro).reindex(idx)

    # Enmascarar explicitamente por rezago de publicacion (real-time por diseno).
    panel.loc[panel.index > SLOW_END, ["IGAE", "ActividadIndustrial", "FBCF", "IMCP"]] = np.nan
    panel.loc[panel.index > MED_END, ["Exportaciones", "Importaciones", "INDPRO_EEUU"]] = np.nan
    panel.loc[panel.index > FAST_END, ["IMSS", "ANTAD", "AUTOS"]] = np.nan

    return panel


def standardize(panel):
    pre = panel.loc[:STD_CUTOFF]
    mu, sd = pre.mean(), pre.std()
    z = (panel - mu) / sd
    return z, mu, sd


def pca_loadings(z_balanced):
    X = z_balanced.dropna().values
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    loading = Vt[0]
    if loading.mean() < 0:  # normalizar signo: factor positivo = actividad alta
        loading = -loading
    return loading / np.linalg.norm(loading)


def project_factor(z_row, loading, cols):
    avail = z_row.notna()
    if avail.sum() == 0:
        return np.nan
    l_avail = loading[avail.values]
    x_avail = z_row[avail].values
    return float((x_avail @ l_avail) / (l_avail @ l_avail))


def kalman_uni_negloglik(params, y):
    phi, logQ, logR = params
    Q, R = np.exp(logQ), np.exp(logR)
    x, P = 0.0, 100.0
    ll = 0.0
    for yt in y:
        x_pred = phi * x
        P_pred = phi * phi * P + Q
        innov = yt - x_pred
        S = P_pred + R
        K = P_pred / S
        x = x_pred + K * innov
        P = (1 - K) * P_pred
        ll += -0.5 * (np.log(2 * np.pi) + np.log(S) + innov * innov / S)
    return -ll


def kalman_uni_filter(params, y):
    phi, logQ, logR = params
    Q, R = np.exp(logQ), np.exp(logR)
    x, P = 0.0, 100.0
    xs, Ps = [], []
    for yt in y:
        x_pred = phi * x
        P_pred = phi * phi * P + Q
        innov = yt - x_pred
        S = P_pred + R
        K = P_pred / S
        x = x_pred + K * innov
        P = (1 - K) * P_pred
        xs.append(x); Ps.append(P)
    return np.array(xs), np.array(Ps)


def factor_pipeline(panel_upto, pib_qoq_upto, cols):
    """Construye el factor PCA + puente + Kalman, usando solo datos con
    fecha de referencia <= la ultima fecha de panel_upto (para poder
    reutilizar esta funcion en el backtest pseudo-real-time)."""
    z, mu, sd = standardize(panel_upto[cols])
    balanced = z.dropna()
    loading = pca_loadings(balanced)

    factor = z.apply(lambda row: project_factor(row, loading, cols), axis=1)
    factor = factor.dropna()

    factor_q = factor.resample("QS").mean()
    factor_q_n = factor.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]  # tolerante: al menos 2 de 3 meses (panel con ragged edge)

    bridge_df = pd.concat([pib_qoq_upto.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    resid_b = yb - Xb @ coef_b
    sigma_b = resid_b.std(ddof=2) if len(yb) > 2 else resid_b.std()

    y = factor.values
    x0 = np.array([0.5, np.log(1.0), np.log(1.0)])
    bounds = [(-0.99, 0.99), (np.log(0.001), np.log(50)), (np.log(0.001), np.log(50))]
    res = minimize(kalman_uni_negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds)
    xs, Ps = kalman_uni_filter(res.x, y)

    return dict(loading=loading, mu=mu, sd=sd, factor=factor, coef_b=coef_b, sigma_b=sigma_b,
                kalman_params=res.x, x_last=xs[-1], P_last=Ps[-1], last_date=factor.index[-1])


def forecast_from_factor(fp, h_months, n_draws, seed):
    phi, logQ, logR = fp["kalman_params"]
    Q = np.exp(logQ)
    rng = np.random.default_rng(seed)
    coef_b, sigma_b = fp["coef_b"], fp["sigma_b"]
    x_last, P_last = fp["x_last"], fp["P_last"]

    growth_draws = np.zeros(n_draws)
    for d in range(n_draws):
        x0 = rng.normal(x_last, np.sqrt(max(P_last, 0)))
        xs = [x0]
        for _ in range(h_months):
            xs.append(phi * xs[-1] + np.sqrt(Q) * rng.standard_normal())
        f_q = np.mean(xs)
        growth_draws[d] = coef_b[0] + coef_b[1] * f_q + rng.normal(0, sigma_b)
    return growth_draws


def main():
    panel = build_panel()
    cols = list(panel.columns)
    print(f"Panel M3: {len(cols)} series, {panel.index.min().date()}->{panel.index.max().date()}")
    print("Ultima fecha no-nula por serie:")
    print(panel.apply(lambda c: c.last_valid_index()).to_string())

    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)

    fp = factor_pipeline(panel, pib_qoq, cols)
    print(f"\nFactor PCA: ultimo mes con proyeccion = {fp['last_date'].date()}, "
          f"x_filtrado={fp['x_last']:.2f}")
    print(f"Puente: PIB_q(%) = {fp['coef_b'][0]:.3f} + {fp['coef_b'][1]:.3f}*factor "
          f"(sigma_resid={fp['sigma_b']:.3f})")
    print(f"Kalman factor: phi={fp['kalman_params'][0]:.3f}, "
          f"Q={np.exp(fp['kalman_params'][1]):.3f}, R={np.exp(fp['kalman_params'][2]):.3f}")

    h_months = 3 - (pd.Timestamp("2020-06-01").month - fp["last_date"].month) % 3
    h_months = pd.Timestamp("2020-06-01").to_period("M").ordinal - fp["last_date"].to_period("M").ordinal
    print(f"Meses a pronosticar desde el factor hasta jun-2020: {h_months}")

    growth_pca = forecast_from_factor(fp, h_months, N_DRAWS, RNG_SEED)

    # --- ElasticNet (comparacion ML) ---
    panel_q = panel.resample("QS").mean()
    panel_q_n = panel.resample("QS").count()
    Xq = panel_q.where(panel_q_n >= 2)
    reg_df = pd.concat([pib_qoq.rename("pib"), Xq], axis=1, sort=True).dropna()
    Xen = reg_df[cols].values
    yen = reg_df["pib"].values

    en = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, 1.0], cv=5, random_state=0, max_iter=20000)
    en.fit(Xen, yen)
    print(f"\nElasticNetCV: alpha={en.alpha_:.4f}, l1_ratio={en.l1_ratio_:.2f}")
    print("Coeficientes no nulos:", {c: round(v, 3) for c, v in zip(cols, en.coef_) if abs(v) > 1e-6})

    # --- Backtest pseudo-real-time (origen movil, 2010T1-2019T4, un paso adelante) ---
    test_quarters = pd.date_range("2010-01-01", "2019-10-01", freq="QS")
    err_pca, err_en = [], []
    for tq in test_quarters:
        panel_train = panel.loc[:tq - pd.DateOffset(months=1)]
        pib_train = pib_qoq.loc[:tq]
        try:
            fp_t = factor_pipeline(panel_train, pib_train.loc[:tq - pd.offsets.QuarterBegin(1)], cols)
            gp = forecast_from_factor(fp_t, 3, 500, 0)
            pred_pca = np.median(gp)
        except Exception:
            pred_pca = np.nan

        reg_train = pd.concat([pib_qoq.rename("pib"), Xq], axis=1, sort=True).dropna()
        reg_train = reg_train.loc[:tq - pd.DateOffset(months=1)]
        if len(reg_train) > 15:
            en_t = ElasticNetCV(l1_ratio=[.5], cv=3, random_state=0, max_iter=5000)
            en_t.fit(reg_train[cols].values, reg_train["pib"].values)
            x_pred = Xq.loc[[tq]].ffill(axis=0).values
            pred_en = float(en_t.predict(x_pred)[0]) if not np.isnan(x_pred).any() else np.nan
        else:
            pred_en = np.nan

        actual = pib_qoq.get(tq, np.nan)
        if not np.isnan(actual):
            if not np.isnan(pred_pca):
                err_pca.append((pred_pca - actual) ** 2)
            if not np.isnan(pred_en):
                err_en.append((pred_en - actual) ** 2)

    rmse_pca = np.sqrt(np.nanmean(err_pca)) if err_pca else np.nan
    rmse_en = np.sqrt(np.nanmean(err_en)) if err_en else np.nan
    print(f"\nBacktest pseudo-real-time (2010T1-2019T4): RMSE factor-PCA={rmse_pca:.3f}, "
          f"RMSE ElasticNet={rmse_en:.3f}  ({len(err_pca)}/{len(err_en)} obs)")

    winner = "PCA" if (not np.isnan(rmse_pca) and (np.isnan(rmse_en) or rmse_pca <= rmse_en)) else "ElasticNet"
    print(f"Metodo elegido para el pronostico final de M3: {winner}")

    if winner == "PCA":
        growth_draws = growth_pca
    else:
        x_pred_full = Xq.loc[[pd.Timestamp("2020-01-01")]].ffill(axis=0).values
        point_pred = float(en.predict(x_pred_full)[0])
        resid_en_full = yen - en.predict(Xen)
        sigma_en = resid_en_full.std(ddof=1)
        rng = np.random.default_rng(RNG_SEED)
        growth_draws = point_pred + sigma_en * rng.standard_normal(N_DRAWS)

    summary = dict(
        modelo="M3", metodo_elegido=winner, rmse_pca_backtest=float(rmse_pca), rmse_elasticnet_backtest=float(rmse_en),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)), p10=float(np.percentile(growth_draws, 10)),
        p25=float(np.percentile(growth_draws, 25)), p75=float(np.percentile(growth_draws, 75)),
        p90=float(np.percentile(growth_draws, 90)), p97_5=float(np.percentile(growth_draws, 97.5)),
    )
    print("\n=== M3: factor amplio + ML, pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M3_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M3_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, fp=fp, en_model=en, cols=cols), f)
    print(f"\nGuardado: {OUTPUT / 'M3_growth_draws.npy'} y {OUTPUT / 'M3_fit.pkl'}")


if __name__ == "__main__":
    main()
