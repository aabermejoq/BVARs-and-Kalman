"""
23_modelo_unico_agosto.py

Modelo UNICO (agosto/3T20), construido desde cero sobre el panel ampliado
(17 series Nivel 1 + mobility/stringency Nivel 2, ver 22_agosto_panel_
extendido.py):

  Base:          BVAR Minnesota (fase temprana) + Factor comun dinamico
                 (DFM), medicion NO LINEAL g(S_t)=c*asinh(S_t/c) (igual
                 que M2, gana sobre tanh por verosimilitud de backcast).
  Volatilidad:   Lenza-Primiceri -- s0 medido del residuo REAL de 2T20
                 (no asumido), decae geometricamente rho=0.8^3 por
                 trimestre.
  Filtro:        Filtro de particulas bootstrap (Gordon 1993), 2 pasadas:
                   Pasada 1: Nivel 1 (17 series + PIB_q) -> factor_hist_1.
                   Calibracion: mobility_retail y stringency (CERO
                   traslape con la ventana limpia) se regresan contra
                   factor_hist_1 SOLO en su ventana 2020 disponible, para
                   obtener una carga/varianza -- calibracion "dentro del
                   episodio", declarada como tal.
                   Pasada 2: Nivel 1 + Nivel 2 (con cargas calibradas) ->
                   factor_hist_final (usado para el puente y el pronostico).
  Puente:        factor trimestral + BMV_yoy (rezagada 1 trimestre; ya
                 CONOCIDA para el trimestre objetivo, no hay que asumir
                 nada) -> PIB, elegido empiricamente entre varios modelos
                 de ML (OLS, ElasticNet, Random Forest, Gradient Boosting)
                 via R^2 fuera de muestra (LOO) en la ventana limpia
                 2000/2001-2019 -- mismo estandar de rigor que el resto del
                 ejercicio (no se asume que ML gane, se prueba).
  Proyeccion:    El factor se propaga con su propia dinamica AR(1) + ruido
                 escalado por LP hasta completar 3T20 (unica pieza
                 genuinamente incierta); el insumo de BMV para el puente
                 NO se extrapola -- BMV(2T20), el rezago relevante, ya es
                 un dato real conocido al 15-ago-2020.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models_agosto"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m2mod = import_module("19_M2_particulas_lp")

N_DRAWS = 10_000
RHO_QUARTERLY = 0.8 ** 3
TARGET_Q = pd.Timestamp("2020-07-01")


def build_measurement(cols, lam, R, extra_lam=None, extra_R=None):
    lam2, R2 = dict(lam), dict(R)
    if extra_lam:
        lam2.update(extra_lam)
        R2.update(extra_R)
    return lam2, R2


def run_particle_pass(panel, cols, lam, R, phi, c, Q_series, seed):
    cols_available = [[col for col in cols + ["PIB_q"] if col in panel.columns and not np.isnan(panel.iloc[t][col])]
                       for t in range(len(panel))]
    res = m2mod.particle_filter(panel, cols_available, lam, R, phi, m2mod.g_asinh, c, Q_series, seed=seed)
    return res


def calibrate_nivel2(factor_hist, nivel2):
    """Regresion auxiliar (SOLO ventana 2020 disponible) de cada serie de
    Nivel 2 contra el factor ya extraido de Nivel 1 -- no hay traslape con
    la ventana limpia, asi que esta es una calibracion dentro del episodio,
    no una validacion fuera de muestra (se declara explicitamente)."""
    lam2, R2 = {}, {}
    for col in nivel2.columns:
        df = pd.concat([factor_hist.rename("f"), nivel2[col].rename("y")], axis=1).dropna()
        if len(df) < 3:
            continue
        X = np.column_stack([np.ones(len(df)), df["f"].values])
        coef, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
        resid = df["y"].values - X @ coef
        # lam[col] representa la pendiente (el intercepto se absorbe aproximando
        # con la desviacion de la serie respecto a su propia media en la ventana)
        y_centered = df["y"] - df["y"].mean()
        f_centered = df["f"] - df["f"].mean()
        slope = np.sum(f_centered * y_centered) / np.sum(f_centered ** 2)
        resid2 = y_centered.values - slope * f_centered.values
        lam2[col] = slope
        R2[col] = max(np.var(resid2), 1e-3)
        print(f"  Calibracion Nivel 2 [{col}]: n={len(df)}  carga={slope:.3f}  R={R2[col]:.2f}")
    return lam2, R2


def select_ml_bridge(factor_q, bmv_q_lag1, pib_qoq, clean_cutoff):
    df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f"), bmv_q_lag1.rename("bmv_l1")],
                    axis=1, sort=True).dropna()
    df_pre = df.loc[:clean_cutoff]
    y = df_pre["pib"].values
    X_f = df_pre[["f"]].values
    X_fb = df_pre[["f", "bmv_l1"]].values

    def loo_r2(X, model_fn):
        loo = LeaveOneOut()
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            m = model_fn().fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)

    X_b = df_pre[["bmv_l1"]].values
    candidates = {
        ("factor", "OLS"): (X_f, lambda: LinearRegression()),
        ("BMV", "OLS"): (X_b, lambda: LinearRegression()),
        ("factor+BMV", "OLS"): (X_fb, lambda: LinearRegression()),
        ("factor+BMV", "ElasticNet"): (X_fb, lambda: ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=5000)),
        ("factor+BMV", "RandomForest"): (X_fb, lambda: RandomForestRegressor(n_estimators=300, max_depth=3, random_state=0)),
        ("factor+BMV", "GradientBoosting"): (X_fb, lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
        ("BMV", "GradientBoosting"): (X_b, lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
    }
    results = {}
    for key, (X, model_fn) in candidates.items():
        r2 = loo_r2(X, model_fn)
        results[key] = r2
        print(f"  Puente [{key[0]:12s} / {key[1]:16s}]  n={len(df_pre)}  LOO R2={r2:+.3f}")

    best_key = max(results, key=results.get)
    Xb, model_fn = candidates[best_key]
    model = model_fn().fit(Xb, y)
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Xb):
        preds[te] = model_fn().fit(Xb[tr], y[tr]).predict(Xb[te])
    sigma = (y - preds).std(ddof=1)
    print(f"  -> Mejor puente: {best_key[0]} / {best_key[1]}  (LOO R2={results[best_key]:.3f}, sigma={sigma:.2f}pp)")

    # --- Anclar la incertidumbre del puente al error REAL en 2T20 (ya
    # conocido al corte), mismo principio LP usado en 06c/07c/19/20: un
    # residuo de tiempos normales (sigma de arriba) subestima brutalmente
    # el riesgo de que el puente simplemente no se sostenga en un choque
    # sin precedente. 3T20 es el trimestre INMEDIATO siguiente al ancla
    # (2T20) -> se usa el multiplicador completo, sin decaimiento todavia. ---
    feat_map = {"factor": ["f"], "BMV": ["bmv_l1"], "factor+BMV": ["f", "bmv_l1"]}
    x_2t20 = df.loc[["2020-04-01"], feat_map[best_key[0]]].values
    pred_2t20 = model.predict(x_2t20)[0]
    real_2t20 = pib_qoq.loc["2020-04-01"]
    resid_2t20 = real_2t20 - pred_2t20
    s0 = max(abs(resid_2t20) / sigma, 1.0) ** 2
    sigma_eff = sigma * np.sqrt(s0)
    print(f"  Residuo REAL del puente en 2T20: predicho={pred_2t20:.2f}%  real={real_2t20:.2f}%  "
          f"error={resid_2t20:.2f}pp  ->  s0={s0:.1f}x  ->  sigma efectivo={sigma_eff:.2f}pp")

    return dict(model=model, features=best_key[0], method=best_key[1], sigma=sigma, sigma_eff=sigma_eff,
                r2_loo=results[best_key])


def main():
    with open(INTERIM / "panel_extendido_agosto.pkl", "rb") as f:
        saved = pickle.load(f)
    panel, cols, pib_qoq, params, nivel2 = (saved["panel"], saved["cols"], saved["pib_qoq"],
                                             saved["params"], saved["nivel2"])
    phi, lam, R = params["phi"], params["lam"], params["R"]

    # --- s0 (LP), medido del residuo REAL de 2T20 (proxy casi-lineal, c grande) ---
    cols_available_flat = [[c for c in cols + ["PIB_q"] if not np.isnan(panel.iloc[t][c])] for t in range(len(panel))]
    Q_flat = np.ones(len(panel))
    pf_proxy = m2mod.particle_filter(panel, cols_available_flat, lam, R, phi, m2mod.g_tanh, 1e6, Q_flat,
                                      n_particles=500, seed=1)
    idx_2t20 = panel.index.get_loc(pd.Timestamp("2020-04-01"))
    resid_2t20 = pf_proxy["filtered_mean"][idx_2t20] - phi * pf_proxy["filtered_mean"][idx_2t20 - 1]
    s0 = max(abs(resid_2t20) / 1.0, 1.0) ** 2
    print(f"s0 (LP, residuo real de 2T20) = {s0:.2f}")

    Q_series = np.ones(len(panel))
    for j, t in enumerate(range(idx_2t20, len(panel))):
        Q_series[t] = 1 + (s0 - 1) * RHO_QUARTERLY ** j

    pre_2020 = panel.loc[:"2019-12-01", cols]
    c = max(1.5 * pre_2020.std().mean() / np.mean(list(lam.values())), 1.0)

    print("\nPasada 1 (Nivel 1 + PIB_q)...")
    pass1 = run_particle_pass(panel, cols, lam, R, phi, c, Q_series, seed=11)
    factor_hist_1 = pd.Series(pass1["filtered_mean"], index=panel.index)

    print("\nCalibrando Nivel 2 (mobility_retail, stringency) contra el factor de la pasada 1...")
    lam2, R2 = calibrate_nivel2(factor_hist_1, nivel2)

    panel_full = panel.copy()
    for col in nivel2.columns:
        panel_full[col] = nivel2[col]
    cols_full = cols + list(nivel2.columns)
    lam_full, R_full = build_measurement(cols_full, lam, R, lam2, R2)

    print("\nPasada 2 (Nivel 1 + Nivel 2 calibrado + PIB_q)...")
    pass2 = run_particle_pass(panel_full, cols_full, lam_full, R_full, phi, c, Q_series, seed=12)
    factor_hist = pd.Series(pass2["filtered_mean"], index=panel.index)
    final_particles, final_weights = pass2["final_particles"], pass2["final_weights"]

    # --- Puente ML: factor trimestral + BMV(rezago 1T, YA CONOCIDO para 3T20) -> PIB ---
    factor_q = factor_hist.resample("QS").mean()
    factor_q = factor_q[factor_hist.resample("QS").count() >= 2]
    bmv_yoy_m = panel_full["BMV_yoy"]
    bmv_q = bmv_yoy_m.resample("QS").mean()
    bmv_q_lag1 = bmv_q.shift(1)

    print("\nSeleccionando puente ML (factor [+ BMV rezagado] -> PIB, LOO en ventana limpia)...")
    bridge = select_ml_bridge(factor_q, bmv_q_lag1, pib_qoq, clean_cutoff="2019-10-01")

    # --- Pronostico: propagar el factor con su propia dinamica + LP hasta 3T20 ---
    h_months = TARGET_Q.to_period("M").ordinal - panel.index[-1].to_period("M").ordinal
    j_next = len(panel) - idx_2t20
    bmv_2t20 = bmv_q.loc["2020-04-01"]  # YA CONOCIDO al 15-ago-2020 -- es el insumo de BMV para 3T20 (rezago 1T)
    print(f"BMV(2T20, a/a) = {bmv_2t20:.1f}%  -- insumo REAL (no extrapolado) del puente para 3T20")

    rng = np.random.default_rng(4001)
    idx_particles = rng.choice(len(final_particles), size=N_DRAWS, p=final_weights)
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        x0 = final_particles[idx_particles[d]]
        xs = [x0]
        for j in range(h_months):
            Q_j = 1 + (s0 - 1) * RHO_QUARTERLY ** (j_next + j)
            xs.append(phi * xs[-1] + np.sqrt(Q_j) * rng.standard_normal())
        f_q = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        if bridge["features"] == "factor":
            x_in = np.array([[f_q]])
        elif bridge["features"] == "BMV":
            x_in = np.array([[bmv_2t20]])
        else:
            x_in = np.array([[f_q, bmv_2t20]])
        pred = bridge["model"].predict(x_in)[0]
        growth[d] = pred + rng.normal(0, bridge["sigma_eff"])

    print(f"\n=== Modelo unico (particulas-asinh + LP + puente ML): pronostico 3T20 ===")
    print(f"Mediana={np.median(growth):.2f}%  IC95%=[{np.percentile(growth,2.5):.2f}, {np.percentile(growth,97.5):.2f}]")

    np.save(OUTPUT / "modelo_unico_growth_draws.npy", growth)
    with open(OUTPUT / "modelo_unico_fit.pkl", "wb") as f:
        pickle.dump(dict(s0=s0, bridge_features=bridge["features"], bridge_method=bridge["method"],
                          bridge_r2_loo=bridge["r2_loo"], bmv_2t20=bmv_2t20, factor_hist=factor_hist), f)
    print(f"Guardado: {OUTPUT / 'modelo_unico_growth_draws.npy'}")


if __name__ == "__main__":
    main()
