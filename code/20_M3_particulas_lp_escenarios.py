"""
20_M3_particulas_lp_escenarios.py

M3 = M2 (Particulas + LP, panel amplio) + escenarios de curvas
epidemiologicas de la SS calibrados con tarjetas via Gradient Boosting
(no lineal), para la TRAYECTORIA FUTURA del factor -- resuelve P3 encima
del paquete de M2 que ya resuelve P1+P2.

El estado FILTRADO (hasta el ultimo mes observado) se toma directamente
de M2 (particulas + LP) -- ahi no cambia nada; lo que cambia es como se
extrapola hacia el trimestre objetivo.

--- BUG ENCONTRADO Y CORREGIDO (puente tarjetas -> PIB) ---
`build_card_bridge()` (en 14_M4_agosto_bayesian_scenarios.py) estima
tarjetas_a/a = a + b*PIB_qoq (tarjetas como VARIABLE DEPENDIENTE; esa es
la direccion correcta para su uso original en 14: traducir un escenario
de PIB en una trayectoria esperada de tarjetas, para comparar contra la
realizacion diaria). Tanto un primer intento de este script como
16_M4_epidemiologico.py reutilizaron esos MISMOS coeficientes al reves,
como si fueran PIB = a + b*tarjetas -- eso no es una inversion valida de
la recta de OLS. Con R^2 bajo (~0.11 en niveles), aplicar los coeficientes
al reves AMPLIFICA en vez de atenuar, y fue lo que produjo el resultado
"exitoso" de +11.93%/+14.3% ya reportado -- un artefacto, no una senal
real.

Se re-estimo el puente en la direccion correcta (PIB como dependiente) y
se probaron VARIAS formas funcionales -- lineal, lineal+tendencia (proxy
de "bancarizacion"/adopcion creciente de tarjetas, ya que no hay una serie
de penetracion bancaria en los datos disponibles), Gradient Boosting
con/sin tendencia -- sobre DOS series (el agregado nacional de tarjetas y
el "cluster" de categorias mas golpeado por el choque, identificado en
12_tarjetas_daily_audit.py), evaluadas por R^2 fuera de muestra
(leave-one-out, dado el tamano pequeno de la muestra trimestral limpia,
n=40, 2009T1-2019T4). Resultado: NINGUNA forma no lineal ni el cluster
mejoran el ajuste; GBM sobreajusta con solo 40 observaciones trimestrales
(R^2 LOO negativo) y el cluster mas golpeado no tiene relacion con el PIB
en tiempos NORMALES (fue definido por su caida en 2020, no por su
covarianza historica con el ciclo). La mejor especificacion (lineal +
tendencia, sobre el agregado) apenas alcanza R^2 LOO=0.093 -- la senal
real de tarjetas para el PIB trimestral, estimada honestamente, es casi
nula.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models_agosto"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
panelmod = import_module("17_agosto_panel_amplio")
m4epi = import_module("16_M4_epidemiologico")

N_DRAWS = 10_000
ASOF = pd.Timestamp("2020-08-15")
TARGET_Q = pd.Timestamp("2020-07-01")
CLEAN_CUTOFF_Q = "2019-10-01"


class _OLS:
    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), X])
        self.coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return Xb @ self.coef


def _gbr():
    return GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=0)


def _loo_r2(X, y, model_fn):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_fn().fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def select_card_bridge():
    """Prueba varias formas funcionales (lineal, lineal+tendencia, GBM
    con/sin tendencia) sobre 2 series (agregado, cluster mas golpeado) y
    elige la de mejor R^2 fuera de muestra (LOO), estimando SOLO con datos
    limpios pre-2020 (2009T1-2019T4)."""
    yoy = pd.read_csv(INTERIM / "tarjetas_daily_yoy.csv", index_col=0, parse_dates=True)
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib_qoq_full = 100 * np.log(series["PIB"]["PIB"] / series["PIB"]["PIB"].shift(1))

    with open(INTERIM / "tarjetas_clusters.pkl", "rb") as f:
        clusters = pickle.load(f)
    hardest_cats = clusters["labels"][clusters["labels"] == clusters["peor_cluster"]].index.tolist()

    agregado_q = yoy["AGREGADO"].replace([np.inf, -np.inf], np.nan).resample("QS").mean()
    hardest_q = yoy[hardest_cats].replace([np.inf, -np.inf], np.nan).mean(axis=1).resample("QS").mean()

    candidates = {}
    for series_name, tarj_q in [("agregado", agregado_q), ("cluster_golpeado", hardest_q)]:
        df = pd.concat([pib_qoq_full.rename("pib"), tarj_q.rename("tarj")], axis=1, sort=True).dropna()
        df_pre = df.loc[:CLEAN_CUTOFF_Q].copy()
        df_pre["t"] = np.arange(len(df_pre))
        y = df_pre["pib"].values
        tarj, t = df_pre["tarj"].values, df_pre["t"].values

        specs = {
            "OLS lineal": (tarj.reshape(-1, 1), _OLS),
            "OLS lineal+tendencia": (np.column_stack([tarj, t]), _OLS),
            "GBM+tendencia": (np.column_stack([tarj, t]), _gbr),
            "GBM sin tendencia": (tarj.reshape(-1, 1), _gbr),
        }
        for spec_name, (X, model_fn) in specs.items():
            r2 = _loo_r2(X, y, model_fn)
            candidates[(series_name, spec_name)] = dict(r2=r2, X=X, y=y, tarj=tarj, t=t,
                                                          model_fn=model_fn, df_pre=df_pre)
            print(f"  [{series_name:16s}] {spec_name:20s} LOO R2={r2:+.3f}")

    best_key = max(candidates, key=lambda k: candidates[k]["r2"])
    best = candidates[best_key]
    print(f"  -> Mejor puente tarjetas->PIB: serie={best_key[0]}, forma={best_key[1]} (LOO R2={best['r2']:.3f})")

    model = best["model_fn"]().fit(best["X"], best["y"])
    loo_preds = np.zeros(len(best["y"]))
    loo = LeaveOneOut()
    for tr, te in loo.split(best["X"]):
        m = best["model_fn"]().fit(best["X"][tr], best["y"][tr])
        loo_preds[te] = m.predict(best["X"][te])
    sigma_oos = (best["y"] - loo_preds).std(ddof=1)

    uses_trend = "tendencia" in best_key[1]
    series_used = best_key[0]
    t_next = len(best["t"])  # siguiente periodo trimestral tras la muestra limpia (2020T1)

    def predict_pib(tarjetas_yoy_value, quarters_ahead):
        """quarters_ahead=0 -> 2020T1, 1 -> 2020T2, 2 -> 2020T3 (3T20, el objetivo)."""
        x = [tarjetas_yoy_value]
        if uses_trend:
            x.append(t_next + quarters_ahead)
        return float(model.predict(np.array(x).reshape(1, -1))[0])

    # --- Anclar la incertidumbre del puente al error REALMENTE observado en
    # 2T20 (ya conocido al 15-ago-2020), en vez de usar sigma_oos (residuo de
    # tiempos normales, 2009-2019) tal cual: mismo principio de anclaje LP
    # usado en 06c/07c/19 (s0 medido del residuo real del choque, no
    # asumido). Aqui el choque es tan extremo que el propio puente (aun con
    # la mejor forma funcional) lo pierde por completo. ---
    tarj_pre2020 = pd.concat([pib_qoq_full.rename("pib"),
                               (agregado_q if series_used == "agregado" else hardest_q).rename("tarj")],
                              axis=1, sort=True).dropna()
    tarj_1t20 = tarj_pre2020["tarj"].get(pd.Timestamp("2020-01-01"), np.nan)
    tarj_2t20 = tarj_pre2020["tarj"].get(pd.Timestamp("2020-04-01"), np.nan)
    pib_1t20_real, pib_2t20_real = tarj_pre2020["pib"].get(pd.Timestamp("2020-01-01"), np.nan), \
                                     tarj_pre2020["pib"].get(pd.Timestamp("2020-04-01"), np.nan)
    resid_2t20 = pib_2t20_real - predict_pib(tarj_2t20, quarters_ahead=1)
    s0_card = max(abs(resid_2t20) / sigma_oos, 1.0) ** 2
    print(f"  Residuo REAL del puente en 2T20 (ya conocido): predicho={predict_pib(tarj_2t20, quarters_ahead=1):.2f}%  "
          f"real={pib_2t20_real:.2f}%  error={resid_2t20:.2f}pp  ->  s0_card={s0_card:.1f}x")

    return dict(predict_pib=predict_pib, sigma=sigma_oos, series_used=series_used,
                spec_used=best_key[1], r2_loo=best["r2"], s0_card=s0_card)


def main():

    # --- Puente tarjetas->PIB, direccion correcta (PIB como dependiente) y
    # forma funcional elegida empiricamente (ver select_card_bridge). ---
    print("Seleccionando puente tarjetas->PIB (varias series y formas funcionales, LOO R2 sobre 2009T1-2019T4)...")
    bridge = select_card_bridge()
    predict_pib, sigma_card = bridge["predict_pib"], bridge["sigma"]
    tarjetas_yoy = m4epi.load_tarjetas_yoy()

    # --- Escenarios epidemiologicos + GBM (reutiliza la logica de 16, sin tocar datos futuros) ---
    cases = m4epi.load_case_data()
    cases_rt = cases.loc[:ASOF - pd.Timedelta(days=1)]
    tarjetas_rt = tarjetas_yoy.loc[:ASOF - pd.Timedelta(days=1)]
    log_casos_hist = np.log(cases_rt["case_7d"].clip(lower=1))
    feat_hist = m4epi.build_features(log_casos_hist)
    train_df = pd.concat([feat_hist, tarjetas_rt.rename("tarjetas_yoy")], axis=1, sort=True).dropna()
    Xtr, ytr = train_df[["log_casos", "crecimiento_7d", "t"]].values, train_df["tarjetas_yoy"].values
    gbr = GradientBoostingRegressor(**m4epi.GBR_PARAMS)
    gbr.fit(Xtr, ytr)
    sigma_gbr = (ytr - gbr.predict(Xtr)).std(ddof=5)
    print(f"GBR: n={len(train_df)}, R2={gbr.score(Xtr,ytr):.3f}, sigma={sigma_gbr:.2f}pp")

    last_level = log_casos_hist.iloc[-1]
    t0 = feat_hist["t"].iloc[-1]
    horizon = (TARGET_Q + pd.DateOffset(months=3) - ASOF).days + 5
    peak_A, decline_A = pd.Timestamp("2020-05-10"), 0.020
    peak_B, decline_B = pd.Timestamp("2020-06-10"), 0.008
    scen_A_log = m4epi.epi_curve(last_level, ASOF - pd.Timedelta(days=1), peak_A, decline_A, horizon, backcast_days=21)
    scen_B_log = m4epi.epi_curve(last_level, ASOF - pd.Timedelta(days=1), peak_B, decline_B, horizon, backcast_days=21)

    backcast_dates = scen_A_log.index[scen_A_log.index < ASOF]
    real_backcast = log_casos_hist.reindex(backcast_dates)
    resid_std = log_casos_hist.diff().dropna().std()
    valid = real_backcast.notna()
    ll_A = norm.logpdf(real_backcast[valid] - scen_A_log.reindex(backcast_dates)[valid], scale=resid_std).sum()
    ll_B = norm.logpdf(real_backcast[valid] - scen_B_log.reindex(backcast_dates)[valid], scale=resid_std).sum()
    m = max(ll_A, ll_B)
    w_A, w_B = np.exp(ll_A - m), np.exp(ll_B - m)
    p_A, p_B = w_A / (w_A + w_B), w_B / (w_A + w_B)
    print(f"Pesos de escenario (backcast, sin datos futuros): P(control)={p_A:.3f}  P(meseta)={p_B:.3f}")

    def implied_tarjetas(scen_log):
        fwd = scen_log[scen_log.index > ASOF - pd.Timedelta(days=1)]
        feats = pd.DataFrame({"log_casos": fwd, "crecimiento_7d": fwd.diff(7).bfill(),
                               "t": np.arange(t0 + 1, t0 + 1 + len(fwd))}, index=fwd.index)
        return pd.Series(gbr.predict(feats.values), index=fwd.index)

    implied_A, implied_B = implied_tarjetas(scen_A_log), implied_tarjetas(scen_B_log)
    tq_days = pd.date_range(TARGET_Q, TARGET_Q + pd.DateOffset(months=3) - pd.Timedelta(days=1), freq="D")
    tarjetas_q_A, tarjetas_q_B = implied_A.reindex(tq_days).mean(), implied_B.reindex(tq_days).mean()
    # PIB implicito por escenario, via el puente tarjetas->PIB elegido empiricamente
    # (2T20=quarters_ahead 1 ya paso; 3T20=objetivo => quarters_ahead=2 desde el fin de la muestra limpia)
    pib_q_A = predict_pib(tarjetas_q_A, quarters_ahead=2)
    pib_q_B = predict_pib(tarjetas_q_B, quarters_ahead=2)
    print(f"Tarjetas (a/a) implicita: control={tarjetas_q_A:.1f}%  meseta={tarjetas_q_B:.1f}%")
    print(f"PIB IMPLICITO por escenario (3T20, puente {bridge['series_used']}/{bridge['spec_used']}, "
          f"LOO R2={bridge['r2_loo']:.3f}): control={pib_q_A:.2f}%  meseta={pib_q_B:.2f}%")

    # --- Combinar M2 (extrapolacion propia AR(1)+LP del factor, ya con su
    # incertidumbre de choque extremo) con el escenario (via el puente
    # tarjetas->PIB), por PRECISION (inverso de varianza) -- la forma
    # correcta de combinar 2 estimadores de la misma cantidad con
    # incertidumbres distintas, en vez de un peso arbitrario (0.3) fijo.
    #
    # sigma_card (bridge["sigma"]) es el residuo LOO en tiempos NORMALES
    # (2009-2019); usarlo tal cual subestimaria brutalmente el riesgo real:
    # el MISMO puente, aplicado al dato REAL de tarjetas de 2T20 (ya
    # conocido al 15-ago-2020), erro por bridge["s0_card"] veces su varianza
    # normal (ver select_card_bridge) -- el choque es tan extremo que el
    # puente lo pierde por completo. 3T20 es el trimestre INMEDIATO
    # siguiente al ancla (2T20), por lo que se usa s0_card sin decaimiento
    # todavia (mismo criterio de "j=1, sin decaimiento" de 06c/07c/19). ---
    s0_card = bridge["s0_card"]
    sigma_card_eff = sigma_card * np.sqrt(s0_card)
    print(f"Factor de inflacion del puente tarjetas->PIB (anclado al error REAL de 2T20): "
          f"{s0_card:.1f}x  ->  sigma_card efectivo={sigma_card_eff:.2f}pp (vs {sigma_card:.2f}pp en tiempos normales)")

    m2_growth = np.load(OUTPUT / "M2_particulas_lp_growth_draws.npy")
    mu_m2, var_m2 = m2_growth.mean(), m2_growth.var()

    mu_scen = p_A * pib_q_A + p_B * pib_q_B
    var_scen_mixture = p_A * (pib_q_A - mu_scen) ** 2 + p_B * (pib_q_B - mu_scen) ** 2
    var_scen = var_scen_mixture + sigma_card_eff ** 2

    w_m2 = (1 / var_m2) / (1 / var_m2 + 1 / var_scen)
    w_scen = 1 - w_m2
    print(f"M2 (propia dinamica): media={mu_m2:.2f}%  sd={np.sqrt(var_m2):.2f}pp  peso={w_m2:.3f}")
    print(f"Escenario (tarjetas->PIB): media={mu_scen:.2f}%  sd={np.sqrt(var_scen):.2f}pp  peso={w_scen:.3f}")

    rng = np.random.default_rng(3001)
    scenario_choice = rng.choice([0, 1], size=N_DRAWS, p=[p_A, p_B])
    boot_sd_card = abs(tarjetas_q_A - tarjetas_q_B) * 0.2 + 1.0
    m2_draws_resampled = rng.choice(m2_growth, size=N_DRAWS, replace=True)

    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        tarjetas_d = tarjetas_q_A if scenario_choice[d] == 0 else tarjetas_q_B
        tarjetas_d = tarjetas_d + rng.normal(0, boot_sd_card)
        pib_scenario_d = predict_pib(tarjetas_d, quarters_ahead=2) + rng.normal(0, sigma_card_eff)
        growth[d] = w_m2 * m2_draws_resampled[d] + w_scen * pib_scenario_d

    print(f"\n=== M3 (Particulas+LP + escenarios SS/tarjetas): pronostico 3T20 ===")
    print(f"Mediana={np.median(growth):.2f}%  IC95%=[{np.percentile(growth,2.5):.2f}, {np.percentile(growth,97.5):.2f}]")

    np.save(OUTPUT / "M3_particulas_lp_escenarios_growth_draws.npy", growth)
    with open(OUTPUT / "M3_summary.pkl", "wb") as f:
        pickle.dump(dict(p_A=p_A, p_B=p_B, pib_q_A=pib_q_A, pib_q_B=pib_q_B,
                          tarjetas_q_A=tarjetas_q_A, tarjetas_q_B=tarjetas_q_B,
                          bridge_series=bridge["series_used"], bridge_spec=bridge["spec_used"],
                          bridge_r2_loo=bridge["r2_loo"], s0_card=s0_card, sigma_card_eff=sigma_card_eff,
                          w_m2=w_m2, w_scen=w_scen), f)
    print(f"Guardado: {OUTPUT / 'M3_particulas_lp_escenarios_growth_draws.npy'}")


if __name__ == "__main__":
    main()
