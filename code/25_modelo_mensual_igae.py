"""
25_modelo_mensual_igae.py

Modelo unico, a frecuencia MENSUAL, para el IGAE (no PIB trimestral),
corrido en los 3 cortes de informacion acordados sobre mayo-2020 (ver
24_panel_mensual_igae.py para el calendario de publicacion VERIFICADO):

  Corte 1: 15-jun-2020 (15 dias tras cerrar mayo)
  Corte 2: 30-jun-2020 (1 mes despues)
  Corte 3: 23-jul-2020 (justo ANTES de que salga el IGAE de mayo, 24-jul)

Diferencias clave vs. el modelo trimestral (23_modelo_unico_agosto.py):
  - IGAE_target entra al DFM como una serie MENSUAL MAS (H=[lam,0,0]), NO
    como "PIB_q" (esa forma especial hace un promedio 1/3+1/3+1/3 para
    puentear trimestre->mes, que ya no aplica). Practica estandar de
    nowcasting (el propio NowTIM de Banxico usa el IGAE como insumo #1).
  - LP: en vez de anclar a un trimestre especifico ya conocido (no
    aplicable aqui: estamos DENTRO del choque, sin un trimestre "ya
    resuelto" como ancla), se mide el mayor residuo de estado observado
    en los meses de 2020 YA DISPONIBLES en cada corte (marzo en el corte
    1; abril en los cortes 2 y 3) -- se adapta automaticamente a cuanta
    informacion hay en cada corte, sin asumir un numero externo.
  - Nivel 2 (mobility/stringency) se OMITE en este ejercicio: con el
    panel truncado a mayo-2020, el traslape disponible seria de apenas
    3-4 meses, insuficiente para una calibracion minimamente estable
    (a diferencia del ejercicio de agosto, con 6-7 meses de traslape).
  - Puente ML: factor + BMV(rezago 1 mes, YA CONOCIDO siempre que este
    dentro del panel) -> IGAE, mismos candidatos (OLS/ElasticNet/Random
    Forest/Gradient Boosting), LOO en la ventana limpia. NOTA de honestidad:
    dado que IGAE_target participa en la MISMA estimacion del factor
    (ver arriba), el LOO de este puente sobre datos historicos es
    optimista (el factor "ya sabe" del IGAE de ese mismo periodo); la
    prueba real de exactitud es la comparacion final contra el IGAE de
    mayo, nunca visto por ningun corte.
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
OUTPUT = ROOT / "output" / "models_mensual_igae"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
panelmod = import_module("24_panel_mensual_igae")
m2mod = import_module("19_M2_particulas_lp")

N_DRAWS = 10_000
RHO_MONTHLY = 0.8  # decaimiento LP mensual (no ^3: ya no se compone por trimestre)
CORTES = [("corte1_15jun", panelmod.CORTE_1), ("corte2_30jun", panelmod.CORTE_2),
          ("corte3_23jul", panelmod.CORTE_3)]


def select_ml_bridge(factor_m, bmv_lag1, igae_target, clean_cutoff):
    df = pd.concat([igae_target.rename("igae"), factor_m.rename("f"), bmv_lag1.rename("bmv_l1")],
                    axis=1, sort=True).dropna()
    df_pre = df.loc[:clean_cutoff]
    y = df_pre["igae"].values
    X_f, X_b, X_fb = df_pre[["f"]].values, df_pre[["bmv_l1"]].values, df_pre[["f", "bmv_l1"]].values

    def loo_r2(X, model_fn):
        loo = LeaveOneOut()
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            preds[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)

    candidates = {
        ("factor", "OLS"): (X_f, lambda: LinearRegression()),
        ("BMV", "OLS"): (X_b, lambda: LinearRegression()),
        ("factor+BMV", "OLS"): (X_fb, lambda: LinearRegression()),
        ("factor+BMV", "ElasticNet"): (X_fb, lambda: ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=5000)),
        ("factor+BMV", "RandomForest"): (X_fb, lambda: RandomForestRegressor(n_estimators=300, max_depth=3, random_state=0)),
        ("factor+BMV", "GradientBoosting"): (X_fb, lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
    }
    results = {}
    for key, (X, model_fn) in candidates.items():
        r2 = loo_r2(X, model_fn)
        results[key] = r2
        print(f"    Puente [{key[0]:12s} / {key[1]:16s}]  n={len(df_pre)}  LOO R2={r2:+.3f}")

    best_key = max(results, key=results.get)
    Xb, model_fn = candidates[best_key]
    model = model_fn().fit(Xb, y)
    preds = np.zeros(len(y))
    loo = LeaveOneOut()
    for tr, te in loo.split(Xb):
        preds[te] = model_fn().fit(Xb[tr], y[tr]).predict(Xb[te])
    sigma = (y - preds).std(ddof=1)
    print(f"    -> Mejor puente: {best_key[0]} / {best_key[1]}  (LOO R2={results[best_key]:.3f}, sigma={sigma:.2f}pp)")
    return dict(model=model, features=best_key[0], sigma=sigma, r2_loo=results[best_key], df=df)


def run_cutoff(label, cutoff):
    print(f"\n{'='*70}\n{label}: corte={cutoff.date()}\n{'='*70}")
    panel = panelmod.build_monthly_panel(cutoff)
    cols = panelmod.NIVEL1_COLS + ["IGAE_target"]
    params = panelmod.clean_params_nivel1(panel)
    phi, lam, R = params["phi"], params["lam"], params["R"]
    print(f"phi={phi:.3f}  cargas: " + ", ".join(f"{k}={v:+.2f}" for k, v in
          sorted(lam.items(), key=lambda kv: -abs(kv[1]))[:5]) + " ...")

    cols_available = [[c for c in cols if not np.isnan(panel.iloc[t][c])] for t in range(len(panel))]

    # --- LP: mayor residuo de estado observado en 2020 (se adapta a lo
    # disponible en cada corte, no se asume un mes ancla fijo) ---
    Q_flat = np.ones(len(panel))
    pf_proxy = m2mod.particle_filter(panel, cols_available, lam, R, phi, m2mod.g_tanh, 1e6, Q_flat,
                                      n_particles=500, seed=1)
    fm = pf_proxy["filtered_mean"]
    idx_2020 = [i for i, d in enumerate(panel.index) if d >= pd.Timestamp("2020-01-01") and i > 0]
    resids_2020 = {panel.index[i]: fm[i] - phi * fm[i - 1] for i in idx_2020 if not np.isnan(fm[i])}
    anchor_month = max(resids_2020, key=lambda d: abs(resids_2020[d]))
    s0 = max(abs(resids_2020[anchor_month]) / 1.0, 1.0) ** 2
    idx_anchor = panel.index.get_loc(anchor_month)
    print(f"LP: mes ancla={anchor_month.date()} (mayor residuo YA disponible en este corte), "
          f"resid={resids_2020[anchor_month]:.2f}, s0={s0:.2f}")

    Q_series = np.ones(len(panel))
    for j, t in enumerate(range(idx_anchor, len(panel))):
        Q_series[t] = 1 + (s0 - 1) * RHO_MONTHLY ** j

    pre_2020 = panel.loc[:"2019-12-01", panelmod.NIVEL1_COLS]
    c = max(1.5 * pre_2020.std().mean() / np.mean(list(lam.values())), 1.0)

    res = m2mod.particle_filter(panel, cols_available, lam, R, phi, m2mod.g_asinh, c, Q_series, seed=21)
    factor_hist = pd.Series(res["filtered_mean"], index=panel.index)
    final_particles, final_weights = res["final_particles"], res["final_weights"]

    bmv_lag1 = panel["BMV_yoy"].shift(1)
    print("  Seleccionando puente ML (factor [+ BMV rezagado] -> IGAE, LOO en ventana limpia)...")
    bridge = select_ml_bridge(factor_hist, bmv_lag1, panel["IGAE_target"], clean_cutoff="2019-10-01")

    h_months = (panelmod.TARGET_MONTH.to_period("M").ordinal - panel.index[-1].to_period("M").ordinal)
    bmv_input = panel["BMV_yoy"].dropna().iloc[-1]  # BMV mas reciente conocido (rezago 1 mes desde el objetivo)
    print(f"  BMV(mas reciente conocido en este corte, a/a)={bmv_input:.1f}%  h_months(hasta mayo)={h_months}")

    rng = np.random.default_rng(5001 + hash(label) % 1000)
    idx_particles = rng.choice(len(final_particles), size=N_DRAWS, p=final_weights)
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        x = final_particles[idx_particles[d]]
        for j in range(h_months):
            Q_j = 1 + (s0 - 1) * RHO_MONTHLY ** (len(panel) - idx_anchor + j)
            x = phi * x + np.sqrt(Q_j) * rng.standard_normal()
        x_in = np.array([[x]]) if bridge["features"] == "factor" else (
            np.array([[bmv_input]]) if bridge["features"] == "BMV" else np.array([[x, bmv_input]]))
        pred = bridge["model"].predict(x_in)[0]
        growth[d] = pred + rng.normal(0, bridge["sigma"])

    mediana, p025, p975 = np.median(growth), np.percentile(growth, 2.5), np.percentile(growth, 97.5)
    print(f"\n  === {label}: pronostico IGAE mayo-2020 (m/m) ===")
    print(f"  Mediana={mediana:.2f}%  IC95%=[{p025:.2f}, {p975:.2f}]  (real publicado: {panelmod.IGAE_REAL_MAYO}%)")

    np.save(OUTPUT / f"{label}_growth_draws.npy", growth)
    return dict(label=label, cutoff=cutoff, mediana=mediana, p025=p025, p975=p975,
                bridge_features=bridge["features"], r2_loo=bridge["r2_loo"], s0=s0, anchor_month=anchor_month)


def main():
    summary = []
    for label, cutoff in CORTES:
        summary.append(run_cutoff(label, cutoff))

    print(f"\n{'='*70}\nRESUMEN: evolucion del pronostico de IGAE mayo-2020 por corte\n{'='*70}")
    print(f"{'Corte':20s} {'Fecha':12s} {'Mediana':>10s} {'IC95%':>22s}")
    for s in summary:
        print(f"{s['label']:20s} {s['cutoff'].date()}  {s['mediana']:>+8.2f}%   "
              f"[{s['p025']:+.2f}, {s['p975']:+.2f}]")
    print(f"{'REAL (24-jul-2020)':20s} {'':12s} {panelmod.IGAE_REAL_MAYO:>+8.2f}%")

    with open(OUTPUT / "resumen_3cortes.pkl", "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()
