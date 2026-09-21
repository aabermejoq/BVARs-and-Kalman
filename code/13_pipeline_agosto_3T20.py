"""
13_pipeline_agosto_3T20.py

Ejercicio adicional (a peticion del usuario): repite TODO el pipeline
M0-M4, pero con corte de informacion al 15-ago-2020, pronosticando el PIB
de 3T20. A esa fecha, el PIB de 2T20 YA ES CONOCIDO (el PIB "oportuno"
trimestral se publica ~32 dias despues del cierre del trimestre: el cierre
de 2T20 es el 30-jun-2020, por lo que el dato se publica ~1-ago-2020,
antes del corte de 15-ago-2020). Por tanto el "choque" que vive el
ejercicio de agosto es DISTINTO al de mayo: aqui ya sabemos que el PIB se
desplomo ~-19% en 2T20, y la pregunta es si 3T20 rebota fuerte (forma de V)
o si el choque persiste (forma de U/L) -- exactamente la pregunta que el
FMI planteaba con sus escenarios de abril-2020.

M0 y M1: identicos metodologicamente a mayo, solo que reestimados con la
muestra ampliada (hasta 2T20) y pronosticando h=1 trimestre (3T20).

M2 y M3: version "DFM robusto" (EM regularizado) de 07b/08b, reestimados
con el panel de indicadores mensuales ampliado hasta el corte de agosto.

M4: rediseno completo con el filtro bayesiano secuencial de escenarios del
FMI (abril-2020), actualizado dia a dia con la base DIARIA de tarjetas de
Banxico (supuesto: rezago de 1 dia, series originales). Ver la funcion
`m4_agosto_bayesian_scenarios` para el detalle.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "output" / "models_agosto"
FIG_OUT = ROOT / "output" / "figures_agosto"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
t3 = import_module("03_transformations")
m0mod = import_module("05_M0_BVAR")
m1mod = import_module("06_M1_volatility")
m2b = import_module("07b_M2_dfm_robusto")
m3b = import_module("08b_M3_dfm_robusto")
m3mod = import_module("08_M3_high_frequency_state")
tarj = import_module("12_tarjetas_daily_audit")

ASOF = pd.Timestamp("2020-08-15")
TARGET_Q = pd.Timestamp("2020-07-01")  # 3T20
N_DRAWS = 10_000


def build_core_agosto():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib_q = t3.quarterly_with_completeness(series["PIB"]["PIB"], "PIB", asof=ASOF, agg="mean")
    inpc_q = t3.quarterly_with_completeness(series["INPC"]["INPC"], "INPC", asof=ASOF, agg="mean")
    tiie_q = t3.quarterly_with_completeness(series["TIIE"]["TIIE"], "TIIE", asof=ASOF, agg="mean")
    tc_q = t3.quarterly_with_completeness(series["TC"]["TC"], "TC", asof=ASOF, agg="mean")
    core = pd.DataFrame({
        "log_PIB": np.log(pib_q), "log_INPC": np.log(inpc_q),
        "TIIE": tiie_q, "log_TC": np.log(tc_q),
    }).dropna()
    assert core.index.max() == pd.Timestamp("2020-04-01"), core.index.max()
    return core


def run_m0(core):
    data = core[m0mod.VARS].values
    fit = m0mod.fit_minnesota_bvar(data, m0mod.P)
    last_obs = data[-m0mod.P:][::-1]
    draws = m0mod.simulate_forecast(fit, last_obs, len(m0mod.VARS), m0mod.P, h=1, n_draws=N_DRAWS, seed=101)
    log_pib_t = data[-1, m0mod.VARS.index("log_PIB")]
    growth = 100 * (np.exp(draws[:, 0, m0mod.VARS.index("log_PIB")] - log_pib_t) - 1)
    return growth, fit


def run_m1(core, fit_m0):
    data = core[m0mod.VARS].values
    n, P = len(m0mod.VARS), m0mod.P
    resid = fit_m0["resid"]
    Sigma_hat = fit_m0["Sigma_hat"]
    Sigma_inv = np.linalg.inv(Sigma_hat)
    mahal = np.einsum("ti,ij,tj->t", resid, Sigma_inv, resid)
    log_scale = np.log(np.maximum(mahal / n, 1e-6))
    Xar = np.column_stack([log_scale[:-1], np.ones(len(log_scale) - 1)])
    yar = log_scale[1:]
    coefv, *_ = np.linalg.lstsq(Xar, yar, rcond=None)
    rho_v, c_v = coefv
    rho_v = float(np.clip(rho_v, -0.95, 0.95))
    eta_v = (yar - Xar @ coefv).std(ddof=2)
    h_T = log_scale[-1]

    rng = np.random.default_rng(202)
    L_hat = np.linalg.cholesky(Sigma_hat + 1e-10 * np.eye(n))
    last_obs = data[-P:][::-1]
    log_pib_t = data[-1, m0mod.VARS.index("log_PIB")]
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        beta_d = np.zeros((n * P + 1, n))
        for i in range(n):
            beta_d[:, i] = rng.multivariate_normal(fit_m0["post_mean"][:, i], fit_m0["post_var"][:, :, i])
        h_next = c_v + rho_v * h_T + eta_v * rng.standard_normal()
        scale = np.exp(h_next / 2)
        x = np.concatenate([last_obs[l] for l in range(P)] + [[1.0]])
        eps = scale * (L_hat @ rng.standard_normal(n))
        y_new = x @ beta_d + eps
        growth[d] = 100 * (np.exp(y_new[m0mod.VARS.index("log_PIB")] - log_pib_t) - 1)
    return growth, dict(rho=rho_v, c=c_v, eta=eta_v, h_T=h_T)


def run_m2_dfm(core):
    panel = m2b.build_data(asof=ASOF, fast_end=pd.Timestamp("2020-07-01"), core_path=None)
    # Reconstruye con la muestra de PIB ampliada (core_agosto), no la de mayo
    panel = panel.drop(columns=["PIB_q"])
    pib_qoq = 100 * core["log_PIB"].diff(1)
    pib_series = pd.Series(index=panel.index, dtype=float)
    for dte, val in pib_qoq.items():
        close_month = dte + pd.DateOffset(months=2)
        if close_month in pib_series.index:
            pib_series[close_month] = val
    panel["PIB_q"] = pib_series

    fit = m2b.em_dfm(panel, verbose=False)
    x_filt = fit["x_filt"][:, 0]
    factor_m = pd.Series(x_filt, index=panel.index)
    factor_q = factor_m.resample("QS").mean()
    factor_q_n = factor_m.resample("QS").count()
    factor_q = factor_q[factor_q_n == 3]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)

    h_months = TARGET_Q.to_period("M").ordinal + 2 - panel.index[-1].to_period("M").ordinal
    rng = np.random.default_rng(303)
    phi = fit["phi"]
    s_last, P_last = fit["x_filt"][-1], fit["P_filt"][-1]
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        xs = [s_d[0]]
        for _ in range(h_months):
            xs.append(phi * xs[-1] + rng.standard_normal())
        f_q = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth[d] = coef_d[0] + coef_d[1] * f_q + rng.normal(0, sigma_b)
    return growth, dict(fit=fit, coef_b=coef_b, sigma_b=sigma_b, panel_end=panel.index[-1])


def run_m3_dfm(core):
    panel_full = m3mod.build_panel()
    # Reconstruir el panel M3 con cortes real-time a agosto en vez de mayo
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    idx = pd.date_range("1993-01-01", "2020-07-01", freq="MS")
    panel = pd.DataFrame(index=idx)
    panel["ActividadIndustrial"] = yoy(series["otros"]["ActividadIndustrial"]).reindex(idx)
    panel["FBCF"] = yoy(series["otros"]["FBCF"]).reindex(idx)
    panel["IMCP"] = yoy(series["Consumo"]["IMCP"]).reindex(idx)
    panel["Exportaciones"] = yoy(series["Balanza"]["Exportaciones"]).reindex(idx)
    panel["Importaciones"] = yoy(series["Balanza"]["Importaciones"]).reindex(idx)
    panel["IMSS"] = yoy(series["IMSS"]["IMSS_empleos"]).reindex(idx)
    panel["ANTAD"] = yoy(series["Consumo"]["ANTAD"]).reindex(idx)
    panel["AUTOS"] = yoy(series["Consumo"]["AUTOS"]).reindex(idx)
    fred = pd.read_csv(RAW.parent / "external" / "fred_us_benchmarks.csv", parse_dates=["date"]).set_index("date")
    indpro = fred["INDPRO"].dropna()
    indpro = indpro[indpro.index <= pd.Timestamp("2020-06-01")]
    panel["INDPRO_EEUU"] = yoy(indpro).reindex(idx)

    panel.loc[panel.index > pd.Timestamp("2020-05-01"), ["ActividadIndustrial", "FBCF", "IMCP"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-06-01"), ["Exportaciones", "Importaciones", "INDPRO_EEUU"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-07-01"), ["IMSS", "ANTAD", "AUTOS"]] = np.nan

    monthly_cols = list(panel.columns)
    pib_qoq = 100 * core["log_PIB"].diff(1)
    pib_series = pd.Series(index=panel.index, dtype=float)
    for dte, val in pib_qoq.items():
        close_month = dte + pd.DateOffset(months=2)
        if close_month in pib_series.index:
            pib_series[close_month] = val
    panel["PIB_q"] = pib_series

    fit = m3b.em_dfm_general(panel, monthly_cols, verbose=False)
    x_filt = fit["x_filt"][:, 0]
    factor_m = pd.Series(x_filt, index=panel.index)
    factor_q = factor_m.resample("QS").mean()
    factor_q_n = factor_m.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)

    last_month = panel.dropna(how="all", subset=monthly_cols).index[-1]
    h_months = TARGET_Q.to_period("M").ordinal + 2 - last_month.to_period("M").ordinal
    rng = np.random.default_rng(404)
    phi = fit["phi"]
    last_i = panel.index.get_loc(last_month)
    s_last, P_last = fit["x_filt"][last_i], fit["P_filt"][last_i]
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        xs = [s_d[0]]
        for _ in range(h_months):
            xs.append(phi * xs[-1] + rng.standard_normal())
        f_q = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth[d] = coef_d[0] + coef_d[1] * f_q + rng.normal(0, sigma_b)
    return growth, dict(fit=fit, coef_b=coef_b, sigma_b=sigma_b, last_month=last_month)


def main():
    print(f"=== PIPELINE AGOSTO 2020 -> pronostico 3T20 (corte {ASOF.date()}) ===\n")
    core = build_core_agosto()
    print(f"Panel CORE: {core.index.min().date()} -> {core.index.max().date()} (n={len(core)})")
    log_pib_2t20 = core["log_PIB"].iloc[-1]
    print(f"PIB 2T20 (YA CONOCIDO al corte de agosto): log={log_pib_2t20:.4f}")

    print("\n--- M0 ---")
    g0, fit0 = run_m0(core)
    print(f"Mediana={np.median(g0):.2f}%  IC95%=[{np.percentile(g0,2.5):.2f}, {np.percentile(g0,97.5):.2f}]")

    print("\n--- M1 ---")
    g1, info1 = run_m1(core, fit0)
    print(f"Mediana={np.median(g1):.2f}%  IC95%=[{np.percentile(g1,2.5):.2f}, {np.percentile(g1,97.5):.2f}]")
    print(f"  h_T (2T20, YA con el colapso completo)={info1['h_T']:.2f} "
          f"(multiplicador {np.exp(info1['h_T']):.2f}x)")

    print("\n--- M2 (DFM robusto) ---")
    g2, info2 = run_m2_dfm(core)
    print(f"Mediana={np.median(g2):.2f}%  IC95%=[{np.percentile(g2,2.5):.2f}, {np.percentile(g2,97.5):.2f}]")

    print("\n--- M3 (DFM robusto, sin IGAE) ---")
    g3, info3 = run_m3_dfm(core)
    print(f"Mediana={np.median(g3):.2f}%  IC95%=[{np.percentile(g3,2.5):.2f}, {np.percentile(g3,97.5):.2f}]")

    np.save(OUTPUT / "M0_growth_draws.npy", g0)
    np.save(OUTPUT / "M1_growth_draws.npy", g1)
    np.save(OUTPUT / "M2_growth_draws.npy", g2)
    np.save(OUTPUT / "M3_growth_draws.npy", g3)
    with open(OUTPUT / "core_info.pkl", "wb") as f:
        pickle.dump(dict(core=core, fit0=fit0, info1=info1, info2=info2, info3=info3), f)
    print(f"\nGuardado M0-M3 en {OUTPUT}")


if __name__ == "__main__":
    main()
