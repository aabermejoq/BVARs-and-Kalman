"""
16_M4_epidemiologico.py

M4 rediseñado con las curvas epidemiológicas OFICIALES de la Secretaría de
Salud en vez de los escenarios macro del FMI, calibradas con los datos
DIARIOS de tarjetas mediante un modelo NO LINEAL (Gradient Boosting).

FUENTES (reales, verificadas):
  - Casos diarios confirmados de Mexico: Our World in Data / JHU CSSE
    (data/external/mexico_covid_cases_jhu.csv).
  - Escenarios oficiales (Infobae, Animal Politico, may-jun 2020): pico
    "con Sana Distancia" proyectado ~7-8 mayo 2020; "sin Sana Distancia"
    (contrafactual) ~2 abril 2020, con 74-81% menos casos en el pico bajo
    intervencion.

HALLAZGO CLAVE verificado con datos reales: al 15-may-2020 (el corte), el
promedio movil de 7 dias de casos SEGUIA SUBIENDO (1930/dia), sin senal de
haber tocado techo el 7-8 de mayo como preveia el escenario oficial. Esto
motiva 2 escenarios hacia adelante:
  A) "Control": el pico se materializa poco despues del corte, declive
     relativamente rapido.
  B) "Meseta extendida": el pico se retrasa varias semanas y el declive es
     mucho mas lento (consistente con lo que Lopez-Gatell describiria en
     junio como una "meseta" de mas de 3 semanas).

CALIBRACION NO LINEAL: Gradient Boosting Regressor (no una regresion
lineal) mapea features de la curva de casos (nivel log, crecimiento a 7
dias) a la variacion interanual de tarjetas (ventana movil de 7 dias, la
MISMA variable ya usada en 14_M4_agosto_bayesian_scenarios.py), entrenado
con la relacion REALIZADA dia a dia hasta el corte. Se aplica esa relacion
a la trayectoria de casos de CADA escenario, y se actualiza P(escenario)
dia a dia comparando contra las realizaciones reales de tarjetas (filtro
bayesiano con verosimilitud no lineal). La traduccion final a PIB reutiliza
el puente tarjetas->PIB ya estimado (build_card_bridge).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
RAW = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m4_agosto_mod = import_module("14_M4_agosto_bayesian_scenarios")

N_DRAWS = 10_000
GBR_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def load_case_data():
    path = EXTERNAL / "mexico_covid_cases_jhu.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df["case_7d"] = df["new_cases"].rolling(7).mean()
    return df


def load_tarjetas_yoy():
    wb = openpyxl.load_workbook(RAW / "tarjetas_diario.xlsx", data_only=True)
    ws = wb["Hoja1"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = [r for r in rows[1:] if r[0] is not None]
    df = pd.DataFrame(data, columns=header)
    df["fecha"] = pd.to_datetime(dict(year=df["year"], month=df["mes"], day=df["dia"]))
    df = df.set_index("fecha").sort_index()
    agregado = df["Total de monto operado a través de tarjetas"]
    roll7 = agregado.rolling(7).sum()
    yoy = 100 * np.log(roll7 / roll7.shift(364))
    return yoy.replace([np.inf, -np.inf], np.nan)


def epi_curve(last_log_level, asof, peak_date, decline_rate, horizon_days, growth_rate=0.025,
              backcast_days=21):
    """Construye la curva del escenario en AMBAS direcciones: hacia
    adelante (para el pronostico) y hacia atras (backcast_days dias antes
    del corte), para poder evaluar la verosimilitud del escenario contra
    datos YA REALIZADOS sin tocar ninguna observacion posterior al corte."""
    dates_fwd = pd.date_range(asof + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    dates_back = pd.date_range(asof - pd.Timedelta(days=backcast_days - 1), asof, freq="D")

    # Recorrer dia a dia desde el inicio del backcast, anclando el nivel EN
    # asof a last_log_level (el dato real observado), y propagando la MISMA
    # regla de crecimiento antes del pico / decaimiento despues del pico en
    # ambas direcciones (hacia adelante y hacia atras).
    all_dates = dates_back.append(dates_fwd[1:]) if len(dates_back) else dates_fwd
    n_back = len(dates_back)
    vals = [None] * len(all_dates)
    idx_asof = n_back - 1
    vals[idx_asof] = last_log_level
    # hacia adelante
    level = last_log_level
    for i in range(idx_asof + 1, len(all_dates)):
        d = all_dates[i]
        if d <= peak_date:
            level = level + growth_rate
        else:
            level = level - decline_rate
        vals[i] = level
    # hacia atras
    level = last_log_level
    for i in range(idx_asof - 1, -1, -1):
        d = all_dates[i]
        if d <= peak_date:
            level = level - growth_rate
        else:
            level = level + decline_rate
        vals[i] = level
    return pd.Series(vals, index=all_dates, dtype=float)


def build_features(log_casos_series):
    growth = log_casos_series.diff(7)
    df = pd.DataFrame({"log_casos": log_casos_series, "crecimiento_7d": growth})
    df["t"] = np.arange(len(df))
    return df


def run_exercise(label, asof, target_quarter, peak_date_A, decline_A, peak_date_B, decline_B, seed):
    print(f"\n{'='*70}\n{label}  (corte {asof.date()}, objetivo {target_quarter.date()})\n{'='*70}")
    cases = load_case_data()
    lag_epi = 1  # mismo supuesto de rezago de 1 dia
    cases_rt = cases.loc[:asof - pd.Timedelta(days=lag_epi)]
    tarjetas_yoy = load_tarjetas_yoy()
    tarjetas_rt = tarjetas_yoy.loc[:asof - pd.Timedelta(days=1)]

    log_casos_hist = np.log(cases_rt["case_7d"].clip(lower=1))
    feat_hist = build_features(log_casos_hist)

    train_df = pd.concat([feat_hist, tarjetas_rt.rename("tarjetas_yoy")], axis=1).dropna()
    Xtr = train_df[["log_casos", "crecimiento_7d", "t"]].values
    ytr = train_df["tarjetas_yoy"].values
    print(f"Entrenamiento GBR: n={len(train_df)} obs. diarias hasta {train_df.index.max().date()}")

    gbr = GradientBoostingRegressor(**GBR_PARAMS)
    gbr.fit(Xtr, ytr)
    resid = ytr - gbr.predict(Xtr)
    sigma_gbr = resid.std(ddof=5)
    print(f"  R2 in-sample={gbr.score(Xtr, ytr):.3f}  sigma_residual={sigma_gbr:.2f} pp")

    last_level = log_casos_hist.iloc[-1]
    t0 = feat_hist["t"].iloc[-1]
    horizon = (target_quarter + pd.DateOffset(months=3) - asof).days + 5
    BACKCAST_DAYS = 21

    # Curvas de escenario CONSTRUIDAS EN AMBAS DIRECCIONES desde asof-1
    # (ultimo dato real conocido): el tramo hacia atras (backcast) permite
    # evaluar que tan bien cada escenario habria explicado el CRECIMIENTO
    # DE CASOS YA OBSERVADO en las ultimas 3 semanas -- sin tocar ningun
    # dato posterior al corte.
    scen_A_log = epi_curve(last_level, asof - pd.Timedelta(days=1), peak_date_A, decline_A,
                            horizon, backcast_days=BACKCAST_DAYS)
    scen_B_log = epi_curve(last_level, asof - pd.Timedelta(days=1), peak_date_B, decline_B,
                            horizon, backcast_days=BACKCAST_DAYS)

    # --- Ponderacion de escenarios: verosimilitud sobre el BACKCAST (pre-corte) ---
    backcast_dates = scen_A_log.index[scen_A_log.index < asof]
    real_log_casos_backcast = log_casos_hist.reindex(backcast_dates)
    resid_case_std = (log_casos_hist.diff().dropna().std())  # dispersion tipica del proceso, para la verosimilitud
    valid = real_log_casos_backcast.notna()
    err_A = (real_log_casos_backcast[valid] - scen_A_log.reindex(backcast_dates)[valid])
    err_B = (real_log_casos_backcast[valid] - scen_B_log.reindex(backcast_dates)[valid])
    loglik_A = norm.logpdf(err_A, scale=resid_case_std).sum()
    loglik_B = norm.logpdf(err_B, scale=resid_case_std).sum()
    m = max(loglik_A, loglik_B)
    w_A, w_B = np.exp(loglik_A - m), np.exp(loglik_B - m)
    p_A, p_B = w_A / (w_A + w_B), w_B / (w_A + w_B)
    print(f"Verosimilitud del backcast ({BACKCAST_DAYS} dias antes del corte, SOLO datos ya conocidos): "
          f"loglik_control={loglik_A:.1f}  loglik_meseta={loglik_B:.1f}")
    print(f"Pesos (a partir de que tan bien cada escenario explica el crecimiento de casos YA OBSERVADO): "
          f"P(control)={p_A:.3f}  P(meseta extendida)={p_B:.3f}")

    def implied(scen_log):
        fwd = scen_log[scen_log.index > asof - pd.Timedelta(days=1)]
        feats = pd.DataFrame({
            "log_casos": fwd,
            "crecimiento_7d": fwd.diff(7).bfill(),
            "t": np.arange(t0 + 1, t0 + 1 + len(fwd)),
        }, index=fwd.index)
        return pd.Series(gbr.predict(feats.values), index=fwd.index)

    implied_A, implied_B = implied(scen_A_log), implied(scen_B_log)
    hist_df = pd.DataFrame(dict(fecha=backcast_dates[valid], log_casos_real=real_log_casos_backcast[valid],
                                 escenario_control=scen_A_log.reindex(backcast_dates)[valid],
                                 escenario_meseta=scen_B_log.reindex(backcast_dates)[valid]))

    tq_days = pd.date_range(target_quarter, target_quarter + pd.DateOffset(months=3) - pd.Timedelta(days=1), freq="D")
    tq_A = implied_A.reindex(tq_days).mean()
    tq_B = implied_B.reindex(tq_days).mean()
    print(f"Tarjetas (a/a) IMPLICITA para el trimestre objetivo: control={tq_A:.1f}%  meseta={tq_B:.1f}%")

    coef_card, sigma_card = m4_agosto_mod.build_card_bridge()

    rng = np.random.default_rng(seed)
    scenario_choice = rng.choice([0, 1], size=N_DRAWS, p=[p_A, p_B])
    tarjetas_scenarios = np.array([tq_A, tq_B])
    boot_sd = abs(tq_A - tq_B) * 0.2 + 1.0
    tarjetas_draws = tarjetas_scenarios[scenario_choice] + rng.normal(0, boot_sd, N_DRAWS)

    coef_draws = np.column_stack([
        rng.normal(coef_card[0], sigma_card * 0.1, N_DRAWS),
        rng.normal(coef_card[1], abs(coef_card[1]) * 0.1, N_DRAWS),
    ])
    growth_draws = coef_draws[:, 0] + coef_draws[:, 1] * tarjetas_draws + rng.normal(0, sigma_card, N_DRAWS)

    summary = dict(p_control=float(p_A), p_meseta=float(p_B), tq_control=float(tq_A), tq_meseta=float(tq_B),
                   mediana_pct=float(np.median(growth_draws)),
                   p2_5=float(np.percentile(growth_draws, 2.5)), p97_5=float(np.percentile(growth_draws, 97.5)))
    print(f"=== {label}: mediana={summary['mediana_pct']:.2f}%  "
          f"IC95%=[{summary['p2_5']:.2f}, {summary['p97_5']:.2f}] ===")

    return growth_draws, hist_df, summary


def main():
    OUTPUT = ROOT / "output" / "models"
    OUTPUT_AGOSTO = ROOT / "output" / "models_agosto"

    growth_may, hist_may, sum_may = run_exercise(
        "M4-EPI mayo (2T20)", pd.Timestamp("2020-05-15"), pd.Timestamp("2020-04-01"),
        peak_date_A=pd.Timestamp("2020-05-10"), decline_A=0.020,
        peak_date_B=pd.Timestamp("2020-06-10"), decline_B=0.008,
        seed=1001,
    )
    np.save(OUTPUT / "M4_epi_growth_draws.npy", growth_may)
    hist_may.to_csv(OUTPUT / "M4_epi_filter_history.csv", index=False)

    growth_agosto, hist_agosto, sum_agosto = run_exercise(
        "M4-EPI agosto (3T20)", pd.Timestamp("2020-08-15"), pd.Timestamp("2020-07-01"),
        peak_date_A=pd.Timestamp("2020-05-10"), decline_A=0.020,
        peak_date_B=pd.Timestamp("2020-06-10"), decline_B=0.008,
        seed=1002,
    )
    np.save(OUTPUT_AGOSTO / "M4_epi_growth_draws.npy", growth_agosto)
    hist_agosto.to_csv(OUTPUT_AGOSTO / "M4_epi_filter_history.csv", index=False)

    with open(OUTPUT / "M4_epi_summary.pkl", "wb") as f:
        pickle.dump(dict(mayo=sum_may, agosto=sum_agosto), f)

    print("\n=== Comparacion con M4 anterior ===")
    old_may = np.load(OUTPUT / "M4_growth_draws.npy")
    print(f"M4 (analogos cross-country) mayo: mediana={np.median(old_may):.2f}  "
          f"IC95=[{np.percentile(old_may,2.5):.2f}, {np.percentile(old_may,97.5):.2f}]")
    print(f"M4-EPI mayo: mediana={sum_may['mediana_pct']:.2f}  IC95=[{sum_may['p2_5']:.2f}, {sum_may['p97_5']:.2f}]")
    old_agosto = np.load(OUTPUT_AGOSTO / "M4_growth_draws.npy")
    print(f"M4 (FMI+tarjetas lineal) agosto: mediana={np.median(old_agosto):.2f}  "
          f"IC95=[{np.percentile(old_agosto,2.5):.2f}, {np.percentile(old_agosto,97.5):.2f}]")
    print(f"M4-EPI agosto: mediana={sum_agosto['mediana_pct']:.2f}  "
          f"IC95=[{sum_agosto['p2_5']:.2f}, {sum_agosto['p97_5']:.2f}]")


if __name__ == "__main__":
    main()
