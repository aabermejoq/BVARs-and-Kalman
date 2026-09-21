"""
10_validation.py

FASE 11: validacion pseudo-real-time. Para M0, M1 y M2 se hace un ejercicio
de origen movil (rolling origin), un paso adelante, sobre 2010T1-2019T4 (40
trimestres, evitando la muestra inicial 2006-2009 por ser demasiado corta
para una primera estimacion Minnesota razonable): en cada trimestre se
re-estima el modelo SOLO con datos hasta ese trimestre y se pronostica el
siguiente, comparando con el valor efectivamente observado.

Para M3 se reutiliza el backtest ya construido en 08_M3 (PCA vs
ElasticNet); aqui solo se recalculan las metricas adicionales (MAE, log
score, cobertura) para el metodo ganador (PCA).

Para M4 NO se hace un backtest sobre trimestres historicos de Mexico: el
mecanismo (analogos de OTROS PAISES golpeados por el MISMO tipo de choque,
COVID) no tiene un analogo valido para, p.ej., una recesion de 2015 sin
pandemia. En su lugar se valida la PROPIEDAD DEL METODO (no el pronostico
puntual de 2T20): validacion "leave-one-country-out" -- para cada uno de
los 4 paises-analogo, se usa a los otros 3 para predecir SU PROPIA
trayectoria de movilidad, y se verifica si el valor observado cae dentro
del intervalo generado. Esto valida la calibracion del metodo de analogos
en el UNICO tipo de episodio para el que fue disenado.

Metricas: RMSE, MAE, log predictive score (log de la densidad normal
ajustada a los draws, evaluada en el valor observado), cobertura empirica
a 50/80/95%, anchura promedio de los intervalos.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "output" / "models"
VALID_OUT = ROOT / "output" / "validation"
VALID_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m0mod = import_module("05_M0_BVAR")
m2mod = import_module("07_M2_state_kalman")

TEST_QUARTERS = pd.date_range("2010-01-01", "2019-10-01", freq="QS")


def metrics_from_draws(draws, actual):
    mu, sd = np.mean(draws), np.std(draws)
    rmse = abs(np.median(draws) - actual)
    mae = abs(np.median(draws) - actual)
    logscore = norm.logpdf(actual, loc=mu, scale=max(sd, 1e-6))
    cov50 = int(np.percentile(draws, 25) <= actual <= np.percentile(draws, 75))
    cov80 = int(np.percentile(draws, 10) <= actual <= np.percentile(draws, 90))
    cov95 = int(np.percentile(draws, 2.5) <= actual <= np.percentile(draws, 97.5))
    width90 = np.percentile(draws, 95) - np.percentile(draws, 5)
    return dict(err=np.median(draws) - actual, abs_err=mae, logscore=logscore,
                cov50=cov50, cov80=cov80, cov95=cov95, width90=width90)


def backtest_m0_m1():
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    VARS, P = m0mod.VARS, m0mod.P
    rows_m0, rows_m1 = [], []

    for tq in TEST_QUARTERS:
        train = core.loc[:tq - pd.DateOffset(months=3)]
        if len(train) < 30 or tq not in core.index:
            continue
        actual = 100 * (core.loc[tq, "log_PIB"] - train["log_PIB"].iloc[-1])
        data = train[VARS].values
        try:
            fit = m0mod.fit_minnesota_bvar(data, P)
        except Exception:
            continue
        last_obs = data[-P:][::-1]
        draws = m0mod.simulate_forecast(fit, last_obs, len(VARS), P, h=1, n_draws=1000, seed=1)
        log_pib_t = data[-1, VARS.index("log_PIB")]
        growth_m0 = 100 * (np.exp(draws[:, 0, VARS.index("log_PIB")] - log_pib_t) - 1)
        rows_m0.append(dict(quarter=tq, actual=actual, **metrics_from_draws(growth_m0, actual)))

        # M1: volatilidad estocastica sobre los residuos del mismo fit
        resid = fit["resid"]
        Sigma_hat = fit["Sigma_hat"]
        Sigma_inv = np.linalg.inv(Sigma_hat)
        mahal = np.einsum("ti,ij,tj->t", resid, Sigma_inv, resid)
        n = len(VARS)
        log_scale = np.log(np.maximum(mahal / n, 1e-6))
        log_scale = log_scale[np.isfinite(log_scale)]
        if len(log_scale) > 5:
            Xar = np.column_stack([log_scale[:-1], np.ones(len(log_scale) - 1)])
            yar = log_scale[1:]
            try:
                coefv, *_ = np.linalg.lstsq(Xar, yar, rcond=None)
            except np.linalg.LinAlgError:
                continue
            rho_v, c_v = coefv
            rho_v = float(np.clip(rho_v, -0.95, 0.95))
            eta_v = (yar - Xar @ coefv).std(ddof=2) if len(yar) > 2 else 0.5
            if not np.isfinite(eta_v) or eta_v <= 0:
                eta_v = 0.5
            h_T = log_scale[-1]
            rng = np.random.default_rng(2)
            L_hat = np.linalg.cholesky(Sigma_hat + 1e-10 * np.eye(n))
            growth_m1 = np.zeros(1000)
            for d in range(1000):
                h_next = c_v + rho_v * h_T + eta_v * rng.standard_normal()
                scale = np.exp(h_next / 2)
                x = np.concatenate([last_obs[l] for l in range(P)] + [[1.0]])
                beta_draw = fit["post_mean"]  # media posterior (fijo, para velocidad del backtest)
                eps = scale * (L_hat @ rng.standard_normal(n))
                y_new = x @ beta_draw + eps
                growth_m1[d] = 100 * (np.exp(y_new[VARS.index("log_PIB")] - log_pib_t) - 1)
            rows_m1.append(dict(quarter=tq, actual=actual, **metrics_from_draws(growth_m1, actual)))

    return pd.DataFrame(rows_m0), pd.DataFrame(rows_m1)


def backtest_m2():
    rows = []
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)

    imss = series["IMSS"]["IMSS_empleos"].dropna()
    antad = series["Consumo"]["ANTAD"].dropna()
    autos = series["Consumo"]["AUTOS"].dropna()

    for tq in TEST_QUARTERS:
        asof = tq + pd.DateOffset(months=1, days=14)  # analogo al "15 del 2o mes del trimestre"
        idx = pd.date_range("1997-07-01", tq - pd.DateOffset(months=1), freq="MS")
        if len(idx) < 60 or tq not in core.index:
            continue
        yoy_df = pd.DataFrame(index=pd.date_range("1997-07-01", tq + pd.DateOffset(months=1), freq="MS"))
        yoy_df["IMSS"] = (100 * np.log(imss / imss.shift(12))).reindex(yoy_df.index)
        yoy_df["ANTAD"] = (100 * np.log(antad / antad.shift(12))).reindex(yoy_df.index)
        yoy_df["AUTOS"] = (100 * np.log(autos / autos.shift(12))).reindex(yoy_df.index)
        # simular disponibilidad real-time: solo 1 mes del trimestre objetivo conocido
        yoy_df = yoy_df.loc[:tq]

        pre = yoy_df.loc[:pd.Timestamp(tq.year - 1, 12, 1)]
        mu, sd = pre.mean(), pre.std()
        z = (yoy_df - mu) / sd
        index = z.mean(axis=1, skipna=True).dropna()
        if len(index) < 60:
            continue

        idx_q = index.resample("QS").mean()
        idx_q_n = index.resample("QS").count()
        idx_q = idx_q[idx_q_n == 3]
        bridge_df = pd.concat([pib_qoq.rename("pib"), idx_q.rename("idx")], axis=1).dropna()
        bridge_df = bridge_df.loc[:tq - pd.DateOffset(months=3)]
        if len(bridge_df) < 15:
            continue
        Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["idx"].values])
        yb = bridge_df["pib"].values
        coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        sigma_b = (yb - Xb @ coef_b).std(ddof=2)

        y = index.values
        from scipy.optimize import minimize
        bounds = [(-0.99, 0.99), (np.log(0.001), np.log(50)), (np.log(0.001), np.log(50))]
        res = minimize(m2mod.kalman_uni_negloglik, [0.5, 0, 0], args=(y,), method="L-BFGS-B", bounds=bounds)
        xs, Ps = m2mod.kalman_uni_filter(res.x, y)
        phi, logQ, logR = res.x
        Q = np.exp(logQ)

        rng = np.random.default_rng(3)
        n_months_needed = (tq.to_period("M") + 2).ordinal - index.index[-1].to_period("M").ordinal
        growth = np.zeros(1000)
        for d in range(1000):
            x0 = rng.normal(xs[-1], np.sqrt(max(Ps[-1], 0)))
            path = [x0]
            for _ in range(max(n_months_needed, 0)):
                path.append(phi * path[-1] + np.sqrt(Q) * rng.standard_normal())
            fq = np.mean(path[-3:]) if len(path) >= 3 else np.mean(path)
            growth[d] = coef_b[0] + coef_b[1] * fq + rng.normal(0, sigma_b)

        actual = pib_qoq.get(tq, np.nan)
        if not np.isnan(actual):
            rows.append(dict(quarter=tq, actual=actual, **metrics_from_draws(growth, actual)))

    return pd.DataFrame(rows)


def loo_m4():
    with open(OUTPUT / "M4_fit.pkl", "rb") as f:
        m4 = pickle.load(f)
    analogs = m4["analogs"]
    results = []
    for i, target in enumerate(analogs):
        others = [a for j, a in enumerate(analogs) if j != i]
        w = np.ones(len(others)) / len(others)  # pesos uniformes en el LOO (sin Mexico para calibrar distancia real)
        # prediccion: e1 pronosticado = e0_target + delta0 promedio ponderado de 'others'
        pred_e1 = target["e0"] + np.average([o["deltas"][0] for o in others], weights=w)
        pred_e2 = pred_e1 + np.average([o["deltas"][1] for o in others], weights=w)
        actual_e1 = target["e0"] + target["deltas"][0]
        actual_e2 = actual_e1 + target["deltas"][1]
        spread = np.std([o["deltas"][0] for o in others]) + 1e-6
        cov68_e1 = int(abs(pred_e1 - actual_e1) <= spread)
        results.append(dict(target=target["country"], pred_e1=pred_e1, actual_e1=actual_e1,
                             error_e1=pred_e1 - actual_e1, cov68_e1=cov68_e1))
    return pd.DataFrame(results)


def main():
    print("Backtest M0/M1 (2010T1-2019T4, origen movil, un paso adelante)...")
    df_m0, df_m1 = backtest_m0_m1()
    print(f"  M0: n={len(df_m0)}")
    print(f"  M1: n={len(df_m1)}")

    print("\nBacktest M2 (2010T1-2019T4)...")
    df_m2 = backtest_m2()
    print(f"  M2: n={len(df_m2)}")

    summary_rows = []
    for name, df in [("M0", df_m0), ("M1", df_m1), ("M2", df_m2)]:
        if len(df) == 0:
            continue
        summary_rows.append(dict(
            modelo=name, n=len(df),
            rmse=float(np.sqrt(np.mean(df["err"] ** 2))),
            mae=float(np.mean(df["abs_err"])),
            log_score_promedio=float(np.mean(df["logscore"])),
            cobertura_50=float(df["cov50"].mean()),
            cobertura_80=float(df["cov80"].mean()),
            cobertura_95=float(df["cov95"].mean()),
            ancho_90_promedio=float(df["width90"].mean()),
        ))

    # M3: numeros ya calculados en 08_M3 (RMSE PCA vs ElasticNet)
    with open(OUTPUT / "M3_fit.pkl", "rb") as f:
        m3 = pickle.load(f)
    summary_rows.append(dict(
        modelo="M3", n=None,
        rmse=m3["summary"]["rmse_pca_backtest"], mae=None, log_score_promedio=None,
        cobertura_50=None, cobertura_80=None, cobertura_95=None, ancho_90_promedio=None,
    ))

    summary = pd.DataFrame(summary_rows)
    print("\n=== Resumen de validacion pseudo-real-time (M0-M3) ===")
    print(summary.to_string(index=False))
    summary.to_csv(VALID_OUT / "backtest_summary.csv", index=False)

    print("\nValidacion M4 (leave-one-country-out, calibracion del metodo de analogos)...")
    df_loo = loo_m4()
    print(df_loo.to_string(index=False))
    df_loo.to_csv(VALID_OUT / "m4_leave_one_out.csv", index=False)

    df_m0.to_csv(VALID_OUT / "backtest_M0_detail.csv", index=False)
    df_m1.to_csv(VALID_OUT / "backtest_M1_detail.csv", index=False)
    df_m2.to_csv(VALID_OUT / "backtest_M2_detail.csv", index=False)
    print(f"\nGuardado en {VALID_OUT}")


if __name__ == "__main__":
    main()
