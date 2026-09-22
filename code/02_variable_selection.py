"""
Para cada variable candidata al puente, probar rezagos 0/1/2 meses contra
IGAE_mom con CV10 (regresion univariada) y elegir por validacion cruzada,
no por herencia de un analisis anterior a otra frecuencia/ventana.

Replica exactamente la metodologia que destapo el error de BMV: para cada
candidata, reporta correlacion contemporanea y R^2 de CV10 en cada rezago.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET = "IGAE_mom"
CANDIDATES = [
    "ActividadIndustrial_mom",
    "FBCF_mom",
    "IMCP_mom",
    "ANTAD_mom",
    "AUTOS_mom",
    "TARJETAS_mom",
    "Desempleo_mom",
    "IMSS_mom",
    "EPU_mom",
    "INPC_mom",
    "TC_mom",
    "TIIE_mom",
    "Balanza_mom",
    "BMV_mom",
    "INDPRO_mom",
]


def cv_r2_for_lag(y: pd.Series, x: pd.Series, lag: int, n_splits: int = 10) -> tuple[float, float, int]:
    xl = x.shift(lag)
    df = pd.concat([y, xl], axis=1).dropna()
    df.columns = ["y", "x"]
    n = len(df)
    if n < 20:
        return np.nan, np.nan, n
    corr = df["y"].corr(df["x"])
    X = df[["x"]].values
    Y = df["y"].values
    kf = KFold(n_splits=min(n_splits, n), shuffle=False)
    scores = cross_val_score(LinearRegression(), X, Y, cv=kf, scoring="r2")
    return corr, scores.mean(), n


def run_selection(window: str | None = None):
    """
    window: si se da, ej. '1993-01':'2019-12', restringe la ventana de
    estimacion (para no dejar que el choque COVID 2020 domine la relacion
    de "tiempos normales"; el choque se maneja aparte via LP/escenarios).
    """
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    if window is not None:
        lo, hi = window
        panel = panel.loc[lo:hi]

    y = panel[TARGET]
    rows = []
    best_lag = {}
    for cand in CANDIDATES:
        if cand not in panel.columns:
            continue
        x = panel[cand]
        results = {}
        for lag in (0, 1, 2):
            corr, r2, n = cv_r2_for_lag(y, x, lag)
            results[lag] = (corr, r2, n)
            rows.append(
                {"variable": cand, "rezago_meses": lag, "corr": corr, "cv10_r2": r2, "n_obs": n}
            )
        valid = {k: v[1] for k, v in results.items() if not np.isnan(v[1])}
        if valid:
            best_lag[cand] = max(valid, key=valid.get)

    table = pd.DataFrame(rows)
    return table, best_lag


if __name__ == "__main__":
    print("=== Ventana completa disponible (puede incluir choque COVID) ===")
    table_full, best_full = run_selection(window=None)
    print(table_full.pivot(index="variable", columns="rezago_meses", values="cv10_r2").round(4))
    print()
    print("Mejor rezago por variable (ventana completa):")
    for k, v in best_full.items():
        print(f"  {k}: rezago {v} meses")

    print()
    print("=== Ventana pre-COVID (1993-2019), para no dejar que el choque domine ===")
    table_pre, best_pre = run_selection(window=("1993-01-01", "2019-12-31"))
    print(table_pre.pivot(index="variable", columns="rezago_meses", values="cv10_r2").round(4))
    print()
    print("Mejor rezago por variable (pre-COVID):")
    for k, v in best_pre.items():
        print(f"  {k}: rezago {v} meses")

    table_full.to_csv(PROC / "variable_selection_full.csv", index=False)
    table_pre.to_csv(PROC / "variable_selection_precovid.csv", index=False)
