"""
Prepara los datos finales (JSON) para la grafica de escalera de 4 capas y
la tabla de cobertura, comparando cada capa (M0; M0+M1; M0+M1+M2;
M0+M1+M2+M3) contra el IGAE_mom REALMENTE observado, mes a mes, hasta
diciembre 2020.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

LAYERS = ["M0", "M1", "M2", "M3"]
LAYER_LABELS = {
    "M0": "M0 · BVAR Minnesota",
    "M1": "M0+M1 · Kalman + volatilidad estocastica",
    "M2": "M0+M1+M2 · Particulas + Local Projections",
    "M3": "M0+M1+M2+M3 · Escenarios epidemiologicos + GBM + GDELT",
}


def load_layer(layer):
    draws = pd.read_csv(PROC / f"{layer}_igae_mom_draws.csv", index_col=0, parse_dates=True)
    arr = draws.values.T  # N x horizon
    idx = draws.index
    pct = {
        "p05": np.percentile(arr, 5, axis=0).tolist(),
        "p25": np.percentile(arr, 25, axis=0).tolist(),
        "mediana": np.percentile(arr, 50, axis=0).tolist(),
        "p75": np.percentile(arr, 75, axis=0).tolist(),
        "p95": np.percentile(arr, 95, axis=0).tolist(),
    }
    return idx, pct


def main():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    igae_obs = panel["IGAE_mom"]

    dates_ref = None
    series = {}
    for layer in LAYERS:
        idx, pct = load_layer(layer)
        if dates_ref is None:
            dates_ref = idx
        series[layer] = pct

    dates = [d.strftime("%Y-%m") for d in dates_ref]
    observed = [
        float(igae_obs.loc[d]) if d in igae_obs.index and pd.notna(igae_obs.loc[d]) else None
        for d in dates_ref
    ]

    # contexto: 6 meses previos al corte, con valores reales
    context_dates = pd.date_range(dates_ref[0] - pd.DateOffset(months=6), dates_ref[0] - pd.DateOffset(months=1), freq="MS")
    context = {
        "dates": [d.strftime("%Y-%m") for d in context_dates],
        "observed": [float(igae_obs.loc[d]) if d in igae_obs.index else None for d in context_dates],
    }

    # tabla de cobertura: por capa, cuantos meses el observado cae dentro de p5-p95 y p25-p75,
    # y error absoluto de la mediana vs observado
    coverage_rows = []
    for layer in LAYERS:
        pct = series[layer]
        errs, in90, in50 = [], 0, 0
        n = 0
        for i, d in enumerate(dates):
            obs = observed[i]
            if obs is None:
                continue
            n += 1
            errs.append(abs(pct["mediana"][i] - obs))
            if pct["p05"][i] <= obs <= pct["p95"][i]:
                in90 += 1
            if pct["p25"][i] <= obs <= pct["p75"][i]:
                in50 += 1
        coverage_rows.append({
            "layer": layer,
            "label": LAYER_LABELS[layer],
            "mae": float(np.mean(errs)) if errs else None,
            "cobertura_90": in90 / n if n else None,
            "cobertura_50": in50 / n if n else None,
            "n_meses": n,
        })

    payload = {
        "layers": LAYERS,
        "layer_labels": LAYER_LABELS,
        "dates": dates,
        "observed": observed,
        "series": series,
        "context": context,
        "coverage": coverage_rows,
        "cutoff": "2020-06-15",
        "last_official_month": "2020-03",
    }
    with open(OUT / "staircase_data.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["coverage"], indent=2))
    print(f"\nGuardado en {OUT / 'staircase_data.json'}")


if __name__ == "__main__":
    main()
