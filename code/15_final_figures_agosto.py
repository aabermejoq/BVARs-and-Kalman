"""
15_final_figures_agosto.py

Densidades finales y tabla comparativa para el ejercicio de agosto/3T20,
analogo a 11_final_figures.py pero con el pipeline reestimado al corte
15-ago-2020 (M0/M1 BVAR+SV, M2/M3 DFM robusto, M4 filtro bayesiano
FMI+tarjetas).
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
MODELS_DIR = ROOT / "output" / "models_agosto"
FIG_OUT = ROOT / "output" / "figures_agosto"
FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["M0", "M1", "M2", "M3", "M4"]
COLORS = {"M0": "#7f8c8d", "M1": "#2980b9", "M2": "#27ae60", "M3": "#e67e22", "M4": "#c0392b"}
LABELS = {
    "M0": "M0 · Benchmark BVAR",
    "M1": "M1 · + Volatilidad estocástica",
    "M2": "M2 · + DFM robusto (IMSS/ANTAD/AUTOS)",
    "M3": "M3 · + DFM robusto (panel amplio, sin IGAE)",
    "M4": "M4 · + Filtro bayesiano FMI + tarjetas",
}


def get_actual():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib = series["PIB"]["PIB"]
    q2, q3 = pib.loc["2020-04-01"], pib.loc["2020-07-01"]
    return 100 * (np.log(q3) - np.log(q2))


def main():
    actual = get_actual()
    print(f"Valor observado (ex-post): PIB 3T20 = {actual:.2f}% t/t")

    draws = {m: np.load(MODELS_DIR / f"{m}_growth_draws.npy") for m in MODELS}

    rows = []
    for m in MODELS:
        d = draws[m]
        mu, sd = d.mean(), d.std()
        rmse = float(np.sqrt(np.mean((d - actual) ** 2)))
        mae = float(np.mean(np.abs(d - actual)))
        logscore = float(norm.logpdf(actual, loc=mu, scale=max(sd, 1e-6)))
        rows.append(dict(
            modelo=m, mediana=float(np.median(d)),
            P95_lo=float(np.percentile(d, 2.5)), P95_hi=float(np.percentile(d, 97.5)),
            RMSE=rmse, MAE=mae, log_score=logscore,
            cubre_95=bool(np.percentile(d, 2.5) <= actual <= np.percentile(d, 97.5)),
        ))
    table = pd.DataFrame(rows)
    print("\n=== Tabla comparativa agosto/3T20 ===")
    print(table.to_string(index=False))
    table.to_csv(FIG_OUT / "tabla_comparativa_agosto_3T20.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(-20, 25, 1000)
    for m in MODELS:
        kde = gaussian_kde(draws[m])
        ax.plot(x_grid, kde(x_grid), color=COLORS[m], linewidth=2.2, label=LABELS[m])
        ax.fill_between(x_grid, kde(x_grid), color=COLORS[m], alpha=0.06)

    ax.axvline(actual, color="black", linestyle="--", linewidth=1.8,
               label=f"Observado 3T20: {actual:.1f}% (ex-post)")
    ax.set_xlabel("Crecimiento del PIB, 3T20 (% t/t)", fontsize=11)
    ax.set_ylabel("Densidad predictiva", fontsize=11)
    ax.set_title("México 3T20: pronóstico con corte 15-ago-2020\n"
                 "(2T20 ya conocido: colapso de -21%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "densidades_agosto_3T20.png", dpi=200)
    print(f"\nGuardado: {FIG_OUT / 'densidades_agosto_3T20.png'}")


if __name__ == "__main__":
    main()
