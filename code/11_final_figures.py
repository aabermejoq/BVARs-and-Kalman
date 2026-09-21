"""
11_final_figures.py

FASE 12: densidades predictivas finales y tabla comparativa M0-M4.

El valor observado del PIB de 2T20 se usa UNICAMENTE aqui, como
informacion ex post para evaluar los pronosticos ya generados. No entro a
ningun paso de estimacion, seleccion de variables o de hiperparametros en
las fases anteriores (verificable: 05-09_*.py nunca leen esta cifra).

Definicion de crecimiento: log-diferencia trimestral del PIB
desestacionalizado a precios de 2018 (INEGI-BIE), en el mismo formato que
las densidades de M0-M4 (100*log(PIB_2T20/PIB_1T20)).
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
OUTPUT = ROOT / "output" / "models"
FIG_OUT = ROOT / "output" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["M0", "M1", "M2", "M3", "M4"]
COLORS = {"M0": "#7f8c8d", "M1": "#2980b9", "M2": "#27ae60", "M3": "#e67e22", "M4": "#c0392b"}
LABELS = {
    "M0": "M0 · Benchmark BVAR",
    "M1": "M1 · + Volatilidad estocástica",
    "M2": "M2 · + Kalman (IMSS/ANTAD/AUTOS)",
    "M3": "M3 · + Factor amplio (PCA/ML)",
    "M4": "M4 · + Análogos cross-country (ML)",
}


def get_actual():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib = series["PIB"]["PIB"]
    q1, q2 = pib.loc["2020-01-01"], pib.loc["2020-04-01"]
    return 100 * (np.log(q2) - np.log(q1))


def main():
    actual = get_actual()
    print(f"Valor observado (ex-post, NO usado en estimacion): PIB 2T20 = {actual:.2f}% t/t")

    draws = {m: np.load(OUTPUT / f"{m}_growth_draws.npy") for m in MODELS}

    rows = []
    for m in MODELS:
        d = draws[m]
        mu, sd = d.mean(), d.std()
        rmse = float(np.sqrt(np.mean((d - actual) ** 2)))
        mae = float(np.mean(np.abs(d - actual)))
        logscore = float(norm.logpdf(actual, loc=mu, scale=max(sd, 1e-6)))
        rows.append(dict(
            modelo=m, mediana=float(np.median(d)),
            P50_lo=float(np.percentile(d, 25)), P50_hi=float(np.percentile(d, 75)),
            P80_lo=float(np.percentile(d, 10)), P80_hi=float(np.percentile(d, 90)),
            P95_lo=float(np.percentile(d, 2.5)), P95_hi=float(np.percentile(d, 97.5)),
            RMSE=rmse, MAE=mae, log_score=logscore,
            cubre_95=bool(np.percentile(d, 2.5) <= actual <= np.percentile(d, 97.5)),
        ))

    table = pd.DataFrame(rows)
    print("\n=== Tabla comparativa final M0-M4 (evaluada contra el PIB 2T20 observado ex-post) ===")
    print(table.to_string(index=False))
    table.to_csv(FIG_OUT / "tabla_comparativa_M0_M4.csv", index=False)

    # --- Figura: 5 densidades + linea vertical del valor observado ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(-30, 15, 1000)
    for m in MODELS:
        d = draws[m]
        kde = gaussian_kde(d)
        ax.plot(x_grid, kde(x_grid), color=COLORS[m], linewidth=2.2, label=LABELS[m])
        ax.fill_between(x_grid, kde(x_grid), color=COLORS[m], alpha=0.06)

    ax.axvline(actual, color="black", linestyle="--", linewidth=1.8,
               label=f"Observado 2T20: {actual:.1f}% (ex-post)")

    ax.set_xlabel("Crecimiento del PIB, 2T20 (% t/t)", fontsize=11)
    ax.set_ylabel("Densidad predictiva", fontsize=11)
    ax.set_title("México 2T20: la escalera M0→M4\nDistribuciones predictivas construidas al 15-may-2020",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-30, 15)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "densidades_M0_M4.png", dpi=200)
    print(f"\nGuardado: {FIG_OUT / 'densidades_M0_M4.png'} y {FIG_OUT / 'tabla_comparativa_M0_M4.csv'}")

    # --- Tabla de la escalera (Seccion 20) ---
    escalera = pd.DataFrame([
        dict(modelo="M0", info_adicional="Ninguna (benchmark)",
             problema="Ninguno -- punto de referencia",
             variables_nuevas="PIB, INPC, TIIE, TC (CORE)",
             metodo="BVAR Minnesota (Litterman), Normal-Normal por ecuación"),
        dict(modelo="M1", info_adicional="Ninguna (misma información que M0)",
             problema="Observaciones extremas / shock ≠ cambio de régimen",
             variables_nuevas="Ninguna -- mismos coeficientes que M0",
             metodo="Volatilidad estocástica común (aprox. Harvey-Ruiz-Shephard 1994)"),
        dict(modelo="M2", info_adicional="3 indicadores mensuales rápidos",
             problema="Estado contemporáneo no observado (PIB/IGAE ciegos hasta feb-2020)",
             variables_nuevas="IMSS, ANTAD, AUTOS",
             metodo="Índice compuesto + ecuación puente MCO + Kalman univariado"),
        dict(modelo="M3", info_adicional="Panel amplio de 10 series + canal externo",
             problema="Muchos indicadores, no comparables 1:1 con el PIB",
             variables_nuevas="+ IGAE, ActividadIndustrial, FBCF, IMCP, Export., Import., INDPRO EEUU",
             metodo="PCA (factor dinámico) vs ElasticNetCV -- gana PCA en backtest"),
        dict(modelo="M4", info_adicional="Trayectorias cross-country de movilidad + stringency",
             problema="Trayectoria futura del choque desconocida",
             variables_nuevas="Movilidad y StringencyIndex de Corea, Italia, España, EE.UU.",
             metodo="Análogos ponderados por kernel + bootstrap + pronóstico condicional "
                    "(simplificación univariante de Waggoner-Zha 1999)"),
    ])
    escalera.to_csv(FIG_OUT / "tabla_escalera_M0_M4.csv", index=False)
    print(f"Guardado: {FIG_OUT / 'tabla_escalera_M0_M4.csv'}")


if __name__ == "__main__":
    main()
