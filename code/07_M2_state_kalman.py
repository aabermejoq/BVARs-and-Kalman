"""
07_M2_state_kalman.py

FASE 8: M2 resuelve P2 (el estado contemporaneo de la economia no se
observa directamente en mayo-2020).

NOTA METODOLOGICA IMPORTANTE (transparencia sobre una decision de diseno):
un primer intento estimo conjuntamente por maxima verosimilitud un modelo
de espacio-estado con 9 parametros (factor + 3 cargas + 4 varianzas
idiosincraticas + varianza de estado), tal como en un Dynamic Factor Model
tradicional. Ese modelo resulto NUMERICAMENTE FRAGIL: pequenos cambios en
las cotas de optimizacion producian estimaciones completamente distintas
(una vez, R_ANTAD colapso a ~0; otra vez, Q y R_IMSS colapsaron al limite
inferior), sintoma de identificacion debil con solo 3-4 series y un factor
casi de raiz unitaria. Reportar un resultado de un modelo numericamente
inestable como si fuera solido violaria el principio de "no inventar
resultados". Por eso se usa una estrategia de DOS PASOS, mas simple y
robusta, estandar en la practica de nowcasting cuando el numero de
indicadores es pequeno:

  PASO 1 (medicion del estado): indice compuesto = promedio simple de las
  variaciones interanuales ESTANDARIZADAS (z-score, con media/sd calculadas
  SOLO con datos hasta dic-2019, para que la escala no la distorsione la
  propia crisis) de IMSS, ANTAD y AUTOS. Es la misma logica que un indice
  tipo "Conference Board" de indicadores coincidentes: pesos iguales,
  transparente, sin parametros que puedan degenerar.

  PASO 2 (ecuacion puente): PIB_trimestral(%) = a + b * indice_trimestral
  + u,  estimado por MCO con toda la muestra real-time disponible
  (1997T3-2020T1). Traduce el indice a unidades de crecimiento del PIB.

  PASO 3 (dinamica y pronostico): filtro de Kalman UNIVARIADO (solo 3
  parametros: phi, Q, R -- bien identificado) sobre el indice compuesto
  mensual: x_t = phi*x_{t-1}+w_t, indice_obs_t = x_t + v_t. Se filtra hasta
  abril-2020 (ultimo mes real-time disponible) y se pronostica mayo y junio
  con la dinamica AR(1), propagando incertidumbre por Monte Carlo.

  El pronostico final de PIB 2T20 combina el paso 3 (incertidumbre del
  indice trimestral abr-may-jun) con el paso 2 (incertidumbre de la
  ecuacion puente, incluyendo el error de estimacion de a,b) via Monte
  Carlo conjunto.

Variaciones interanuales (a/a, no m/m) para IMSS/ANTAD/AUTOS porque estas
series no traen metadato de ajuste estacional en el archivo (Fase 1).

Movilidad de Mexico: solo ~3 meses de historia al corte, insuficiente para
participar en la construccion del indice (que requiere una serie larga
para fijar phi/Q/R con confianza). Se reporta unicamente como
corroboracion cualitativa fuera de muestra (no entra al indice ni a la
ecuacion puente).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "output" / "models"

N_DRAWS = 10_000
RNG_SEED = 32345
STANDARDIZE_CUTOFF = pd.Timestamp("2019-12-01")


def build_monthly_indicators():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    imss = series["IMSS"]["IMSS_empleos"].dropna()
    antad = series["Consumo"]["ANTAD"].dropna()
    autos = series["Consumo"]["AUTOS"].dropna()

    idx = pd.date_range("1997-07-01", "2020-04-01", freq="MS")
    yoy = pd.DataFrame(index=idx)
    yoy["IMSS"] = (100 * np.log(imss / imss.shift(12))).reindex(idx)
    yoy["ANTAD"] = (100 * np.log(antad / antad.shift(12))).reindex(idx)
    yoy["AUTOS"] = (100 * np.log(autos / autos.shift(12))).reindex(idx)
    return yoy


def build_composite_index(yoy):
    pre = yoy.loc[:STANDARDIZE_CUTOFF]
    mu, sd = pre.mean(), pre.std()
    z = (yoy - mu) / sd
    index = z.mean(axis=1, skipna=True)
    return index, mu, sd


def kalman_uni_negloglik(params, y):
    phi, logQ, logR = params
    Q, R = np.exp(logQ), np.exp(logR)
    x, P = 0.0, 100.0
    ll = 0.0
    for yt in y:
        x_pred = phi * x
        P_pred = phi * phi * P + Q
        innov = yt - x_pred
        S = P_pred + R
        K = P_pred / S
        x = x_pred + K * innov
        P = (1 - K) * P_pred
        ll += -0.5 * (np.log(2 * np.pi) + np.log(S) + innov * innov / S)
    return -ll


def kalman_uni_filter(params, y):
    phi, logQ, logR = params
    Q, R = np.exp(logQ), np.exp(logR)
    x, P = 0.0, 100.0
    xs, Ps = [], []
    for yt in y:
        x_pred = phi * x
        P_pred = phi * phi * P + Q
        innov = yt - x_pred
        S = P_pred + R
        K = P_pred / S
        x = x_pred + K * innov
        P = (1 - K) * P_pred
        xs.append(x)
        Ps.append(P)
    return np.array(xs), np.array(Ps)


def main():
    yoy = build_monthly_indicators()
    index, mu, sd = build_composite_index(yoy)
    print(f"Indice compuesto (promedio de z-scores IMSS/ANTAD/AUTOS): "
          f"{index.index.min().date()} -> {index.index.max().date()}")
    print("Ultimos 6 meses del indice:")
    print(index.tail(6).round(2).to_string())

    # --- Ecuacion puente (PASO 2): PIB_q(%) = a + b*indice_trimestral ---
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)

    idx_q = index.resample("QS").mean()
    idx_q_count = index.resample("QS").count()
    idx_q = idx_q[idx_q_count == 3]  # solo trimestres con los 3 meses completos

    bridge_df = pd.concat([pib_qoq.rename("pib"), idx_q.rename("idx")], axis=1).dropna()
    X_b = np.column_stack([np.ones(len(bridge_df)), bridge_df["idx"].values])
    y_b = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(X_b, y_b, rcond=None)
    a_b, b_b = coef_b
    resid_b = y_b - X_b @ coef_b
    sigma_b = resid_b.std(ddof=2)
    XtX_inv = np.linalg.inv(X_b.T @ X_b)

    print(f"\nEcuacion puente (MCO, n={len(bridge_df)} trimestres 1997T3-2020T1):")
    print(f"  PIB_q(%) = {a_b:.3f} + {b_b:.3f} * indice_trimestral   (sigma_resid={sigma_b:.3f})")

    # --- Kalman univariado sobre el indice (PASO 3) ---
    y = index.values
    x0 = np.array([0.5, np.log(1.0), np.log(1.0)])
    bounds = [(-0.99, 0.99), (np.log(0.01), np.log(50)), (np.log(0.01), np.log(50))]
    res = minimize(kalman_uni_negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds)
    phi, logQ, logR = res.x
    Q, R = np.exp(logQ), np.exp(logR)
    print(f"\nKalman univariado sobre el indice: exito={res.success}")
    print(f"  phi={phi:.3f}  Q={Q:.3f}  R={R:.3f}")

    xs, Ps = kalman_uni_filter(res.x, y)
    print("\nIndice FILTRADO (estado latente), ultimos 6 meses:")
    for d, xv in list(zip(index.index, xs))[-6:]:
        print(f"  {d.date()}: {xv:+.2f}")

    x_abr, P_abr = xs[-1], Ps[-1]

    # --- Movilidad: corroboracion cualitativa fuera de muestra (NO se usa en el indice) ---
    mob = pd.read_csv(EXTERNAL / "google_mobility_benchmarks.csv", parse_dates=["date"])
    mx_mob = mob[mob.country_code == "MX"].set_index("date")
    mx_mob_m = mx_mob[["workplaces_percent_change_from_baseline",
                        "retail_and_recreation_percent_change_from_baseline"]].resample("MS").mean().mean(axis=1)
    print(f"\n[Corroboracion cualitativa, NO usada en la estimacion] Movilidad promedio MX "
          f"feb-abr 2020: {mx_mob_m.loc['2020-02':'2020-04'].round(1).to_dict()}")
    print(f"[Corroboracion] Indice compuesto en los mismos meses: "
          f"{index.loc['2020-02':'2020-04'].round(2).to_dict()}")
    print("(Ambas series caen abruptamente en el mismo periodo -- consistencia cualitativa, "
          "no se usa para recalibrar el indice por el tamano insuficiente de muestra de movilidad.)")

    # --- Monte Carlo conjunto: incertidumbre de estado (paso 3) + puente (paso 2) ---
    rng = np.random.default_rng(RNG_SEED)
    growth_draws = np.zeros(N_DRAWS)
    idx_draws_record = np.zeros(N_DRAWS)

    for d in range(N_DRAWS):
        x_abr_d = rng.normal(x_abr, np.sqrt(max(P_abr, 0)))
        x_may_d = phi * x_abr_d + np.sqrt(Q) * rng.standard_normal()
        x_jun_d = phi * x_may_d + np.sqrt(Q) * rng.standard_normal()
        idx_q2 = (x_abr_d + x_may_d + x_jun_d) / 3.0
        idx_draws_record[d] = idx_q2

        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * XtX_inv)
        eps_d = rng.normal(0, sigma_b)
        growth_draws[d] = coef_d[0] + coef_d[1] * idx_q2 + eps_d

    summary = dict(
        modelo="M2",
        phi=float(phi), Q=float(Q), R=float(R),
        a_bridge=float(a_b), b_bridge=float(b_b), sigma_bridge=float(sigma_b),
        x_abril_2020=float(x_abr),
        indice_2T20_mediana=float(np.median(idx_draws_record)),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)),
        p10=float(np.percentile(growth_draws, 10)),
        p25=float(np.percentile(growth_draws, 25)),
        p75=float(np.percentile(growth_draws, 75)),
        p90=float(np.percentile(growth_draws, 90)),
        p97_5=float(np.percentile(growth_draws, 97.5)),
    )

    print("\n=== M2: indice compuesto + ecuacion puente + Kalman univariado, pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M2_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M2_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, kalman_params=res.x, bridge_coef=coef_b,
                          bridge_sigma=sigma_b, index=index, mu=mu, sd=sd), f)
    print(f"\nGuardado: {OUTPUT / 'M2_growth_draws.npy'} y {OUTPUT / 'M2_fit.pkl'}")


if __name__ == "__main__":
    main()
