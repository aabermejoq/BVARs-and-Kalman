"""
14_M4_agosto_bayesian_scenarios.py

M4 para el ejercicio de agosto/3T20: filtro bayesiano secuencial sobre los
2 escenarios del FMI (WEO abril-2020), actualizado dia a dia con la base
DIARIA de tarjetas de Banxico (supuesto: rezago de 1 dia, series originales,
ver 12_tarjetas_daily_audit.py).

PASO 1 -- Reconciliar los escenarios anuales del FMI con lo YA CONOCIDO:
  FMI (fuente: El Universal, citando WEO abril-2020): Mexico 2020 = -6.6%
  (escenario base). El FMI NO publico una cifra especifica para Mexico en
  su escenario adverso ("prolonged outbreak"); ese escenario si trae una
  cifra GLOBAL: un choque adicional de ~3pp de contraccion en 2020. Se
  extiende ese mismo delta a Mexico como SUPUESTO DECLARADO (no una cifra
  oficial del FMI para Mexico): escenario adverso Mexico 2020 = -9.6%.

  Al 15-ago-2020 ya se conocen 1T20 y 2T20 (el PIB oportuno de 2T20 se
  publica ~1-ago-2020). Se resuelve, para cada escenario, la tasa de
  crecimiento trimestral CONSTANTE g en 3T20-4T20 (partiendo del nivel YA
  CONOCIDO de 2T20) que hace que el promedio anual de 2020 alcance la cifra
  del escenario. Esto traduce cada escenario ANUAL del FMI en una
  trayectoria trimestral IMPLICITA para 3T20, dado lo que ya paso.

PASO 2 -- Traducir esa trayectoria a una escala comparable con tarjetas:
  Regresion historica (2009T1-2019T4, pre-COVID) de tarjetas_agregado(a/a)
  sobre PIB(t/t %) para obtener la escala de traduccion.

PASO 3 -- Filtro bayesiano secuencial dia a dia (jul-14ago-2020):
  Se compara la realizacion diaria de tarjetas contra la trayectoria
  ("glide path") implicita de cada escenario, y se actualiza P(escenario)
  por la regla de Bayes, con una pequena probabilidad de transicion (5%)
  para evitar que el filtro colapse de forma numericamente absoluta a 0/1
  -- equivalente a un filtro de particulas de 2 particulas fijas (una por
  escenario), donde solo se actualiza el peso.

PASO 4 -- Combinar g_base y g_adverso, ponderados por las probabilidades
  finales, mas dispersion bootstrap, para la distribucion predictiva final
  de M4 para 3T20.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models_agosto"

ASOF = pd.Timestamp("2020-08-15")
LAG_DAYS = 1
N_DRAWS = 10_000

FMI_BASE_2020 = -0.066
FMI_ADVERSO_2020 = -0.096  # -6.6% - 3pp (extrapolacion del delta GLOBAL del FMI, no cifra oficial para Mexico)


def implied_q3q4_growth(target_annual, pib_2019, pib_q1_2020, pib_q2_2020):
    """Resuelve g (tasa t/t constante en 3T20 y 4T20) tal que el promedio
    anual 2020 (respecto a 2019) alcance target_annual, dado 1T20 y 2T20
    YA CONOCIDOS."""
    sum_2019 = sum(pib_2019)
    # (q1+q2+q2*(1+g)+q2*(1+g)^2) / sum_2019 - 1 = target_annual
    target_sum = (1 + target_annual) * sum_2019
    # resolver cuadratica en u=(1+g): q2*u^2 + q2*u + (q1+q2-target_sum) = 0
    a = pib_q2_2020
    b = pib_q2_2020
    c = pib_q1_2020 + pib_q2_2020 - target_sum
    disc = b ** 2 - 4 * a * c
    u = (-b + np.sqrt(disc)) / (2 * a)
    g = u - 1
    q3 = pib_q2_2020 * u
    return g, q3


def build_card_bridge():
    yoy = pd.read_csv(INTERIM / "tarjetas_daily_yoy.csv", index_col=0, parse_dates=True)
    agregado = yoy["AGREGADO"].replace([np.inf, -np.inf], np.nan)
    agregado_q = agregado.resample("QS").mean()

    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib_full = series["PIB"]["PIB"]
    pib_qoq_full = 100 * np.log(pib_full / pib_full.shift(1))

    df = pd.concat([pib_qoq_full.rename("pib"), agregado_q.rename("tarjetas")], axis=1, sort=True).dropna()
    df_pre = df.loc[:"2019-10-01"]
    X = np.column_stack([np.ones(len(df_pre)), df_pre["pib"].values])
    y = df_pre["tarjetas"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = resid.std(ddof=2)
    print(f"Puente tarjetas~PIB (n={len(df_pre)}, 2009T1-2019T4): "
          f"tarjetas_a/a = {coef[0]:.2f} + {coef[1]:.3f}*PIB_qoq(%)  (sigma={sigma:.2f})")
    return coef, sigma


def main():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    pib = series["PIB"]["PIB"]
    pib_2019 = [pib.loc[f"2019-{m:02d}-01"] for m in [1, 4, 7, 10]]
    pib_q1_2020 = pib.loc["2020-01-01"]
    pib_q2_2020 = pib.loc["2020-04-01"]

    g_base, q3_base = implied_q3q4_growth(FMI_BASE_2020, pib_2019, pib_q1_2020, pib_q2_2020)
    g_adv, q3_adv = implied_q3q4_growth(FMI_ADVERSO_2020, pib_2019, pib_q1_2020, pib_q2_2020)
    print(f"Escenario BASE FMI (-6.6% anual 2020): implica 3T20 t/t = {g_base*100:.2f}%")
    print(f"Escenario ADVERSO FMI (-9.6% anual 2020, extrapolado): implica 3T20 t/t = {g_adv*100:.2f}%")

    coef_card, sigma_card = build_card_bridge()

    target_base = coef_card[0] + coef_card[1] * (g_base * 100)
    target_adv = coef_card[0] + coef_card[1] * (g_adv * 100)
    print(f"\nNivel de tarjetas (a/a) IMPLICITO por escenario al final de 3T20:")
    print(f"  Base:    {target_base:.2f}%")
    print(f"  Adverso: {target_adv:.2f}%")

    # --- Nivel de partida: tarjetas observado a fin de junio-2020 ---
    yoy = pd.read_csv(INTERIM / "tarjetas_daily_yoy.csv", index_col=0, parse_dates=True)
    agregado = yoy["AGREGADO"].replace([np.inf, -np.inf], np.nan)
    start_level = agregado.loc[:"2020-06-30"].iloc[-1]
    print(f"Nivel de partida (tarjetas a/a, 30-jun-2020): {start_level:.2f}%")

    # --- Filtro bayesiano secuencial dia a dia (1-jul a 14-ago-2020) ---
    last_allowed = ASOF - pd.Timedelta(days=LAG_DAYS)
    window = agregado.loc["2020-07-01":last_allowed]
    n_days_q3 = (pd.Timestamp("2020-09-30") - pd.Timestamp("2020-07-01")).days

    def glide(target, day_idx):
        frac = day_idx / n_days_q3
        return start_level + frac * (target - start_level)

    p_base, p_adv = 0.5, 0.5  # prior no informativo
    trans = 0.05  # probabilidad de transicion (evita colapso absoluto 0/1)
    history = []
    for i, (dte, val) in enumerate(window.items()):
        if np.isnan(val):
            continue
        day_idx = (dte - pd.Timestamp("2020-07-01")).days
        mean_base = glide(target_base, day_idx)
        mean_adv = glide(target_adv, day_idx)
        lik_base = norm.pdf(val, loc=mean_base, scale=sigma_card)
        lik_adv = norm.pdf(val, loc=mean_adv, scale=sigma_card)

        # prediccion (mezcla con probabilidad de transicion) + actualizacion bayesiana
        p_base_pred = p_base * (1 - trans) + p_adv * trans
        p_adv_pred = p_adv * (1 - trans) + p_base * trans
        num_base = lik_base * p_base_pred
        num_adv = lik_adv * p_adv_pred
        denom = num_base + num_adv
        p_base, p_adv = num_base / denom, num_adv / denom
        history.append(dict(fecha=dte, tarjetas=val, mean_base=mean_base, mean_adv=mean_adv,
                             p_base=p_base, p_adv=p_adv))

    hist_df = pd.DataFrame(history)
    print(f"\nEvolucion de P(escenario base) -- primeros y ultimos 5 dias:")
    print(hist_df[["fecha", "tarjetas", "p_base"]].head(5).to_string(index=False))
    print("...")
    print(hist_df[["fecha", "tarjetas", "p_base"]].tail(5).to_string(index=False))

    w_base, w_adv = p_base, p_adv
    print(f"\nPesos finales al 14-ago-2020: P(base)={w_base:.3f}  P(adverso)={w_adv:.3f}")

    # --- Mezcla final: g_base y g_adverso ponderados + dispersion bootstrap ---
    rng = np.random.default_rng(505)
    scenario_choice = rng.choice([0, 1], size=N_DRAWS, p=[w_base, w_adv])
    g_scenarios = np.array([g_base, g_adv]) * 100
    # dispersion: usar el sigma del puente tarjetas (trasladado a unidades de PIB via 1/pendiente)
    sigma_g = sigma_card / abs(coef_card[1])
    growth_draws = g_scenarios[scenario_choice] + rng.normal(0, sigma_g, N_DRAWS)

    summary = dict(
        modelo="M4_agosto", g_base_pct=float(g_base * 100), g_adverso_pct=float(g_adv * 100),
        w_base=float(w_base), w_adverso=float(w_adv),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)), p97_5=float(np.percentile(growth_draws, 97.5)),
        p10=float(np.percentile(growth_draws, 10)), p90=float(np.percentile(growth_draws, 90)),
    )
    print("\n=== M4-agosto: pronostico 3T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M4_growth_draws.npy", growth_draws)
    hist_df.to_csv(OUTPUT / "M4_bayesian_filter_history.csv", index=False)
    with open(OUTPUT / "M4_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, hist_df=hist_df, coef_card=coef_card, sigma_card=sigma_card), f)
    print(f"\nGuardado: {OUTPUT / 'M4_growth_draws.npy'}, {OUTPUT / 'M4_bayesian_filter_history.csv'}")


if __name__ == "__main__":
    main()
