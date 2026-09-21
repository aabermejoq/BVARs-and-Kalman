"""
09_M4_trajectory_ML.py

FASE 10: M4 -- trayectorias futuras de Z generadas por analogos
cross-country, ponderadas por kernel, propagadas a PIB sin reestimar el
modelo macro (en el espiritu de Waggoner & Zha 1999).

Z = movilidad promedio (retail+trabajo, Google) de Mexico. NO se entrena
sobre los 2 episodios historicos de Mexico (Tequila 94-95, GFC 08-09)
porque son choques de OTRA naturaleza (financiero/cambiario y de demanda
externa, respectivamente) y usarlos mezclaria episodios no comparables. En
su lugar se usan las trayectorias YA REALIZADAS de Corea del Sur, Italia,
Espana y Estados Unidos -- el MISMO tipo de choque (confinamiento COVID),
observado ANTES que en Mexico y por tanto informacion publica legitima al
15-may-2020 (no es el futuro de Mexico).

Metodo ("transplante de deltas por tiempo de evento"):
  1. mes_0 = primer mes en que la movilidad promedio cae por debajo de -15
     en cada pais (evento de referencia, no fecha calendario).
  2. Para cada pais s, se registran los DELTAS mes a mes despues de mes_0:
     d1_s = e1_s - e0_s,  d2_s = e2_s - e1_s,  d3_s = e3_s - e2_s (si no
     esta disponible por el corte, se extrapola con persistencia amortiguada
     del ultimo delta observado, rho=0.7 -- supuesto declarado).
  3. Escenario de Mexico bajo el analogo s: aplica esos MISMOS deltas
     encima del nivel observado de Mexico en sus propios mes_0/mes_1
     (que SI estan observados al corte).
  4. Ponderacion por kernel: w_s ~ exp(-0.5 d(X_mex,X_s)^2/h^2), con
     X = (StringencyIndex en mes_0, delta1 propio) estandarizado entre los
     4 analogos + Mexico. NO son pesos arbitrarios 1/3-1/3-1/3.
  5. Incertidumbre: bootstrap de residuos alrededor de cada trayectoria
     analogo (perturbacion) + mixtura ponderada por w_s entre analogos.
  6. Propagacion a PIB: se usa la MISMA ecuacion puente ya estimada en
     M2/M3 (parametros FIJOS, no reestimados) para traducir el escenario
     de movilidad -> indice compatible -> PIB. Esto es una version
     simplificada (univariante) del pronostico condicional de
     Waggoner-Zha (1999): se fijan los parametros estructurales estimados
     y se condiciona unicamente la trayectoria futura de una variable
     auxiliar, propagando su efecto sin reestimar el modelo. Una
     implementacion completa conditionaria un VAR multivariado que incluya
     la movilidad como variable propia; no fue posible aqui porque la
     movilidad tiene solo ~3 meses de historia en Mexico (insuficiente
     para estimar un VAR conjunto) -- se declara esta simplificacion
     explicitamente.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "output" / "models"

N_DRAWS = 10_000
RNG_SEED = 52345
THRESHOLD = -15.0
DAMPING_RHO = 0.7


def monthly_mobility_index(country_code):
    mob = pd.read_csv(EXTERNAL / "google_mobility_benchmarks.csv", parse_dates=["date"])
    c = mob[mob.country_code == country_code].set_index("date")
    idx = c[["workplaces_percent_change_from_baseline",
             "retail_and_recreation_percent_change_from_baseline"]].mean(axis=1)
    return idx.resample("MS").mean()


def event_aligned(monthly_idx, threshold=THRESHOLD):
    below = monthly_idx[monthly_idx < threshold]
    if below.empty:
        return None
    m0 = below.index.min()
    months = [m0 + pd.DateOffset(months=k) for k in range(4)]
    e = [monthly_idx.get(m, np.nan) for m in months]
    return dict(m0=m0, e=e)


def stringency_at(country_iso3, month0):
    ox = pd.read_csv(EXTERNAL / "oxcgrt_benchmarks.csv", parse_dates=["Date"])
    c = ox[ox.CountryCode == country_iso3].set_index("Date")["StringencyIndex_Average"]
    window = c.loc[month0: month0 + pd.DateOffset(months=1)]
    return float(window.mean()) if not window.empty else np.nan


ISO3 = {"MX": "MEX", "KR": "KOR", "IT": "ITA", "ES": "ESP", "US": "USA"}


def get_deltas_and_features(country_code):
    idx = monthly_mobility_index(country_code)
    ea = event_aligned(idx)
    e = ea["e"]
    deltas = []
    for k in range(3):
        if not np.isnan(e[k]) and not np.isnan(e[k + 1]):
            deltas.append(e[k + 1] - e[k])
        elif deltas:
            deltas.append(deltas[-1] * DAMPING_RHO)  # persistencia amortiguada
        else:
            deltas.append(0.0)
    stringency0 = stringency_at(ISO3[country_code], ea["m0"])
    return dict(country=country_code, m0=ea["m0"], e0=e[0], deltas=deltas, stringency0=stringency0)


def main():
    mx = get_deltas_and_features("MX")
    analogs = [get_deltas_and_features(c) for c in ["KR", "IT", "ES", "US"]]

    print(f"Mexico: mes_0={mx['m0'].date()}, e0={mx['e0']:.1f}, deltas={np.round(mx['deltas'],1)}, "
          f"stringency0={mx['stringency0']:.1f}")
    for a in analogs:
        print(f"  Analogo {a['country']}: mes_0={a['m0'].date()}, e0={a['e0']:.1f}, "
              f"deltas={np.round(a['deltas'],1)}, stringency0={a['stringency0']:.1f}")

    # --- Ponderacion por kernel (distancia en (stringency0, delta1)) ---
    feat_mx = np.array([mx["stringency0"], mx["deltas"][0]])
    feats = np.array([[a["stringency0"], a["deltas"][0]] for a in analogs])
    all_feats = np.vstack([feat_mx, feats])
    mu, sd = all_feats.mean(axis=0), all_feats.std(axis=0)
    z_mx = (feat_mx - mu) / sd
    z_an = (feats - mu) / sd
    dist = np.linalg.norm(z_an - z_mx, axis=1)
    bandwidth = max(dist.std(), 0.5)
    w = np.exp(-0.5 * (dist / bandwidth) ** 2)
    w = w / w.sum()

    print("\nPesos kernel (no arbitrarios, por similitud del choque con Mexico):")
    for a, wi, di in zip(analogs, w, dist):
        print(f"  {a['country']}: distancia={di:.2f}  peso={wi:.3f}")

    # --- Mexico: e0 (su propio mes_0, resulto ser abril-2020, ver output) ya
    # observado. Se necesita pronosticar e1 (mes_0+1) y e2 (mes_0+2) usando
    # los deltas EVENT-TIME (relativos al propio mes_0 de cada pais, no a la
    # fecha calendario) de los analogos. ---
    with open(OUTPUT / "M2_fit.pkl", "rb") as f:
        m2 = pickle.load(f)
    a_bridge, b_bridge = m2["bridge_coef"]
    sigma_bridge = m2["bridge_sigma"]
    m2_index = m2["index"]  # indice compuesto mensual de M2 (feb-1993 a abr-2020)

    mob_full = monthly_mobility_index("MX")
    mx_e0 = mx["e0"]
    mx_e1_obs = mob_full.get(mx["m0"] + pd.DateOffset(months=1))  # mayo, parcial (hasta 10-may)

    # Traduccion movilidad -> escala del indice M2: regresion simple con los
    # UNICOS meses donde ambas series se traslapan (feb,mar,abr-2020). N=3 es
    # muy pequeno -- se declara explicitamente esta limitacion; el objetivo
    # es solo obtener una escala razonable, no una relacion estructural.
    overlap = mob_full.index.intersection(m2_index.index[-3:])
    mob_ov = mob_full.reindex(overlap).values
    idx_ov = m2_index.reindex(overlap).values
    beta_mob, alpha_mob = np.polyfit(mob_ov, idx_ov, 1)
    print(f"\nEscala movilidad->indice M2 (N={len(overlap)}, solo para conversion de unidades): "
          f"indice = {alpha_mob:.3f} + {beta_mob:.4f} * movilidad")

    rng = np.random.default_rng(RNG_SEED)
    growth_draws = np.zeros(N_DRAWS)
    scenario_choice = rng.choice(len(analogs), size=N_DRAWS, p=w)

    # dispersion empirica de los deltas entre analogos, para el bootstrap
    delta0_pool = np.array([a["deltas"][0] for a in analogs])
    delta1_pool = np.array([a["deltas"][1] for a in analogs])
    boot_sd0, boot_sd1 = delta0_pool.std(ddof=1), delta1_pool.std(ddof=1)

    for d in range(N_DRAWS):
        s_idx = scenario_choice[d]
        a = analogs[s_idx]
        mx_e1 = mx_e0 + a["deltas"][0] + rng.normal(0, boot_sd0)  # pronostico de mayo
        mx_e2 = mx_e1 + a["deltas"][1] + rng.normal(0, boot_sd1)  # pronostico de junio

        # mayo-2020 esta parcialmente observado (hasta 10-may): se mezcla el
        # dato real parcial con el escenario, como aproximacion simple de
        # que el escenario cubre el resto del mes.
        if mx_e1_obs is not None and not np.isnan(mx_e1_obs):
            mx_e1 = 0.34 * mx_e1_obs + 0.66 * mx_e1

        idx_may = alpha_mob + beta_mob * mx_e1
        idx_jun = alpha_mob + beta_mob * mx_e2

        # combinar con el ultimo estado filtrado de M2 (abril) para el trimestre completo
        idx_q2 = (m2["summary"]["x_abril_2020"] + idx_may + idx_jun) / 3.0
        growth_draws[d] = a_bridge + b_bridge * idx_q2 + rng.normal(0, sigma_bridge)

    summary = dict(
        modelo="M4", pesos={a["country"]: float(wi) for a, wi in zip(analogs, w)},
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)), p10=float(np.percentile(growth_draws, 10)),
        p25=float(np.percentile(growth_draws, 25)), p75=float(np.percentile(growth_draws, 75)),
        p90=float(np.percentile(growth_draws, 90)), p97_5=float(np.percentile(growth_draws, 97.5)),
    )
    print("\n=== M4: analogos cross-country ponderados + puente, pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M4_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M4_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, mx=mx, analogs=analogs, weights=w), f)
    print(f"\nGuardado: {OUTPUT / 'M4_growth_draws.npy'} y {OUTPUT / 'M4_fit.pkl'}")


if __name__ == "__main__":
    main()
