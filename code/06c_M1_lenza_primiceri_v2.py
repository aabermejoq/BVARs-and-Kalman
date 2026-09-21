"""
06c_M1_lenza_primiceri_v2.py

M1 re-auditado tras leer el paper original (Lenza & Primiceri, NBER WP
27771, publicado SEPTIEMBRE-2020 -- por tanto sus VALORES ESTIMADOS
(s0~17, s1~70, s2~20 para EEUU) NO pueden usarse como informacion real-time
en mayo NI en agosto (ambos cortes son anteriores a la publicacion). Lo que
si es legitimo usar es su METODOLOGIA (es una herramienta econometrica,
no un dato observado).

CORRECCIONES frente a 06b_M1_lenza_primiceri.py:

1. CONTAMINACION (bug real, encontrado al re-leer el paper): en la version
   anterior, el M1 de AGOSTO reutilizaba beta y Sigma_hat de un M0
   reestimado con la muestra COMPLETA hasta 2T20 -- es decir, el propio
   choque extremo de 2T20 ya habia distorsionado beta y Sigma_hat ANTES de
   aplicarles el factor de escala adicional, doble-contando el choque. El
   paper es explicito en que esto es exactamente lo que hay que evitar
   ("parameter estimates can be substantially influenced, and not
   necessarily in a good way"). Aqui beta y Sigma se estiman UNA SOLA VEZ,
   con datos SOLO hasta 4T19 (limpios, sin ningun trimestre de COVID), y
   se reutilizan identicos para mayo y para agosto.

2. INDEXACION DE PERIODOS ESPECIALES: el paper reserva parametros de
   escala LIBRES e independientes (no decaidos) para los primeros 3
   periodos especiales, y el decaimiento geometrico solo empieza en el
   CUARTO periodo en adelante. Dado que 1T20 no muestra senal anormal en
   Mexico (ver 06_M1_volatility.py: el choque de marzo se diluye en el
   promedio trimestral), se trata a 1T20 como un trimestre NORMAL, y se
   redefine t* (inicio de la variacion anormal) = 2T20. Bajo este
   esquema:
     - s0 = escala de 2T20 (el PRIMER periodo especial)
     - s1 = escala de 3T20 (el SEGUNDO periodo especial -- todavia un
       parametro LIBRE segun el paper, no un valor decaido de s0)

3. PRIOR PARA PERIODOS NO OBSERVADOS: en MAYO, s0 (2T20) no se puede medir
   (el trimestre aun no ocurre) -- se ancla con el PIB 1T20 de Italia
   (-4.7%, Istat, publicado 30-abr-2020, real-time-disponible). En AGOSTO,
   s0 (2T20) SI se puede medir del residuo observado (contra el modelo
   LIMPIO). Para s1 (3T20, tampoco observado en agosto), se usa la
   metodologia (no los valores) del paper: su prior Beta(moda=0.8, sd=0.2)
   para la tasa de decaimiento MENSUAL se traduce a una tasa TRIMESTRAL
   equivalente (0.8^3=0.512) y se aplica como un unico paso de decaimiento
   desde s0.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"
OUTPUT_AGOSTO = ROOT / "output" / "models_agosto"

ITALIA_Q1_2020_SHOCK_PCT = 4.7  # Istat, publicado 30-abr-2020 (real-time-disponible en mayo Y en agosto)
RHO_MONTHLY = 0.8  # centro del prior Beta del paper (decaimiento MENSUAL, metodologia, no dato)
RHO_QUARTERLY = RHO_MONTHLY ** 3  # traduccion a un paso trimestral


def clean_fit():
    """beta y Sigma estimados UNA SOLA VEZ, solo con datos hasta 4T19
    (limpios de cualquier trimestre afectado por COVID)."""
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from importlib import import_module
    m0mod = import_module("05_M0_BVAR")

    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    core_clean = core.loc[:"2019-10-01"]
    print(f"Muestra LIMPIA (sin ningun trimestre de COVID): "
          f"{core_clean.index.min().date()} -> {core_clean.index.max().date()} (n={len(core_clean)})")
    data_clean = core_clean[m0mod.VARS].values
    fit = m0mod.fit_minnesota_bvar(data_clean, m0mod.P)
    normal_sd = 100 * np.sqrt(fit["Sigma_hat"][0, 0])
    print(f"SD normal trimestral de log_PIB (modelo limpio): {normal_sd:.3f}%")
    return fit, m0mod, normal_sd, core


def realized_residual_pct(fit, m0mod, core, target_quarter, lag_quarter_data):
    """Calcula el residuo (en % de crecimiento) del trimestre `target_quarter`
    bajo el modelo LIMPIO, usando como insumo los rezagos REALES ya
    observados (lag_quarter_data: array (P,n) con las P observaciones mas
    recientes ANTES de target_quarter, orden mas reciente primero)."""
    VARS, P = m0mod.VARS, m0mod.P
    beta_mean = fit["post_mean"]  # media posterior (para el residuo puntual)
    x = np.concatenate([lag_quarter_data[l] for l in range(P)] + [[1.0]])
    y_pred = x @ beta_mean
    actual = core.loc[target_quarter, VARS].values
    resid = actual - y_pred
    return resid, y_pred


def simulate(fit, m0mod, last_obs, log_pib_t, scale, n_draws=10_000, seed=920):
    VARS, P = m0mod.VARS, m0mod.P
    n = len(VARS)
    rng = np.random.default_rng(seed)
    Sigma_hat = fit["Sigma_hat"]
    L_hat = np.linalg.cholesky(Sigma_hat + 1e-10 * np.eye(n))
    growth = np.zeros(n_draws)
    for d in range(n_draws):
        beta_d = np.zeros((n * P + 1, n))
        for i in range(n):
            beta_d[:, i] = rng.multivariate_normal(fit["post_mean"][:, i], fit["post_var"][:, :, i])
        x = np.concatenate([last_obs[l] for l in range(P)] + [[1.0]])
        eps = scale * (L_hat @ rng.standard_normal(n))
        y_new = x @ beta_d + eps
        growth[d] = 100 * (np.exp(y_new[VARS.index("log_PIB")] - log_pib_t) - 1)
    return growth


def main():
    fit, m0mod, normal_sd, core = clean_fit()
    VARS, P = m0mod.VARS, m0mod.P

    # === MAYO: pronostico de 2T20 = t* (primer periodo especial, s0 no observable) ===
    s0_may = ITALIA_Q1_2020_SHOCK_PCT / normal_sd  # multiplicador de DESV. ESTANDAR (consistente con s_t del paper)
    print(f"\nMAYO: s0 (ancla Italia, prior para 2T20) = {s0_may:.2f}x (desv. estandar)")

    core_hasta_1t20 = core.loc[:"2020-01-01"]
    data_may = core_hasta_1t20[VARS].values
    last_obs_may = data_may[-P:][::-1]
    log_pib_1t20 = data_may[-1, VARS.index("log_PIB")]

    growth_may = simulate(fit, m0mod, last_obs_may, log_pib_1t20, s0_may, seed=920)
    print(f"M1-LPv2 mayo: mediana={np.median(growth_may):.2f}%  "
          f"IC95%=[{np.percentile(growth_may,2.5):.2f}, {np.percentile(growth_may,97.5):.2f}]")
    actual_2t20 = -20.96
    print(f"  Cubre observado ({actual_2t20}%)? "
          f"{np.percentile(growth_may,2.5)<=actual_2t20<=np.percentile(growth_may,97.5)}")

    # === AGOSTO: 2T20 ya observado (s0 medible); pronostico 3T20 = s1 (fresco, con prior de decaimiento) ===
    # El panel `core` (vintage de mayo) NO incluye 2T20 (correctamente truncado
    # al corte real-time de mayo); se usa el panel CORE de agosto (que si
    # incluye a 2T20, ya conocido a ese corte) para el residuo observado.
    with open(OUTPUT_AGOSTO / "core_info.pkl", "rb") as f:
        info_agosto = pickle.load(f)
    core_agosto_full = info_agosto["core"]
    core_hasta_2t20 = core_agosto_full.loc[:"2020-04-01"]
    lag_para_2t20 = data_may[-P:][::-1]  # las P obs. anteriores a 2T20 (mismas que arriba)
    resid_2t20, pred_2t20 = realized_residual_pct(fit, m0mod, core_hasta_2t20, "2020-04-01", lag_para_2t20)
    # residuo en la ecuacion de log_PIB, convertido a % de forma aproximada (residuo pequeno -> log~%)
    resid_pib_pct = 100 * resid_2t20[VARS.index("log_PIB")]
    s0_agosto = abs(resid_pib_pct) / normal_sd
    print(f"\nAGOSTO: residuo de 2T20 bajo el modelo LIMPIO = {resid_pib_pct:.2f} pp "
          f"(vs. prediccion {100*pred_2t20[VARS.index('log_PIB')]:.2f}%)")
    print(f"s0_agosto (medido, 2T20) = {s0_agosto:.2f}x (desv. estandar)")

    s1_agosto = 1 + (s0_agosto - 1) * RHO_QUARTERLY
    print(f"rho trimestral (0.8 mensual del paper, metodologia)^3 = {RHO_QUARTERLY:.3f}")
    print(f"s1_agosto (3T20, decaido desde s0) = {s1_agosto:.2f}x (desv. estandar)")

    data_agosto = core_hasta_2t20[VARS].values
    last_obs_agosto = data_agosto[-P:][::-1]
    log_pib_2t20 = data_agosto[-1, VARS.index("log_PIB")]

    growth_agosto = simulate(fit, m0mod, last_obs_agosto, log_pib_2t20, s1_agosto, seed=921)
    print(f"M1-LPv2 agosto: mediana={np.median(growth_agosto):.2f}%  "
          f"IC95%=[{np.percentile(growth_agosto,2.5):.2f}, {np.percentile(growth_agosto,97.5):.2f}]")
    actual_3t20 = 14.47
    print(f"  Cubre observado ({actual_3t20}%)? "
          f"{np.percentile(growth_agosto,2.5)<=actual_3t20<=np.percentile(growth_agosto,97.5)}")

    print("\n=== Comparacion con versiones anteriores de M1 ===")
    for label, path in [("M1 original (AR1 c/reversion) mayo", OUTPUT / "M1_growth_draws.npy"),
                         ("M1-LPv1 (Italia, contaminado en agosto) mayo", OUTPUT / "M1_lenza_primiceri_growth_draws.npy")]:
        d = np.load(path)
        print(f"  {label}: mediana={np.median(d):.2f}  IC95=[{np.percentile(d,2.5):.2f}, {np.percentile(d,97.5):.2f}]")
    for label, path in [("M1 original (AR1 c/reversion) agosto", OUTPUT_AGOSTO / "M1_growth_draws.npy"),
                         ("M1-LPv1 (contaminado) agosto", OUTPUT_AGOSTO / "M1_lenza_primiceri_growth_draws.npy")]:
        d = np.load(path)
        print(f"  {label}: mediana={np.median(d):.2f}  IC95=[{np.percentile(d,2.5):.2f}, {np.percentile(d,97.5):.2f}]")

    np.save(OUTPUT / "M1_lenza_primiceri_v2_growth_draws.npy", growth_may)
    np.save(OUTPUT_AGOSTO / "M1_lenza_primiceri_v2_growth_draws.npy", growth_agosto)
    with open(OUTPUT / "M1_lenza_primiceri_v2_params.pkl", "wb") as f:
        pickle.dump(dict(s0_may=s0_may, s0_agosto=s0_agosto, s1_agosto=s1_agosto,
                          rho_quarterly=RHO_QUARTERLY, normal_sd=normal_sd,
                          resid_2t20_pct=resid_pib_pct), f)
    print(f"\nGuardado: M1_lenza_primiceri_v2_growth_draws.npy (mayo y agosto)")


if __name__ == "__main__":
    main()
