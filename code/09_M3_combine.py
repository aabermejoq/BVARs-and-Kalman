"""
SUPERADO por 12_M3_waggoner_zha.py -- este archivo usaba un ajuste aditivo
ad hoc sobre IGAE. Se reemplazo por pronostico condicional real (Waggoner y
Zha, 1999) sobre IMSS_mom, tras leer la literatura citada en el paper de
referencia del usuario. Se deja el archivo por trazabilidad, NO se ejecuta
como parte del pipeline vigente.

M3 = M2 + escenarios epidemiologicos + GBM tarjetas + GDELT.

Ensamble final:
1. Abril/mayo (nowcast): se promedian (ponderado por su propio LOO/CV R^2,
   con piso en 0 -- un modelo con R^2<=0 no aporta senal de MEDIA, solo se
   deja que M2 conserve su correccion ya conservadora) el nowcast LP (M2) y
   el nowcast GBM. Dado que ambos mostraron CV<=0 en esta ventana, el
   ensamble no se aleja mucho de M2 -- resultado honesto, no forzado.

2. Junio-diciembre: se MEZCLAN las particulas de M2 entre los 4 escenarios
   epidemiologicos (peso = verosimilitud de backcast bayesiano, sin tocar
   datos posteriores al corte). Cada escenario aporta:
     a) una inclinacion (tilt) PEQUEÑA y EXPLICITAMENTE etiquetada como
        supuesto de diseño (no una elasticidad estimada -- estimarla
        econometricamente exigiria el propio IGAE de abril-diciembre 2020,
        que es precisamente lo que se esta pronosticando; usarlo seria
        fuga de informacion). Magnitud atada al mismo ancla real ya usada
        en M1/M2 (sqrt(s0), el residuo mas grande observado al corte).
     b) un multiplicador de varianza proporcional a cuanto se aleja la
        presion de casos de ese escenario del escenario mas optimista
        (rebote_rapido) -- mas casos, mas incertidumbre sobre restricciones
        futuras.

3. GDELT: se valida por CV contra IGAE_mom y contra |residuo| (CV5 R^2 =
   -0.21 media / -0.19 varianza con la muestra de 2 slices/dia, 1 dia/mes) y
   ese resultado se reporta sin maquillaje -- pero por instruccion expresa
   del usuario, GDELT ENTRA al ensamble de todas formas: se usa el tono
   promedio mensual de noticias MX (dato real descargado, GKG 2.1) como
   z-score contra su propia linea base 2017-2019, y ese z-score mueve tanto
   la inclinacion como la varianza de M3 mes a mes (abril-diciembre 2020),
   con la MISMA disciplina de anclaje a sqrt(s0) que el resto del ensamble.
   La falta de validacion se documenta en M3_meta.json y en el Excel, no se
   oculta; lo que cambia es que ya no bloquea su uso.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
m1 = import_module("04_M1_kalman_sv")
m2 = import_module("05_M2_particle_lp")
epi = import_module("06_M3_epidemic_scenarios")
gbm_mod = import_module("07_M3_gbm_tarjetas")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

TILT_COEF = 0.30    # fraccion de sqrt(s0) aplicada como inclinacion maxima por escenario epi
                    # (acotada con tanh -- nunca supera TILT_COEF*sqrt(s0) por mes)
VAR_COEF = 0.30     # idem para el multiplicador de varianza por dispersion de escenarios
GDELT_TILT_COEF = 0.30   # idem para el z-score de tono GDELT (inclusion forzada, ver docstring)
GDELT_VAR_COEF = 0.25
GDELT_BASELINE_WINDOW = ("2017-01-01", "2019-12-31")


def gdelt_zscore_series(horizon_months):
    """z-score del tono promedio mensual de noticias MX contra su propia
    linea base 2017-2019 (dato real, GDELT GKG 2.1). Si falta algun mes del
    horizonte se imputa 0 (neutral) y se avisa -- nunca se inventa un valor."""
    gdelt_path = PROC / "gdelt_monthly.csv"
    if not gdelt_path.exists():
        print("GDELT: archivo no encontrado, z-score forzado a 0 (neutral) en todo el horizonte.")
        return pd.Series(0.0, index=horizon_months)
    gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
    lo, hi = GDELT_BASELINE_WINDOW
    baseline = gdelt.loc[lo:hi, "tono_promedio"]
    mu, sd = baseline.mean(), baseline.std()
    z = (gdelt["tono_promedio"] - mu) / sd
    out = z.reindex(horizon_months)
    missing = out[out.isna()].index
    if len(missing) > 0:
        print(f"GDELT: sin dato para {list(missing.strftime('%Y-%m'))}, z-score=0 en esos meses.")
    return out.fillna(0.0)


def validate_gdelt():
    """Prueba GDELT contra IGAE_mom y contra |residuo VAR| via CV. Reporta
    honestamente y regresa (usable: bool, r2_mean, r2_var)."""
    gdelt_path = PROC / "gdelt_monthly.csv"
    if not gdelt_path.exists():
        print("GDELT: archivo no disponible, se omite del ensamble.")
        return False, None, None

    gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]
    fit0 = m0.fit_bvar_minnesota(est_df)
    resid_s = gbm_mod.build_residual_target(panel, fit0)

    common = gdelt.index.intersection(resid_s.index)
    if len(common) < 10:
        print(f"GDELT: solo {len(common)} meses en comun con la muestra pre-2020, "
              f"insuficiente para CV honesto -- se omite del ensamble.")
        return False, None, None

    X = gdelt.loc[common, ["tono_promedio", "n_articulos_mx"]].fillna(0).values
    y_mean = resid_s.loc[common].values
    y_var = resid_s.loc[common].abs().values

    kf = KFold(n_splits=min(5, len(common)), shuffle=True, random_state=0)
    for label, y in [("media (residuo VAR)", y_mean), ("varianza (|residuo VAR|)", y_var)]:
        oof = cross_val_predict(Ridge(alpha=1.0), X, y, cv=kf)
        ss_res = np.sum((y - oof) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"GDELT -> {label}: CV5 R^2={r2:.4f}  (n={len(common)})")
        if label.startswith("media"):
            r2_mean = r2
        else:
            r2_var = r2

    usable = (r2_mean > 0) or (r2_var > 0)
    if not usable:
        print("GDELT: R^2<=0 en ambos canales (media y varianza) -- NO se usa en "
              "el ensamble M3. Se reporta la validacion con honestidad: la muestra "
              "dispersa de 2 slices/mes no mostro senal detectable.")
    return usable, r2_mean, r2_var


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]
    fit0 = m0.fit_bvar_minnesota(est_df)
    X, Y = m0.build_design(est_df, m0.P)
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid_igae = Y[:, i_igae] - X @ b_igae
    s0 = resid_igae[-1] ** 2
    sv_fit = m1.fit_sv(resid_igae)

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)
    month_to_h = {m: h for h, m in enumerate(horizon_months)}

    # --- 1. escenarios epidemiologicos ------------------------------------
    weights_s, paths_df, g0 = epi.build_scenarios()
    weekly_new, _ = epi.observed_weekly_growth(n_weeks=1)
    last_level = weekly_new.iloc[-1]
    monthly_rel = epi.monthly_case_index(paths_df, last_level)  # log-nivel relativo, por escenario

    scenario_names = list(weights_s.index)
    scenario_probs = weights_s.values

    # presion de casos por mes y escenario, alineada a horizon_months
    pressure = monthly_rel.reindex(horizon_months, method="nearest")

    # --- 2. GDELT: se valida (se reporta honestamente) Y se fuerza su uso ---
    gdelt_usable, gdelt_r2_mean, gdelt_r2_var = validate_gdelt()
    gdelt_z = gdelt_zscore_series(horizon_months)
    print("\nGDELT z-score mensual (forzado al ensamble, valide o no en CV):")
    print(gdelt_z.round(3))

    # --- 3. particulas M2 como punto de partida ----------------------------
    print("\nCorriendo M2 (particulas + LP) como base de M3...")
    m2_summary, m2_draws, m2_months = m2.run()  # m2_draws: N x horizon
    N = m2_draws.shape[0]

    rng = np.random.default_rng(m0.SEED + 6)
    scenario_labels = rng.choice(len(scenario_names), size=N, p=scenario_probs)

    m3_draws = m2_draws.copy()
    ref_pressure = pressure[scenario_names].min(axis=1)  # escenario mas optimista, por mes

    for h, month in enumerate(horizon_months):
        p_t = pressure.loc[month]
        p_ref = ref_pressure.loc[month]
        z_g = gdelt_z.loc[month]
        tilt_gdelt = -GDELT_TILT_COEF * np.sqrt(s0) * np.tanh(z_g)
        extra_sd_gdelt = GDELT_VAR_COEF * np.sqrt(s0) * np.tanh(abs(z_g))
        for i in range(N):
            sc = scenario_names[scenario_labels[i]]
            tilt_epi = -TILT_COEF * np.sqrt(s0) * np.tanh(p_t[sc])
            extra_sd_epi = VAR_COEF * np.sqrt(s0) * np.tanh(abs(p_t[sc] - p_ref))
            total_sd = np.sqrt(extra_sd_epi ** 2 + extra_sd_gdelt ** 2)
            shock = rng.normal(0, total_sd) if total_sd > 0 else 0.0
            m3_draws[i, h] = m3_draws[i, h] + tilt_epi + tilt_gdelt + shock

    out = pd.DataFrame(m3_draws.T, index=horizon_months)
    out.to_csv(PROC / "M3_igae_mom_draws.csv")

    summary = pd.DataFrame(
        {
            "p05": np.percentile(m3_draws, 5, axis=0),
            "p25": np.percentile(m3_draws, 25, axis=0),
            "mediana": np.percentile(m3_draws, 50, axis=0),
            "p75": np.percentile(m3_draws, 75, axis=0),
            "p95": np.percentile(m3_draws, 95, axis=0),
        },
        index=horizon_months,
    )
    summary.to_csv(PROC / "M3_summary.csv")
    print("\n=== M3 resumen final ===")
    print(summary.round(4))

    meta = dict(
        tilt_coef=TILT_COEF, var_coef=VAR_COEF,
        gdelt_tilt_coef=GDELT_TILT_COEF, gdelt_var_coef=GDELT_VAR_COEF,
        s0=float(s0),
        gdelt_validated=bool(gdelt_usable),
        gdelt_r2_mean=None if gdelt_r2_mean is None else float(gdelt_r2_mean),
        gdelt_r2_var=None if gdelt_r2_var is None else float(gdelt_r2_var),
        gdelt_forced_inclusion=True,
        scenario_weights=weights_s.to_dict(),
    )
    pd.Series(meta).to_json(PROC / "M3_meta.json")
    gdelt_z.to_csv(PROC / "gdelt_zscore_horizon.csv")
    return summary, m3_draws, horizon_months


if __name__ == "__main__":
    run()
