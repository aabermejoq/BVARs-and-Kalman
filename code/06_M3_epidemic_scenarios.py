"""
M3 (parte 1/3): curvas epidemiologicas de escenario.

4 curvas de crecimiento SEMANAL de casos nuevos (log-crecimiento g(t), t en
semanas desde el corte 2020-06-15) via relajacion exponencial hacia una
tasa de largo plazo g_inf con velocidad 1/tau:

    g(t) = g_inf + (g0 - g_inf) * exp(-t / tau)

g0 = tasa de crecimiento semanal REAL observada al corte (dato real, no
supuesto -- ver calculo abajo). Los 4 escenarios:

  1. Nueva ola / rebrote : decae, pero con un repunte (bump gaussiano) antes de diciembre
  2. Rebote lento         : g_inf ligeramente negativo, tau grande (decae despacio)
  3. Rebote rapido        : g_inf muy negativo, tau chico (decae rapido)
  4. Meseta extendida     : g_inf ~ 0, tau muy grande (se estanca, no baja)

Ponderacion por verosimilitud bayesiana de backcast: se evalua CADA
escenario hacia ATRAS (t<0, las 4 semanas previas al corte) y se compara
contra el crecimiento REALMENTE observado en esas semanas (sin tocar
ningun dato posterior al corte). Un escenario cuya forma implique una
subida implausible del crecimiento en el pasado reciente (inconsistente
con lo ya observado) recibe menos peso.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

CUTOFF = pd.Timestamp("2020-06-15")

SCENARIOS = {
    "nueva_ola":        dict(g_inf=-0.02, tau=10.0, bump_week=22, bump_amp=0.10, bump_width=3.0),
    "rebote_lento":     dict(g_inf=-0.02, tau=16.0, bump_week=None, bump_amp=0.0, bump_width=1.0),
    "rebote_rapido":    dict(g_inf=-0.10, tau=3.0,  bump_week=None, bump_amp=0.0, bump_width=1.0),
    "meseta_extendida": dict(g_inf=0.00,  tau=26.0, bump_week=None, bump_amp=0.0, bump_width=1.0),
}


def observed_weekly_growth(cutoff=CUTOFF, n_weeks=8):
    """Crecimiento semanal log(casos_nuevos) de las n_weeks semanas antes del
    corte, SOLO con datos <= cutoff (sin fuga de informacion)."""
    covid = pd.read_csv(RAW / "covid_mexico.csv", index_col=0, parse_dates=True)
    covid = covid.loc[:cutoff]
    weekly_new = covid["total_cases"].resample("W").last().diff().dropna()
    weekly_new = weekly_new[(weekly_new.index <= cutoff) & (weekly_new > 0)]
    g = np.log(weekly_new).diff().dropna()
    return weekly_new.tail(n_weeks + 1), g.tail(n_weeks)


def g_scenario(t, params):
    g_inf, tau = params["g_inf"], params["tau"]
    g0 = params["g0"]
    base = g_inf + (g0 - g_inf) * np.exp(-t / tau)
    if params.get("bump_week") is not None:
        bump = params["bump_amp"] * np.exp(-0.5 * ((t - params["bump_week"]) / params["bump_width"]) ** 2)
        base = base + bump
    return base


def backcast_weight(g0, g_obs_recent: pd.Series, params, tau_lik=0.04):
    """Verosimilitud gaussiana comparando el crecimiento IMPLICADO por el
    escenario en las semanas t=-k..-1 contra el crecimiento REALMENTE
    observado en esas mismas semanas (backcast, no usa nada posterior al corte)."""
    k = len(g_obs_recent)
    t_grid = np.arange(-k, 0)
    g_impl = np.array([g_scenario(t, {**params, "g0": g0}) for t in t_grid])
    g_obs = g_obs_recent.values
    ll = -0.5 * np.sum(((g_obs - g_impl) / tau_lik) ** 2)
    return ll, g_impl


def build_scenarios(n_backcast_weeks=4):
    weekly_new, g = observed_weekly_growth(n_weeks=n_backcast_weeks + 1)
    g0 = g.iloc[-1]  # tasa de crecimiento semanal REAL en la ultima semana antes/en el corte
    g_recent = g.iloc[-n_backcast_weeks:]
    print(f"Crecimiento semanal observado (log) ultimas {n_backcast_weeks} semanas antes del corte:")
    print(g_recent.round(4))
    print(f"g0 (tasa al corte) = {g0:.4f}  (~{100*(np.exp(g0)-1):.1f}% semanal)")

    lls = {}
    impls = {}
    for name, params in SCENARIOS.items():
        ll, g_impl = backcast_weight(g0, g_recent, params)
        lls[name] = ll
        impls[name] = g_impl
        print(f"  {name:<18} backcast logLik={ll:8.3f}  g_impl(recientes)={np.round(g_impl,4)}")

    ll_arr = np.array(list(lls.values()))
    w = np.exp(ll_arr - ll_arr.max())
    w /= w.sum()
    weights = dict(zip(lls.keys(), w))
    print("\nPesos bayesianos de backcast por escenario:")
    for k, v in weights.items():
        print(f"  {k:<18} {v:.4f}")

    # trayectoria futura de crecimiento semanal, semanas 0..28 (corte -> ~fin dic 2020)
    weeks_fwd = np.arange(0, 29)
    scenario_paths = {}
    for name, params in SCENARIOS.items():
        scenario_paths[name] = np.array([g_scenario(t, {**params, "g0": g0}) for t in weeks_fwd])

    fwd_dates = CUTOFF + pd.to_timedelta(weeks_fwd, unit="W")
    paths_df = pd.DataFrame(scenario_paths, index=fwd_dates)
    paths_df.to_csv(PROC / "epi_scenario_growth_paths.csv")

    weights_s = pd.Series(weights)
    weights_s.to_csv(PROC / "epi_scenario_weights.csv")

    return weights_s, paths_df, g0


def monthly_case_index(paths_df: pd.DataFrame, last_level: float):
    """Integra el crecimiento semanal a nivel de casos, agrega a promedio
    mensual, y expresa como log-nivel relativo al nivel observado en el
    corte (para usarlo como 'presion epidemiologica' mensual)."""
    log_level = np.log(last_level) + paths_df.cumsum()
    weekly_level = np.exp(log_level)
    monthly_level = weekly_level.resample("MS").mean()
    monthly_log_rel = np.log(monthly_level) - np.log(last_level)
    return monthly_log_rel


if __name__ == "__main__":
    weights, paths_df, g0 = build_scenarios()
    weekly_new, _ = observed_weekly_growth(n_weeks=1)
    last_level = weekly_new.iloc[-1]
    monthly_rel = monthly_case_index(paths_df, last_level)
    monthly_rel.to_csv(PROC / "epi_scenario_monthly_case_pressure.csv")
    print("\nPresion epidemiologica mensual relativa al corte (log-nivel de casos semanales, por escenario):")
    print(monthly_rel.round(3))
