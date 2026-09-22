"""
M1 = M0 (BVAR Minnesota) + reescalamiento de covarianza tipo Lenza y
Primiceri (2022), "How to estimate a VAR after March 2020" (Journal of
Applied Econometrics).

Metodologia real del paper (verificada via busqueda, no aproximada de
memoria): la covarianza residual del VAR se escala por un factor s_t que
vale 1 antes del inicio del choque (t*) y, a partir de t*, sigue

    s_{t*}   = s0_barra
    s_{t*+1} = s1_barra
    s_{t*+2} = s2_barra
    s_{t*+j} = 1 + (s2_barra - 1) * rho^(j-2)   para j >= 2

con theta = (s0_barra, s1_barra, s2_barra, rho) estimado por maxima
verosimilitud junto con el resto del VAR.

RESTRICCION REAL DE DATOS en este ejercicio: el corte de informacion
(15-jun-2020) deja SOLO UNA observacion post-choque dentro de la muestra
de estimacion (marzo 2020 -- ultimo IGAE oficial conocido). El modelo de
4 parametros de Lenza-Primiceri no es identificable con un solo punto
post-choque. Se usa entonces una version reducida, honesta sobre esa
limitacion:

  - s0_barra: MLE conjunta usando las 3 ecuaciones del VAR simultaneamente
    en t* (formula cerrada de un factor de escala comun sobre una normal
    multivariada: s0^2 = (1/n) * e_t*' Sigma^-1 e_t*, con Sigma estimada
    EXCLUYENDO t* para no contaminar la escala con el propio atipico).
  - rho: NO estimable con un solo punto; se fija en 0.75, dentro del rango
    tipico reportado por Lenza y Primiceri (2022) para series macro
    mensuales/trimestrales (rango aproximado 0.6-0.9 segun la serie).
    Documentado como supuesto, no como estimacion.
  - s1_barra, s2_barra: colapsados a la misma trayectoria geometrica que
    rho ya implica desde t* (s_{t*+j} = 1+(s0_barra-1)*rho^j, j=1,2,...),
    en vez de parametros libres adicionales sin informacion para
    identificarlos.

La covarianza COMPLETA (no solo la varianza de IGAE) se escala por s_t^2
en cada paso de la simulacion hacia adelante -- a diferencia de la version
anterior de este archivo (volatilidad estocastica univariada estilo
Harvey-Ruiz-Shephard 1994), que solo escalaba el residuo de IGAE.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
m0 = import_module("03_M0_bvar_minnesota")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

RHO = 0.75  # fijo, ver docstring -- no identificable con 1 obs. post-choque
T_STAR = pd.Timestamp("2020-03-01")  # primer mes anomalo dentro de la muestra


def fit_lenza_primiceri(est_df: pd.DataFrame, fit0: dict):
    """Devuelve (Sigma_normal, s0_barra, dates) -- covarianza 'normal' (sin
    el choque) y la escala conjunta MLE del punto anomalo t*."""
    X, Y = m0.build_design(est_df, m0.P)
    dates = est_df.index[m0.P:]
    resids = []
    for v in m0.VARS:
        b = fit0["equations"][v]["b_post"]
        r = Y[:, m0.VARS.index(v)] - X @ b
        resids.append(r)
    resids = np.array(resids).T  # T x n

    is_tstar = dates == T_STAR
    if not is_tstar.any():
        raise ValueError(f"T_STAR={T_STAR} no esta en la muestra de estimacion")
    idx_tstar = np.where(is_tstar)[0][0]

    mask_normal = np.ones(len(dates), dtype=bool)
    mask_normal[idx_tstar] = False
    Sigma_normal = np.cov(resids[mask_normal].T)

    e_tstar = resids[idx_tstar]
    n = len(m0.VARS)
    s0_sq = (e_tstar @ np.linalg.inv(Sigma_normal) @ e_tstar) / n
    s0_barra = np.sqrt(s0_sq)

    return Sigma_normal, s0_barra, dates, idx_tstar


def scale_path(s0_barra: float, horizon: int, rho: float = RHO) -> np.ndarray:
    """s_t para j=1..horizon meses despues de t* (t*=marzo 2020 = mes 0 de
    referencia; abril 2020 = j=1, primer mes del horizonte de pronostico)."""
    j = np.arange(1, horizon + 1)
    return 1.0 + (s0_barra - 1.0) * (rho ** j)


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]

    fit0 = m0.fit_bvar_minnesota(est_df)
    Sigma_normal, s0_barra, dates, idx_tstar = fit_lenza_primiceri(est_df, fit0)
    print(f"Lenza-Primiceri: s0_barra (escala conjunta en t*={T_STAR.date()}) = {s0_barra:.3f}")
    print(f"rho fijo (no identificable con 1 obs. post-choque) = {RHO}")

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)
    s_t = scale_path(s0_barra, horizon)
    print("\nTrayectoria de escala s_t por mes de pronostico:")
    for mth, s in zip(horizon_months, s_t):
        print(f"  {mth.date()}: s_t={s:.3f}  (Sigma escalada x{s**2:.2f})")

    L_normal = np.linalg.cholesky(Sigma_normal)
    i_igae = m0.VARS.index("IGAE_mom")

    rng = np.random.default_rng(m0.SEED + 1)
    N = m0.N_DRAWS
    draws = np.zeros((N, horizon, len(m0.VARS)))
    for d in range(N):
        betas, _ = m0.sample_posterior_draw(fit0, rng)
        hist = est_df[m0.VARS].values[-m0.P:].tolist()
        path = []
        for h in range(horizon):
            row = [1.0]
            for lag in range(1, m0.P + 1):
                row.extend(hist[-lag])
            row = np.array(row)
            mean = np.array([betas[v] @ row for v in m0.VARS])
            z = rng.standard_normal(len(m0.VARS))
            shock = s_t[h] * (L_normal @ z)  # covarianza COMPLETA escalada por s_t
            y_t = mean + shock
            path.append(y_t)
            hist.append(y_t.tolist())
        draws[d] = np.array(path)

    igae_draws = draws[:, :, i_igae]
    out = pd.DataFrame(igae_draws.T, index=horizon_months)
    out.to_csv(PROC / "M1_igae_mom_draws.csv")

    summary = pd.DataFrame(
        {
            "p05": np.percentile(igae_draws, 5, axis=0),
            "p25": np.percentile(igae_draws, 25, axis=0),
            "mediana": np.percentile(igae_draws, 50, axis=0),
            "p75": np.percentile(igae_draws, 75, axis=0),
            "p95": np.percentile(igae_draws, 95, axis=0),
        },
        index=horizon_months,
    )
    summary.to_csv(PROC / "M1_summary.csv")
    print()
    print(summary.round(4))

    meta = dict(s0_barra=float(s0_barra), rho=RHO, t_star=str(T_STAR.date()))
    pd.Series(meta).to_json(PROC / "M1_meta.json")
    return summary, igae_draws, horizon_months, Sigma_normal, s0_barra


# API reutilizada por M2/M3 (mismo nombre que la version anterior para no
# romper los imports rio abajo)
def simulate_forward_sv(sv_fit, horizon, rng, n_draws):
    """Compatibilidad: da la trayectoria de escala s_t repetida N veces
    (Lenza-Primiceri es determinista dado theta, a diferencia de la SV
    estocastica anterior -- no hay incertidumbre de estado adicional aqui,
    solo la que ya trae el remuestreo posterior de beta/Sigma)."""
    s_t = sv_fit["s_t"]
    return np.tile(s_t, (n_draws, 1))


def simulate_forward_sv_var(fit, history_df, betas, sigmas, corr, horizon, rng,
                             sigma_path_igae, i_igae):
    """Compatibilidad con la firma anterior (usada por M2). Ahora sigma_path_igae
    es en realidad la trayectoria s_t de Lenza-Primiceri, y se aplica a la
    covarianza COMPLETA via Cholesky de Sigma_normal (pasada en `corr`, que
    aqui contiene Sigma_normal en vez de una matriz de correlacion)."""
    names = fit["names"]
    p = fit["p"]
    n = len(names)
    L = np.linalg.cholesky(corr)
    hist = history_df[names].values[-p:].tolist()

    path = []
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(hist[-lag])
        row = np.array(row)
        mean = np.array([betas[v] @ row for v in names])
        z = rng.standard_normal(n)
        shock = sigma_path_igae[h] * (L @ z)
        y_t = mean + shock
        path.append(y_t)
        hist.append(y_t.tolist())
    return np.array(path)


def fit_sv(resid_igae):
    """Compatibilidad: envuelve fit_lenza_primiceri para exponer la misma
    interfaz que usan M2/M3 (que llaman m1.fit_sv(resid_igae))."""
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]
    fit0 = m0.fit_bvar_minnesota(est_df)
    Sigma_normal, s0_barra, dates, idx_tstar = fit_lenza_primiceri(est_df, fit0)

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    s_t = scale_path(s0_barra, len(horizon_months))
    return dict(Sigma_normal=Sigma_normal, s0_barra=s0_barra, s_t=s_t)


if __name__ == "__main__":
    run()
