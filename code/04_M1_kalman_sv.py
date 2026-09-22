"""
M1 = M0 (BVAR Minnesota) + filtro de Kalman lineal-normal + volatilidad
estocastica (SV) sobre el residuo de la ecuacion de IGAE.

Tecnica: aproximacion cuasi-verosimil lineal-Gaussiana de Harvey, Ruiz y
Shephard (1994). Se toma el residuo e_t de la ecuacion de IGAE del VAR M0
(mismos coeficientes posteriores de M0 -- M1 no reestima la media, solo
anade dinamica de varianza), se define

  y*_t = log(e_t^2 + c)  ~  h_t + xi_t,      xi_t ~ N(-1.2704, pi^2/2)
  h_t  = mu(1-phi) + phi*h_{t-1} + eta_t,    eta_t ~ N(0, sigma_eta^2)

que es un modelo espacio-estado LINEAL Y GAUSSIANO (la no-gaussianidad de
xi_t se aproxima por una normal, truco estandar QML de Harvey-Ruiz-Shephard;
la version NO lineal/no aproximada con mixtura de normales o particulas se
deja para M2). Se filtra con Kalman estandar, se estiman (mu, phi,
sigma_eta) por maxima verosimilitud (descomposicion de errores de
prediccion del propio filtro de Kalman), y se propaga h_t hacia adelante
con la recursion AR(1) ANCLADA al ultimo estado filtrado real (no un valor
supuesto) -- lo que automaticamente ensancha la banda cuando el choque de
marzo 2020 ya esta en el residuo observado al corte.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from importlib import import_module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
m0 = import_module("03_M0_bvar_minnesota")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

XI_MEAN = -1.2704
XI_VAR = (np.pi ** 2) / 2  # 4.9348


def kalman_filter_sv(ystar, mu, phi, sigma_eta2):
    """Kalman filter para h_t = mu(1-phi)+phi*h_{t-1}+eta ; ystar_t = h_t + xi_t (xi ~ N(XI_MEAN,XI_VAR))."""
    T = len(ystar)
    h_pred = np.zeros(T)
    P_pred = np.zeros(T)
    h_filt = np.zeros(T)
    P_filt = np.zeros(T)
    loglik = 0.0

    h_prev = mu
    P_prev = sigma_eta2 / max(1 - phi ** 2, 1e-6)

    for t in range(T):
        h_pred[t] = mu * (1 - phi) + phi * h_prev if t > 0 else mu
        P_pred[t] = phi ** 2 * P_prev + sigma_eta2 if t > 0 else P_prev

        v = (ystar[t] - XI_MEAN) - h_pred[t]
        F = P_pred[t] + XI_VAR
        K = P_pred[t] / F

        h_filt[t] = h_pred[t] + K * v
        P_filt[t] = P_pred[t] * (1 - K)

        loglik += -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)

        h_prev, P_prev = h_filt[t], P_filt[t]

    return h_filt, P_filt, h_pred, P_pred, loglik


def fit_sv(resid: np.ndarray):
    c = 1e-6 * np.var(resid)
    ystar = np.log(resid ** 2 + c)

    def negloglik(params):
        mu, phi_raw, log_sigma_eta2 = params
        phi = np.tanh(phi_raw)  # mapea a (-1,1)
        sigma_eta2 = np.exp(log_sigma_eta2)
        try:
            *_, ll = kalman_filter_sv(ystar, mu, phi, sigma_eta2)
        except Exception:
            return 1e10
        if not np.isfinite(ll):
            return 1e10
        return -ll

    x0 = [ystar.mean(), np.arctanh(0.9), np.log(0.05)]
    res = minimize(negloglik, x0, method="Nelder-Mead",
                    options=dict(maxiter=2000, xatol=1e-6, fatol=1e-6))
    mu, phi_raw, log_sigma_eta2 = res.x
    phi = np.tanh(phi_raw)
    sigma_eta2 = np.exp(log_sigma_eta2)
    h_filt, P_filt, *_ = kalman_filter_sv(ystar, mu, phi, sigma_eta2)
    return dict(mu=mu, phi=phi, sigma_eta2=sigma_eta2, h_filt=h_filt, P_filt=P_filt, ystar=ystar)


def simulate_forward_sv(sv_fit, horizon, rng, n_draws):
    """Propaga h_t (log-varianza) desde el ULTIMO ESTADO FILTRADO REAL (ancla real,
    no supuesta) usando la recursion AR(1) estimada, con incertidumbre de estado."""
    mu, phi, sigma_eta2 = sv_fit["mu"], sv_fit["phi"], sv_fit["sigma_eta2"]
    h0 = sv_fit["h_filt"][-1]
    P0 = sv_fit["P_filt"][-1]

    h_draws = np.zeros((n_draws, horizon))
    for d in range(n_draws):
        h_prev = rng.normal(h0, np.sqrt(P0))
        for t in range(horizon):
            eta = rng.normal(0, np.sqrt(sigma_eta2))
            h_t = mu * (1 - phi) + phi * h_prev + eta
            h_draws[d, t] = h_t
            h_prev = h_t
    sigma_draws = np.exp(h_draws / 2)  # n_draws x horizon
    return sigma_draws


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]

    fit0 = m0.fit_bvar_minnesota(est_df)
    X, Y = m0.build_design(est_df, m0.P)
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid_igae = Y[:, i_igae] - X @ b_igae
    print(f"M1: residuo IGAE en muestra, std={resid_igae.std():.4f}, "
          f"max|resid| en {est_df.index[m0.P:][np.argmax(np.abs(resid_igae))].date()}")

    sv_fit = fit_sv(resid_igae)
    print(f"SV estimado: mu={sv_fit['mu']:.3f}  phi={sv_fit['phi']:.3f}  "
          f"sigma_eta={np.sqrt(sv_fit['sigma_eta2']):.3f}")
    print(f"Ultimo estado filtrado h_T={sv_fit['h_filt'][-1]:.3f} "
          f"-> sigma_T={np.exp(sv_fit['h_filt'][-1]/2):.4f} "
          f"(sigma promedio historico={np.exp(sv_fit['mu']/2):.4f})")

    horizon_months = pd.date_range(
        m0.LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), m0.FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)
    rng = np.random.default_rng(m0.SEED + 1)

    sigma_draws = simulate_forward_sv(sv_fit, horizon, rng, m0.N_DRAWS)

    # Simulacion VAR igual que M0 pero reemplazando sigma constante de IGAE por
    # la trayectoria SV muestreada en cada draw (lesson #5: escalar por sqrt(s0)).
    draws = np.zeros((m0.N_DRAWS, horizon, len(m0.VARS)))
    rng2 = np.random.default_rng(m0.SEED + 2)
    for d in range(m0.N_DRAWS):
        betas, sigmas = m0.sample_posterior_draw(fit0, rng2)
        sigmas_sv = dict(sigmas)
        sigmas_sv["IGAE_mom"] = None  # se reemplaza dentro de la simulacion paso a paso
        path = simulate_forward_sv_var(
            fit0, est_df, betas, sigmas, fit0["corr"], horizon, rng2,
            sigma_path_igae=sigma_draws[d], i_igae=i_igae,
        )
        draws[d] = path

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
    print(summary.round(4))
    return summary, igae_draws, horizon_months


def simulate_forward_sv_var(fit, history_df, betas, sigmas, corr, horizon, rng,
                             sigma_path_igae, i_igae):
    names = fit["names"]
    p = fit["p"]
    n = len(names)
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(n))
    hist = history_df[names].values[-p:].tolist()

    path = []
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(hist[-lag])
        row = np.array(row)
        mean = np.array([betas[v] @ row for v in names])
        z = rng.standard_normal(n)
        shock = L @ z
        sig = np.array([sigmas[v] for v in names])
        sig[i_igae] = sigma_path_igae[h]  # sustituye por SV en la ecuacion de IGAE
        y_t = mean + sig * shock
        path.append(y_t)
        hist.append(y_t.tolist())
    return np.array(path)


if __name__ == "__main__":
    run()
