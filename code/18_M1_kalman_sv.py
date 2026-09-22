"""
18_M1_kalman_sv.py

M1 = BVAR Minnesota + Kalman lineal-normal + Volatilidad Estocastica (SV).

A diferencia de LP (reescalamiento con decaimiento DETERMINISTA hacia 1),
la SV al estilo Carriero es una CAMINATA ALEATORIA en el log-volatilidad
(sin reversion): h_t = h_{t-1} + eta_t. Es la "escala endogena, evoluciona
con la informacion" de la diapositiva 6.

Se prueban 2 especificaciones (Carriero et al. 2016 permite ambas variantes
en la practica) y se elige la de mejor ajuste por verosimilitud de backcast
(ultimos 6 meses ANTES del corte, sin tocar nada posterior):

  SV-A: la volatilidad afecta SOLO la innovacion del estado (Q_t = h_t).
  SV-B: la volatilidad afecta el estado Y las mediciones por igual
        (Q_t = h_t, R_i,t = h_t * R_i) -- un solo factor de volatilidad
        "de la economia", no solo del factor.

Estimacion de h_t: metodo aproximado de Harvey-Ruiz-Shephard (1994), igual
que en 06_M1_volatility.py, pero SIN reversion (rho=1, caminata aleatoria
pura) -- la diferencia clave con LP.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "models_agosto"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
panelmod = import_module("17_agosto_panel_amplio")
m2mod = import_module("07b_M2_dfm_robusto")

N_DRAWS = 10_000
BACKCAST_MONTHS = 6


def kalman_filter_sv(panel, cols_available, H_dict, R_dict, F, Q_base, h_series, mode="A"):
    """Igual que kalman_filter_missing, pero Q y (en modo B) R se escalan
    por exp(h_t) periodo a periodo."""
    T, m = len(panel), F.shape[0]
    x_filt = np.zeros((T, m)); P_filt = np.zeros((T, m, m))
    x_pred = np.zeros((T, m)); P_pred = np.zeros((T, m, m))
    x, P = np.zeros(m), np.eye(m) * 10.0
    loglik = 0.0
    for t in range(T):
        scale_t = np.exp(h_series[t])
        Q_t = Q_base * scale_t
        xp = F @ x
        Pp = F @ P @ F.T + Q_t
        x_pred[t], P_pred[t] = xp, Pp
        avail = cols_available[t]
        if not avail:
            x, P = xp, Pp
        else:
            Hrows = np.vstack([H_dict[c] for c in avail])
            Rdiag = np.array([R_dict[c] * (scale_t if mode == "B" else 1.0) for c in avail])
            y = np.array([panel.iloc[t][c] for c in avail])
            yhat = Hrows @ xp
            S = Hrows @ Pp @ Hrows.T + np.diag(Rdiag)
            innov = y - yhat
            Sinv = np.linalg.inv(S)
            K = Pp @ Hrows.T @ Sinv
            x = xp + K @ innov
            P = Pp - K @ Hrows @ Pp
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (len(y) * np.log(2 * np.pi) + logdet + innov @ Sinv @ innov)
        x_filt[t], P_filt[t] = x, P
    return x_filt, P_filt, x_pred, P_pred, loglik


def estimate_h(panel, cols_available, H_dict, R_dict, F, mode, n_iter=15):
    """Estima h_t (log-volatilidad, caminata aleatoria) via el metodo
    aproximado Harvey-Ruiz-Shephard: filtra con h=0 fijo, usa el tamano del
    residuo de medicion como proxy de la volatilidad realizada, suaviza con
    un Kalman simple sobre log(residuo^2), itera."""
    T = len(panel)
    h = np.zeros(T)
    Q_base = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for it in range(n_iter):
        x_filt, P_filt, x_pred, P_pred, ll = kalman_filter_sv(
            panel, cols_available, H_dict, R_dict, F, Q_base, h, mode=mode)
        # proxy de volatilidad realizada: innovacion de PIB_q cuando esta disponible,
        # si no, la innovacion promedio de las series disponibles ese mes
        proxies = []
        for t in range(T):
            avail = cols_available[t]
            if not avail:
                proxies.append(np.nan)
                continue
            Hrows = np.vstack([H_dict[c] for c in avail])
            innov = np.array([panel.iloc[t][c] for c in avail]) - Hrows @ x_pred[t]
            Rdiag = np.array([R_dict[c] for c in avail])
            z = innov / np.sqrt(Rdiag)
            proxies.append(np.mean(z ** 2))
        proxies = pd.Series(proxies, index=panel.index).ffill().bfill()
        log_scale_obs = np.log(np.maximum(proxies.values, 1e-3))
        # suavizado simple: promedio movil causal (aprox. de un Kalman-smoother
        # de una caminata aleatoria con ruido de observacion) -- suficiente
        # para esta aproximacion declarada.
        h_new = pd.Series(log_scale_obs).ewm(span=3, adjust=False).mean().values
        h = 0.5 * h + 0.5 * h_new
    return h, ll


def run(panel, monthly_cols, params, target_last_month=pd.Timestamp("2020-07-01")):
    phi, lam, R = params["phi"], params["lam"], params["R"]
    F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])
    H_dict = {c: np.array([lam[c], 0, 0]) for c in monthly_cols}
    H_dict["PIB_q"] = np.array([1 / 3, 1 / 3, 1 / 3])
    R_dict = dict(R)

    cols_available = [[c for c in panel.columns if not np.isnan(panel.iloc[t][c])] for t in range(len(panel))]

    results = {}
    for mode in ["A", "B"]:
        h, ll = estimate_h(panel, cols_available, H_dict, R_dict, F, mode)
        Q_base = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
        x_filt, P_filt, x_pred, P_pred, loglik = kalman_filter_sv(
            panel, cols_available, H_dict, R_dict, F, Q_base, h, mode=mode)

        # verosimilitud de backcast: ultimos BACKCAST_MONTHS meses antes del corte
        backcast_ll = 0.0
        for t in range(len(panel) - BACKCAST_MONTHS, len(panel)):
            avail = cols_available[t]
            if not avail:
                continue
            Hrows = np.vstack([H_dict[c] for c in avail])
            Rdiag = np.array([R_dict[c] * (np.exp(h[t]) if mode == "B" else 1.0) for c in avail])
            S = Hrows @ P_pred[t] @ Hrows.T + np.diag(Rdiag)
            innov = np.array([panel.iloc[t][c] for c in avail]) - Hrows @ x_pred[t]
            sign, logdet = np.linalg.slogdet(S)
            backcast_ll += -0.5 * (len(avail) * np.log(2 * np.pi) + logdet + innov @ np.linalg.inv(S) @ innov)

        results[mode] = dict(h=h, x_filt=x_filt, P_filt=P_filt, loglik=loglik, backcast_ll=backcast_ll)
        print(f"  SV-{mode}: loglik total={loglik:.1f}  loglik backcast ({BACKCAST_MONTHS}m)={backcast_ll:.1f}  "
              f"h_ultimo={h[-1]:.2f} (mult. varianza={np.exp(h[-1]):.2f}x)")

    best_mode = max(results, key=lambda m: results[m]["backcast_ll"])
    print(f"  -> Mejor especificacion SV: {best_mode}")
    return results, best_mode, F


def forecast_and_bridge(results, best_mode, F, panel, pib_qoq, target_last_month, seed=2001):
    fit = results[best_mode]
    x_filt, P_filt, h = fit["x_filt"], fit["P_filt"], fit["h"]
    phi = F[0, 0]
    h_months = target_last_month.to_period("M").ordinal - panel.index[-1].to_period("M").ordinal

    factor_hist = pd.Series(x_filt[:, 0], index=panel.index)
    factor_q = factor_hist.resample("QS").mean()
    factor_q_n = factor_hist.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)

    rng = np.random.default_rng(seed)
    s_last, P_last = x_filt[-1], P_filt[-1]
    h_last = h[-1]
    growth = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        xs = [s_d[0]]
        h_t = h_last
        for j in range(h_months):
            h_t = h_t + rng.normal(0, 0.3)  # caminata aleatoria (sin reversion) para el pronostico
            xs.append(phi * xs[-1] + np.exp(h_t / 2) * rng.standard_normal())
        f_q = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth[d] = coef_d[0] + coef_d[1] * f_q + rng.normal(0, sigma_b)
    return growth


def main():
    panel, monthly_cols, pib_qoq = panelmod.build_panel_agosto()
    params = panelmod.clean_params(panel, monthly_cols)
    print("Parametros limpios (compartidos):", {k: round(v, 3) for k, v in params["lam"].items()})

    print("\nEstimando y comparando especificaciones SV...")
    results, best_mode, F = run(panel, monthly_cols, params)

    growth = forecast_and_bridge(results, best_mode, F, panel, pib_qoq, pd.Timestamp("2020-07-01"))
    print(f"\n=== M1 (Kalman + SV-{best_mode}, panel amplio): pronostico 3T20 ===")
    print(f"Mediana={np.median(growth):.2f}%  IC95%=[{np.percentile(growth,2.5):.2f}, {np.percentile(growth,97.5):.2f}]")

    np.save(OUTPUT / "M1_kalman_sv_growth_draws.npy", growth)
    with open(OUTPUT / "M1_kalman_sv_fit.pkl", "wb") as f:
        pickle.dump(dict(best_mode=best_mode, results={k: {kk: vv for kk, vv in v.items() if kk != "P_filt"}
                                                          for k, v in results.items()}), f)
    print(f"Guardado: {OUTPUT / 'M1_kalman_sv_growth_draws.npy'}")


if __name__ == "__main__":
    main()
