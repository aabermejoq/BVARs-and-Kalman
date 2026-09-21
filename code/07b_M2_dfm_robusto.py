"""
07b_M2_dfm_robusto.py

Rehace M2 con un DFM propiamente estimado (EM regularizado, estilo
Doz-Giannone-Reichlin 2011/2012), en vez del atajo de "promedio simple +
Kalman univariado" de 07_M2_state_kalman.py. Esto responde directamente a
la critica de que ese atajo, aunque robusto, no deja que los datos digan
cuanto pesa cada serie.

Modelo (forma companion, para poder promediar 3 meses = 1 trimestre):
  S_t = [x_t, x_{t-1}, x_{t-2}]'
  x_t = phi*x_{t-1} + w_t,      w_t ~ N(0, Q=1)   <- Q normalizado (identificacion)
  IMSS_t   = lambda_IMSS  * x_t + eps_1,  eps_1 ~ N(0,R_IMSS)
  ANTAD_t  = lambda_ANTAD * x_t + eps_2,  eps_2 ~ N(0,R_ANTAD)
  AUTOS_t  = lambda_AUTOS * x_t + eps_3,  eps_3 ~ N(0,R_AUTOS)
  PIB_q_t  = (x_t+x_{t-1}+x_{t-2})/3 + eps_Q,  eps_Q ~ N(0,R_PIB)   (solo fin de trimestre)

Estimacion: EM con manejo nativo de datos faltantes en el filtro/suavizador
de Kalman (Shumway-Stoffer). LA CORRECCION CLAVE frente a mi primer intento
fragil: cada R_i se actualiza en el M-step como la MODA POSTERIOR bajo un
prior Inverse-Gamma(a0, b0_i) -- no la varianza muestral cruda -- con
b0_i anclado a la varianza de un AR(1) univariado de esa serie (su escala
"de sentido comun"). Esto evita que R_i colapse a 0 cuando una serie ajusta
"demasiado bien" en una muestra pequena, que fue exactamente el problema
del intento original documentado en 07_M2_state_kalman.py.

Dado que el estado esta en forma companion, Cov(x_t, x_{t-1} | datos) es
simplemente el elemento fuera de la diagonal (0,1) de la matriz de
covarianza SUAVIZADA en el tiempo t -- no hace falta un suavizador de
covarianza retardada aparte.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"

N_DRAWS = 10_000
RNG_SEED = 62345
MAX_EM_ITER = 100
TOL = 1e-5


def ar1_variance(y):
    y = y[~np.isnan(y)]
    X = np.column_stack([y[:-1], np.ones(len(y) - 1)])
    coef, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    resid = y[1:] - X @ coef
    return resid.var(ddof=2)


def build_data(asof=pd.Timestamp("2020-05-15"), fast_end=pd.Timestamp("2020-04-01"),
               core_path=None):
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)
    core = pd.read_csv(core_path or (INTERIM / "core_quarterly_panel.csv"), index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)

    imss = series["IMSS"]["IMSS_empleos"].dropna()
    antad = series["Consumo"]["ANTAD"].dropna()
    autos = series["Consumo"]["AUTOS"].dropna()

    idx = pd.date_range("1997-07-01", fast_end, freq="MS")
    panel = pd.DataFrame(index=idx)
    panel["IMSS"] = (100 * np.log(imss / imss.shift(12))).reindex(idx)
    panel["ANTAD"] = (100 * np.log(antad / antad.shift(12))).reindex(idx)
    panel["AUTOS"] = (100 * np.log(autos / autos.shift(12))).reindex(idx)

    pib_series = pd.Series(index=idx, dtype=float)
    for dte, val in pib_qoq.items():
        close_month = dte + pd.DateOffset(months=2)
        if close_month in pib_series.index:
            pib_series[close_month] = val
    panel["PIB_q"] = pib_series

    return panel


def kalman_filter_missing(Y, cols_available, H_dict, R_dict, F, Q):
    T, m = len(Y), F.shape[0]
    x_pred = np.zeros((T, m)); P_pred = np.zeros((T, m, m))
    x_filt = np.zeros((T, m)); P_filt = np.zeros((T, m, m))
    x, P = np.zeros(m), np.eye(m) * 10.0
    loglik = 0.0
    for t in range(T):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        x_pred[t], P_pred[t] = xp, Pp
        avail = cols_available[t]
        if not avail:
            x, P = xp, Pp
        else:
            Hrows = np.vstack([H_dict[c] for c in avail])
            Rdiag = np.array([R_dict[c] for c in avail])
            y = np.array([Y.loc[Y.index[t], c] for c in avail])
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


def rts_smoother(x_filt, P_filt, x_pred, P_pred, F):
    T, m = x_filt.shape
    x_s = np.zeros_like(x_filt); P_s = np.zeros_like(P_filt)
    x_s[-1], P_s[-1] = x_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        J = P_filt[t] @ F.T @ np.linalg.inv(Pp_next)
        x_s[t] = x_filt[t] + J @ (x_s[t + 1] - F @ x_filt[t])
        P_s[t] = P_filt[t] + J @ (P_s[t + 1] - Pp_next) @ J.T
    return x_s, P_s


def em_dfm(panel, a0=3.0, verbose=True):
    monthly_cols = ["IMSS", "ANTAD", "AUTOS"]
    prior_scale = {c: ar1_variance(panel[c].values) for c in monthly_cols}
    prior_scale["PIB_q"] = panel["PIB_q"].dropna().var()

    cols_available = []
    for t in range(len(panel)):
        row = panel.iloc[t]
        cols_available.append([c for c in panel.columns if not np.isnan(row[c])])

    phi, lam = 0.3, {c: 1.0 for c in monthly_cols}
    R = {c: prior_scale[c] for c in monthly_cols}
    R["PIB_q"] = prior_scale["PIB_q"]
    Q = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])

    H_PIB = np.array([1 / 3, 1 / 3, 1 / 3])
    prev_ll = -np.inf

    for it in range(MAX_EM_ITER):
        H_dict = {c: np.array([lam[c], 0, 0]) for c in monthly_cols}
        H_dict["PIB_q"] = H_PIB
        R_dict = dict(R)

        x_filt, P_filt, x_pred, P_pred, ll = kalman_filter_missing(
            panel, cols_available, H_dict, R_dict, F, Q)
        x_s, P_s = rts_smoother(x_filt, P_filt, x_pred, P_pred, F)

        # --- M-step: cargas y varianzas de las series mensuales ---
        new_lam, new_R = {}, {}
        for c in monthly_cols:
            obs_t = [t for t in range(len(panel)) if c in cols_available[t]]
            Sxx = sum(P_s[t, 0, 0] + x_s[t, 0] ** 2 for t in obs_t)
            Sxy = sum(x_s[t, 0] * panel.iloc[t][c] for t in obs_t)
            lam_new = Sxy / Sxx
            ss = sum((panel.iloc[t][c] - lam_new * x_s[t, 0]) ** 2 + (lam_new ** 2) * P_s[t, 0, 0]
                     for t in obs_t)
            n_i = len(obs_t)
            b0 = (a0 + 1) * prior_scale[c]
            R_new = (ss + 2 * b0) / (n_i + 2 * a0 + 2)
            new_lam[c], new_R[c] = lam_new, R_new

        # --- M-step: varianza de la medicion trimestral del PIB ---
        obs_q = [t for t in range(len(panel)) if "PIB_q" in cols_available[t]]
        ss_q = sum((panel.iloc[t]["PIB_q"] - H_PIB @ x_s[t]) ** 2 + H_PIB @ P_s[t] @ H_PIB
                   for t in obs_q)
        b0_q = (a0 + 1) * prior_scale["PIB_q"]
        R_q_new = (ss_q + 2 * b0_q) / (len(obs_q) + 2 * a0 + 2)

        # --- M-step: phi (usando que x_{t-1} es la 2a componente del estado companion) ---
        num = sum(P_s[t, 0, 1] + x_s[t, 0] * x_s[t, 1] for t in range(1, len(panel)))
        den = sum(P_s[t, 1, 1] + x_s[t, 1] ** 2 for t in range(1, len(panel)))
        phi_new = float(np.clip(num / den, -0.995, 0.995))

        lam, R, phi = new_lam, dict(new_R, PIB_q=R_q_new), phi_new
        F = np.array([[phi, 0, 0], [1, 0, 0], [0, 1, 0]])

        if verbose and (it % 10 == 0 or it == MAX_EM_ITER - 1):
            print(f"  EM iter {it}: loglik={ll:.2f}  phi={phi:.3f}  "
                  f"lam={{k: round(v,2) for k,v in lam.items()}}  "
                  f"R={{k: round(v,3) for k,v in R.items()}}")

        if abs(ll - prev_ll) < TOL * abs(prev_ll if prev_ll != 0 else 1):
            break
        prev_ll = ll

    return dict(phi=phi, lam=lam, R=R, Q=Q, F=F, x_filt=x_filt, P_filt=P_filt,
                x_smooth=x_s, P_smooth=P_s, loglik=ll, n_iter=it + 1)


def main():
    panel = build_data()
    print(f"Panel: {panel.index.min().date()} -> {panel.index.max().date()} (n={len(panel)})")
    print("Ajustando DFM via EM regularizado...")
    fit = em_dfm(panel)

    print(f"\nConvergencia en {fit['n_iter']} iteraciones, log-verosimilitud final={fit['loglik']:.2f}")
    print(f"phi={fit['phi']:.3f}")
    print("Cargas (lambda):", {k: round(v, 3) for k, v in fit["lam"].items()})
    print("Varianzas idiosincraticas (R):", {k: round(v, 3) for k, v in fit["R"].items()})

    x_filt = fit["x_filt"][:, 0]
    print("\nEstado FILTRADO (ultimos 8 meses):")
    for d, x in list(zip(panel.index, x_filt))[-8:]:
        print(f"  {d.date()}: {x:+.2f}")

    # --- Ecuacion puente: PIB_q(%) = a + b*factor_trimestral (usa el estado suavizado historico) ---
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    pib_qoq = 100 * core["log_PIB"].diff(1)
    factor_m = pd.Series(x_filt, index=panel.index)
    factor_q = factor_m.resample("QS").mean()
    factor_q_n = factor_m.resample("QS").count()
    factor_q = factor_q[factor_q_n == 3]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)
    print(f"\nEcuacion puente: PIB_q(%) = {coef_b[0]:.3f} + {coef_b[1]:.3f}*factor (sigma={sigma_b:.3f})")

    # --- Pronostico h=2 (mayo, junio) ---
    rng = np.random.default_rng(RNG_SEED)
    phi, F, Q = fit["phi"], fit["F"], fit["Q"]
    s_last, P_last = fit["x_filt"][-1], fit["P_filt"][-1]
    growth_draws = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        s_d = rng.multivariate_normal(s_last, P_last + 1e-10 * np.eye(3))
        x_may = phi * s_d[0] + rng.standard_normal()  # Q=1
        x_jun = phi * x_may + rng.standard_normal()
        f_q2 = (s_d[0] + x_may + x_jun) / 3.0
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth_draws[d] = coef_d[0] + coef_d[1] * f_q2 + rng.normal(0, sigma_b)

    summary = dict(
        modelo="M2_dfm_robusto", phi=float(phi), lam=fit["lam"], R=fit["R"],
        a_bridge=float(coef_b[0]), b_bridge=float(coef_b[1]), sigma_bridge=float(sigma_b),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)), p97_5=float(np.percentile(growth_draws, 97.5)),
        p10=float(np.percentile(growth_draws, 10)), p90=float(np.percentile(growth_draws, 90)),
    )
    print("\n=== M2 (DFM robusto): pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    old = np.load(OUTPUT / "M2_growth_draws.npy")
    print(f"\nComparacion: M2 (indice simple) mediana={np.median(old):.2f}  "
          f"vs  M2 (DFM robusto) mediana={summary['mediana_pct']:.2f}")

    np.save(OUTPUT / "M2_dfm_robusto_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M2_dfm_robusto_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, fit=fit, bridge_coef=coef_b, bridge_sigma=sigma_b,
                          factor=factor_m), f)
    print(f"\nGuardado: {OUTPUT / 'M2_dfm_robusto_growth_draws.npy'}")


if __name__ == "__main__":
    main()
