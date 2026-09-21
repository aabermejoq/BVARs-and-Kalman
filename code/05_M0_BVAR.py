"""
05_M0_BVAR.py

FASE 6: M0, el benchmark macroeconomico.

VAR pequeno de economia abierta: y = [log_PIB, log_INPC, TIIE, log_TC]',
p=4 rezagos (un ano de dinamica trimestral), con PRIOR MINNESOTA clasico
(Litterman 1986): cada ecuacion se trata por separado, con media a priori
de caminata aleatoria en el primer rezago propio (delta_i=1, estandar en la
literatura de BVARs incluso para variables que podrian ser estacionarias
como la TIIE -- es la simplificacion "de libro de texto" del prior
Minnesota) y encogimiento hacia cero de los demas coeficientes, mas rapido
para rezagos mayores y para coeficientes cruzados.

theta = (c, A_1, ..., A_p, Sigma):
  - c, A_1..A_p: posterior Normal por ecuacion (conjugado Normal-Normal,
    con la varianza de la verosimilitud FIJADA al residual de un AR(p)
    univariado -- la convencion clasica de Litterman que evita estimar
    Sigma conjuntamente con los coeficientes).
  - Sigma: covarianza MUESTRAL de los residuos evaluados en la media
    posterior de cada ecuacion (para poder simular shocks correlacionados
    en el pronostico conjunto).

Muestra: 2006Q1-2020Q1 (57 trimestres). La TIIE de Fondeo a 1 dia solo
existe desde 2006 en el archivo -- ver 03_transformations.py para la
justificacion completa de por que la muestra NO llega hasta 1993.

Simulacion predictiva: Monte Carlo de theta ~ posterior, propagando 1 paso
adelante (h=1, objetivo 2020Q2) con shocks ~ N(0,Sigma), para obtener
p(Y_2020Q2 | I_2020Q1) de forma empirica (no una formula cerrada).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"
OUTPUT.mkdir(parents=True, exist_ok=True)

VARS = ["log_PIB", "log_INPC", "TIIE", "log_TC"]
P = 4  # rezagos
N_DRAWS = 10_000
RNG_SEED = 12345

# Hiperparametros Minnesota (valores estandar de Litterman/BGR, no ajustados
# usando el desempeno de 2T20 -- fijados por convencion de la literatura).
LAMBDA1 = 0.2   # tightness general
LAMBDA2 = 0.5   # tightness cruzada (relativa a la propia)
LAMBDA3 = 1.0   # decaimiento por rezago
CONST_VAR = 1e6  # varianza (grande = poco informativa) del intercepto


def fit_ar_sigma(y, p):
    """Ajusta un AR(p) univariado por OLS y devuelve la desviacion estandar
    residual (escala de Litterman, sigma_i)."""
    T = len(y)
    X = np.column_stack([y[p - l - 1: T - l - 1] for l in range(p)] + [np.ones(T - p)])
    Y = y[p:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return resid.std(ddof=X.shape[1])


def build_design(data, p):
    """Y: (T-p) x n :: X: (T-p) x (n*p+1) con [lag1_allvars,...,lagp_allvars, const]"""
    T, n = data.shape
    rows = []
    for t in range(p, T):
        row = []
        for l in range(1, p + 1):
            row.extend(data[t - l, :])
        row.append(1.0)
        rows.append(row)
    X = np.array(rows)
    Y = data[p:, :]
    return Y, X


def minnesota_prior_for_equation(i, n, p, sigma):
    """Devuelve (mean, var_diag) del prior para la ecuacion i, en el orden
    de columnas de X: [lag1_var1..lag1_varn, lag2_var1..lag2_varn, ...,
    const]. sigma: array (n,) escalas de Litterman."""
    k = n * p + 1
    mean = np.zeros(k)
    var = np.zeros(k)
    col = 0
    for l in range(1, p + 1):
        for j in range(n):
            if j == i and l == 1:
                mean[col] = 1.0  # caminata aleatoria en el propio primer rezago
            if j == i:
                var[col] = (LAMBDA1 / (l ** LAMBDA3)) ** 2
            else:
                var[col] = (LAMBDA1 * LAMBDA2 / (l ** LAMBDA3)) ** 2 * (sigma[i] / sigma[j]) ** 2
            col += 1
    mean[-1] = 0.0
    var[-1] = CONST_VAR
    return mean, var


def fit_minnesota_bvar(data, p):
    T, n = data.shape
    Y, X = build_design(data, p)
    sigma = np.array([fit_ar_sigma(data[:, i], p) for i in range(n)])

    post_mean = np.zeros((X.shape[1], n))
    post_var = np.zeros((X.shape[1], X.shape[1], n))

    for i in range(n):
        m0, v0 = minnesota_prior_for_equation(i, n, p, sigma)
        V0_inv = np.diag(1.0 / v0)
        lik_var = sigma[i] ** 2  # plug-in, convencion clasica de Litterman
        V_post = np.linalg.inv(V0_inv + (X.T @ X) / lik_var)
        m_post = V_post @ (V0_inv @ m0 + (X.T @ Y[:, i]) / lik_var)
        post_mean[:, i] = m_post
        post_var[:, :, i] = V_post

    resid = Y - X @ post_mean
    Sigma_hat = np.cov(resid.T, ddof=X.shape[1])

    return dict(post_mean=post_mean, post_var=post_var, Sigma_hat=Sigma_hat,
                sigma_ar=sigma, X=X, Y=Y, resid=resid)


def simulate_forecast(fit, last_obs, n, p, h, n_draws, seed=RNG_SEED):
    """Monte Carlo de theta~posterior y propagacion h pasos adelante.
    last_obs: array (p, n) con las p observaciones MAS RECIENTES conocidas
    (orden: last_obs[0] = t, last_obs[1] = t-1, ..., last_obs[p-1] = t-p+1)."""
    rng = np.random.default_rng(seed)
    k = n * p + 1
    draws = np.zeros((n_draws, h, n))

    L = np.linalg.cholesky(fit["Sigma_hat"] + 1e-12 * np.eye(n))

    for d in range(n_draws):
        beta_d = np.zeros((k, n))
        for i in range(n):
            beta_d[:, i] = rng.multivariate_normal(fit["post_mean"][:, i], fit["post_var"][:, :, i])

        hist = last_obs.copy()  # (p, n)
        for step in range(h):
            x = np.concatenate([hist[l] for l in range(p)] + [[1.0]])
            eps = L @ rng.standard_normal(n)
            y_new = x @ beta_d + eps
            draws[d, step, :] = y_new
            hist = np.vstack([y_new, hist[:-1]])

    return draws


def main():
    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    assert core.index.max() == pd.Timestamp("2020-01-01"), "M0 debe estimarse solo hasta 2020Q1."

    data = core[VARS].values
    n = len(VARS)

    fit = fit_minnesota_bvar(data, P)

    last_obs = data[-P:][::-1]  # last_obs[0] = 2020Q1, last_obs[1]=2019Q4, ...
    draws = simulate_forecast(fit, last_obs, n, P, h=1, n_draws=N_DRAWS)

    log_pib_t = data[-1, VARS.index("log_PIB")]
    log_pib_draws_2020Q2 = draws[:, 0, VARS.index("log_PIB")]
    growth_pct = 100.0 * (np.exp(log_pib_draws_2020Q2 - log_pib_t) - 1.0)

    summary = dict(
        modelo="M0",
        n_obs_estimacion=len(data),
        muestra_inicio=str(core.index.min().date()),
        muestra_fin=str(core.index.max().date()),
        mediana_pct=float(np.median(growth_pct)),
        p10=float(np.percentile(growth_pct, 10)),
        p25=float(np.percentile(growth_pct, 25)),
        p75=float(np.percentile(growth_pct, 75)),
        p90=float(np.percentile(growth_pct, 90)),
        p2_5=float(np.percentile(growth_pct, 2.5)),
        p97_5=float(np.percentile(growth_pct, 97.5)),
    )

    print("=== M0: BVAR Minnesota, pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M0_growth_draws.npy", growth_pct)
    with open(OUTPUT / "M0_fit.pkl", "wb") as f:
        pickle.dump(dict(fit=fit, summary=summary, vars=VARS, p=P), f)

    print(f"\nGuardado: {OUTPUT / 'M0_growth_draws.npy'} y {OUTPUT / 'M0_fit.pkl'}")


if __name__ == "__main__":
    main()
