"""
M0: BVAR Minnesota.

VAR(p) sobre [IGAE_mom, ActividadIndustrial_mom, IMSS_mom] con prior
Minnesota (Litterman) aplicado ecuacion por ecuacion como regresion
ridge generalizada (shrinkage propio vs cruzado, decayendo con el rezago),
mas Inverse-Gamma conjugado para la varianza residual de cada ecuacion.
Simplificacion explicita respecto al NIW-BVAR conjunto completo (Banbura,
Giannone y Reichlin 2010): aqui las ecuaciones se estiman de forma
condicionalmente independiente dado sigma_i^2 (enfoque Litterman clasico),
en vez de la verosimilitud matriz-normal conjunta. Para simular hacia
adelante con correlacion contemporanea realista entre residuales, se
impone la correlacion empirica de los residuos OLS via Cholesky sobre los
choques simulados.

Estimado SOLO con informacion dura ya publicada al corte (sin variables
rapidas/puente -- eso es lo que agrega M2). Pronostico multi-mes desde el
ultimo mes con IGAE oficial conocido hasta diciembre 2020.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

VARS = ["IGAE_mom", "ActividadIndustrial_mom", "IMSS_mom"]
P = 3  # orden del VAR
N_DRAWS = 2000
SEED = 12345

# --- Corte de informacion ---------------------------------------------
# Mes de partida: mayo 2020. Corte: 15 dias tras el cierre de mayo -> 15-jun-2020.
# A ese corte, por el calendario de vintage (rezago 56 dias), el ultimo IGAE
# oficial conocido es MARZO 2020 (ver code/01_vintage_calendar.py).
CUTOFF = pd.Timestamp("2020-06-15")
LAST_OFFICIAL_MONTH = pd.Timestamp("2020-03-01")  # ultimo IGAE ya publicado al corte
FORECAST_END = pd.Timestamp("2020-12-01")


def minnesota_prior_var(sigma_i, sigma_j, lag, same_var, lam=0.2, theta=0.5, d=1.0):
    """Varianza prior para el coeficiente de var j en el rezago `lag` de la ecuacion i."""
    if same_var:
        return (lam / (lag ** d)) ** 2
    return (lam * theta / (lag ** d)) ** 2 * (sigma_i / sigma_j) ** 2


def build_design(df: pd.DataFrame, p: int):
    """Y: (T-p) x n ; X: (T-p) x (1+n*p), columnas = [const, var1_lag1..lag_p, var2_lag1..,...]."""
    n = df.shape[1]
    T = df.shape[0]
    rows = []
    Y = []
    for t in range(p, T):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(df.iloc[t - lag].values.tolist())
        rows.append(row)
        Y.append(df.iloc[t].values)
    X = np.array(rows)
    Y = np.array(Y)
    return X, Y


def fit_bvar_minnesota(df: pd.DataFrame, p: int = P, lam: float = 0.2, theta: float = 0.5):
    n = df.shape[1]
    names = list(df.columns)
    sigma = {v: df[v].diff().std() if False else df[v].std() for v in names}
    # sigma_i: escala de cada variable, de un AR(1) residual simple (mas fiel a Minnesota clasico)
    sigma_ar = {}
    for v in names:
        y = df[v].values[1:]
        x = df[v].values[:-1]
        X1 = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        sigma_ar[v] = resid.std(ddof=2)

    X, Y = build_design(df, p)
    k = X.shape[1]  # 1 + n*p

    equations = {}
    for i, vi in enumerate(names):
        b0 = np.zeros(k)  # prior mean 0 (VAR ya en tasas de crecimiento, estacionario)
        V0_diag = np.zeros(k)
        V0_diag[0] = 1e6  # intercepto: prior difuso
        col = 1
        for lag in range(1, p + 1):
            for j, vj in enumerate(names):
                same = vi == vj
                V0_diag[col] = minnesota_prior_var(
                    sigma_ar[vi], sigma_ar[vj], lag, same, lam=lam, theta=theta
                )
                col += 1
        V0 = np.diag(V0_diag)
        V0_inv = np.diag(1.0 / V0_diag)

        y_i = Y[:, i]
        # sigma_i^2 empirico (OLS) para el update conjugado (empirical-Bayes plug-in)
        beta_ols, *_ = np.linalg.lstsq(X, y_i, rcond=None)
        resid_ols = y_i - X @ beta_ols
        sigma2_i = resid_ols.var(ddof=k)

        V_post = np.linalg.inv(V0_inv + X.T @ X / sigma2_i)
        b_post = V_post @ (V0_inv @ b0 + X.T @ y_i / sigma2_i)

        # Inverse-Gamma posterior para sigma2_i (prior debil: nu0=3, s0^2=sigma_ar^2)
        nu0, s0sq = 3, sigma_ar[vi] ** 2
        Tn = len(y_i)
        nu_post = nu0 + Tn
        resid_post = y_i - X @ b_post
        s_post = nu0 * s0sq + resid_post @ resid_post
        equations[vi] = dict(b_post=b_post, V_post=V_post, nu_post=nu_post, s_post=s_post)

    # correlacion empirica de residuales OLS (para choques correlacionados en la simulacion)
    resid_matrix = np.column_stack(
        [Y[:, i] - X @ np.linalg.lstsq(X, Y[:, i], rcond=None)[0] for i in range(n)]
    )
    corr = np.corrcoef(resid_matrix, rowvar=False)

    return dict(equations=equations, names=names, p=p, corr=corr, k=k)


def sample_posterior_draw(fit, rng):
    names = fit["names"]
    betas = {}
    sigmas = {}
    for v in names:
        eq = fit["equations"][v]
        sigma2 = eq["s_post"] / rng.chisquare(eq["nu_post"])
        # aproximacion: covarianza del coef escalada por sigma2 muestreado / sigma2 empirico usado en V_post
        beta = rng.multivariate_normal(eq["b_post"], eq["V_post"])
        betas[v] = beta
        sigmas[v] = np.sqrt(sigma2)
    return betas, sigmas


def simulate_forward(fit, history_df: pd.DataFrame, betas, sigmas, corr, horizon: int, rng):
    """history_df: ultimas p filas observadas (en orden cronologico) para arrancar la simulacion."""
    names = fit["names"]
    p = fit["p"]
    n = len(names)
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(n))
    hist = history_df[names].values[-p:].tolist()  # lista de arrays largo p, cada uno n-vector

    path = []
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(hist[-lag])
        row = np.array(row)
        mean = np.array([betas[v] @ row for v in names])
        z = rng.standard_normal(n)
        shock = L @ z
        y_t = mean + np.array([sigmas[v] for v in names]) * shock
        path.append(y_t)
        hist.append(y_t.tolist())
    return np.array(path)  # horizon x n


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[VARS].dropna()
    # estimacion SOLO con datos hasta el ultimo mes con IGAE oficial conocido al corte
    est_df = df.loc[:LAST_OFFICIAL_MONTH]
    print(f"M0: estimando BVAR Minnesota con {len(est_df)} obs, hasta {est_df.index[-1].date()}")

    fit = fit_bvar_minnesota(est_df)

    horizon_months = pd.date_range(
        LAST_OFFICIAL_MONTH + pd.DateOffset(months=1), FORECAST_END, freq="MS"
    )
    horizon = len(horizon_months)

    rng = np.random.default_rng(SEED)
    draws = np.zeros((N_DRAWS, horizon, len(VARS)))
    for d in range(N_DRAWS):
        betas, sigmas = sample_posterior_draw(fit, rng)
        draws[d] = simulate_forward(fit, est_df, betas, sigmas, fit["corr"], horizon, rng)

    igae_idx = VARS.index("IGAE_mom")
    igae_draws = draws[:, :, igae_idx]  # N_DRAWS x horizon

    out = pd.DataFrame(igae_draws.T, index=horizon_months)
    out.to_csv(PROC / "M0_igae_mom_draws.csv")

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
    summary.to_csv(PROC / "M0_summary.csv")
    print(summary.round(4))
    return summary, igae_draws, horizon_months


if __name__ == "__main__":
    run()
