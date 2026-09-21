"""
06_M1_volatility.py

FASE 7: M1 resuelve P1 (observaciones extremas / choque de volatilidad).

Decision metodologica (justificada, no arbitraria): de las opciones
evaluadas (volatilidad estocastica, errores-t, mixture innovations), se
elige un factor de VOLATILIDAD ESTOCASTICA COMUN de un solo factor:

    Sigma_t = exp(h_t) * Sigma_hat        h_t = c + rho*h_{t-1} + eta_t

manteniendo los COEFICIENTES del VAR FIJOS en la media posterior de M0.
Esta eleccion aisla deliberadamente el canal de "cambio en la varianza" del
canal de "cambio en los coeficientes" (item 3 de la lista del usuario): M1
NO reestima A_1..A_p, solo relaja el supuesto de varianza constante.

Se descarta el enfoque de errores-t (fat tails ESTATICOS) porque no
distingue un choque TEMPORAL de volatilidad de una propiedad permanente de
la distribucion, y no permite que la volatilidad de 1T20 (que YA muestra
senales de tension por el inicio de COVID a mediados de marzo) se propague
con persistencia hacia 2T20 -- exactamente el mecanismo de "cambio en la
varianza" que solicito el usuario. Se descarta mixture-of-regimes porque no
hay evidencia (todavia) de que el choque sea un cambio de regimen
PERMANENTE en lugar de una innovacion extrema transitoria: la SV captura
persistencia sin comprometerse con una ruptura estructural.

Estimacion de h_t: se usa el metodo aproximado de Harvey-Ruiz-Shephard
(1994) -- transformar el "tamano del choque" a escala log y ajustarlo con
un AR(1) por MCO. Es una aproximacion (no un muestreador Gibbs completo de
Kim-Shephard-Chib), declarada explicitamente como simplificacion razonable
para el alcance de esta presentacion.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output" / "models"

N_DRAWS = 10_000
RNG_SEED = 22345


def main():
    with open(OUTPUT / "M0_fit.pkl", "rb") as f:
        m0 = pickle.load(f)
    fit, VARS, P = m0["fit"], m0["vars"], m0["p"]
    n = len(VARS)

    core = pd.read_csv(INTERIM / "core_quarterly_panel.csv", index_col=0, parse_dates=True)
    data = core[VARS].values

    resid = fit["resid"]  # (T-p) x n, residuos evaluados en la media posterior de M0
    Sigma_hat = fit["Sigma_hat"]
    Sigma_inv = np.linalg.inv(Sigma_hat)

    # "Tamano del choque" por trimestre: forma cuadratica tipo Mahalanobis,
    # ~ chi2(n) bajo el supuesto nulo de volatilidad constante Gaussiana.
    mahal = np.einsum("ti,ij,tj->t", resid, Sigma_inv, resid)
    log_scale = np.log(mahal / n)  # log(1) = 0 bajo volatilidad "normal"

    print("Ultimos 6 trimestres del factor de volatilidad log-realizada (log_scale):")
    dates = core.index[-len(log_scale):]
    for dte, ls in list(zip(dates, log_scale))[-6:]:
        print(f"  {dte.date()}: {ls:+.3f}  (multiplicador de varianza = {np.exp(ls):.2f}x)")

    # AR(1) de h_t por MCO (aproximacion Harvey-Ruiz-Shephard).
    h = log_scale
    X_ar = np.column_stack([h[:-1], np.ones(len(h) - 1)])
    y_ar = h[1:]
    coef, *_ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
    rho, c_ar = coef
    eta = y_ar - X_ar @ coef
    sigma_eta = eta.std(ddof=2)
    rho = float(np.clip(rho, -0.95, 0.95))  # estabilidad numerica

    print(f"\nAR(1) de la volatilidad log-realizada: rho={rho:.3f}, c={c_ar:.3f}, sigma_eta={sigma_eta:.3f}")

    h_T = log_scale[-1]  # volatilidad realizada en 1T20 (el trimestre en que inicio COVID en Mexico)
    print(f"h_T (1T20, ya con datos post-11-mar-2020 en el trimestre) = {h_T:+.3f} "
          f"(multiplicador {np.exp(h_T):.2f}x)")

    # Simulacion: misma incertidumbre de parametros que M0 (coeficientes
    # FIJOS del VAR, solo se agrega SV a la covarianza de los choques).
    rng = np.random.default_rng(RNG_SEED)
    k = n * P + 1
    last_obs = data[-P:][::-1]
    log_pib_t = data[-1, VARS.index("log_PIB")]
    growth_draws = np.zeros(N_DRAWS)

    L_hat = np.linalg.cholesky(Sigma_hat + 1e-12 * np.eye(n))

    for d in range(N_DRAWS):
        beta_d = np.zeros((k, n))
        for i in range(n):
            beta_d[:, i] = rng.multivariate_normal(fit["post_mean"][:, i], fit["post_var"][:, :, i])

        h_next = c_ar + rho * h_T + sigma_eta * rng.standard_normal()
        scale = np.exp(h_next / 2.0)  # se aplica a la desv. estandar, no a la varianza

        x = np.concatenate([last_obs[l] for l in range(P)] + [[1.0]])
        eps = scale * (L_hat @ rng.standard_normal(n))
        y_new = x @ beta_d + eps
        growth_draws[d] = 100.0 * (np.exp(y_new[VARS.index("log_PIB")] - log_pib_t) - 1.0)

    summary = dict(
        modelo="M1",
        rho_vol=float(rho), c_vol=float(c_ar), sigma_eta_vol=float(sigma_eta),
        h_T_1T20=float(h_T), multiplicador_h_T=float(np.exp(h_T)),
        mediana_pct=float(np.median(growth_draws)),
        p2_5=float(np.percentile(growth_draws, 2.5)),
        p10=float(np.percentile(growth_draws, 10)),
        p25=float(np.percentile(growth_draws, 25)),
        p75=float(np.percentile(growth_draws, 75)),
        p90=float(np.percentile(growth_draws, 90)),
        p97_5=float(np.percentile(growth_draws, 97.5)),
    )

    print("\n=== M1: BVAR + volatilidad estocastica comun, pronostico 2T20 (%q/q) ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    np.save(OUTPUT / "M1_growth_draws.npy", growth_draws)
    with open(OUTPUT / "M1_fit.pkl", "wb") as f:
        pickle.dump(dict(summary=summary, rho=rho, c_ar=c_ar, sigma_eta=sigma_eta, h_T=h_T), f)

    print(f"\nGuardado: {OUTPUT / 'M1_growth_draws.npy'} y {OUTPUT / 'M1_fit.pkl'}")

    m0_draws = np.load(OUTPUT / "M0_growth_draws.npy")
    print(f"\nComparacion de amplitud IC95%: M0=[{np.percentile(m0_draws,2.5):.2f}, "
          f"{np.percentile(m0_draws,97.5):.2f}]  vs  M1=[{summary['p2_5']:.2f}, {summary['p97_5']:.2f}]")


if __name__ == "__main__":
    main()
