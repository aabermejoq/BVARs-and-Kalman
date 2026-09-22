"""
19_M2_particulas_lp.py

M2 = BVAR Minnesota + Filtro de Particulas (Gordon, Salmond & Smith 1993,
bootstrap/SIR) + Lenza-Primiceri (reescalamiento con decaimiento).

Implementa la critica de la diapositiva 7: en vez de la medicion LINEAL
y_t = Lambda*x_t + eps_t (que asume que la relacion senal-estado sigue
siendo la del punto normal S-bar aun durante un choque extremo), se usa
una funcion g(x_t) NO LINEAL que se aplana en los extremos -- exactamente
el dibujo de la diapositiva. Un filtro de Kalman (que asume linealidad y
normalidad exacta) ya no es el filtro optimo bajo esta medicion no lineal;
se usa un filtro de particulas bootstrap.

Se prueban 2 formas de g (ambas monotonas, lineales cerca de 0, que se
"aplanan" para valores extremos del estado -- igual que el dibujo):

  g-tanh:    g(x; c) = c * tanh(x / c)          (SATURA por completo)
  g-asinh:   g(x; c) = c * asinh(x / c)         (se comprime pero no
                                                  satura del todo)

c se calibra como 1.5 veces la desviacion estandar historica del factor
(pre-2020), para que la zona "lineal" cubra la variacion normal y el
aplanamiento empiece justo donde empiezan los valores atipicos.

Se comparan via la verosimilitud marginal del filtro de particulas sobre
el BACKCAST (ultimos 6 meses antes del corte, sin tocar datos futuros) y
se elige la de mejor ajuste.

LP: mismo criterio que 06c/07c -- s0 medido del residuo de 2T20 bajo el
modelo limpio, rho=0.8^3=0.512 (metodologia del paper, no sus cifras).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "models_agosto"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
panelmod = import_module("17_agosto_panel_amplio")

N_PARTICLES = 2000
N_DRAWS = 10_000
BACKCAST_MONTHS = 6
RHO_QUARTERLY = 0.8 ** 3


def g_tanh(x, c):
    return c * np.tanh(x / c)


def g_asinh(x, c):
    return c * np.arcsinh(x / c)


G_FORMS = {"tanh": g_tanh, "asinh": g_asinh}


def systematic_resample(weights, rng):
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def particle_filter(panel, cols_available, lam, R, phi, g_func, c, Q_scale_series, n_particles=N_PARTICLES, seed=42):
    T = len(panel)
    rng = np.random.default_rng(seed)
    particles = rng.normal(0, 1.0, n_particles)  # x_{t-1}, x_{t-2} no se necesitan explicitamente:
    particles_lag1 = np.zeros(n_particles)        # se trackean por separado para el puente trimestral
    particles_lag2 = np.zeros(n_particles)
    weights = np.ones(n_particles) / n_particles

    loglik_total, loglik_backcast = 0.0, 0.0
    filtered_mean = np.zeros(T)
    filtered_particles_hist = np.zeros((T, n_particles))

    for t in range(T):
        Q_t = Q_scale_series[t]
        particles = phi * particles + np.sqrt(Q_t) * rng.standard_normal(n_particles)

        avail = cols_available[t]
        if avail:
            log_w = np.zeros(n_particles)
            for c_name in avail:
                if c_name == "PIB_q":
                    pred = (particles + particles_lag1 + particles_lag2) / 3.0
                else:
                    pred = lam[c_name] * g_func(particles, c)
                y = panel.iloc[t][c_name]
                log_w += norm.logpdf(y, loc=pred, scale=np.sqrt(R[c_name]))
            m = log_w.max()
            w_unnorm = np.exp(log_w - m)
            lik_t = np.log(w_unnorm.mean()) + m  # log de la verosimilitud marginal de este periodo
            loglik_total += lik_t
            if t >= T - BACKCAST_MONTHS:
                loglik_backcast += lik_t
            weights = w_unnorm / w_unnorm.sum()

            ess = 1.0 / np.sum(weights ** 2)
            if ess < n_particles / 2:
                idx = systematic_resample(weights, rng)
                particles, particles_lag1, particles_lag2 = particles[idx], particles_lag1[idx], particles_lag2[idx]
                weights = np.ones(n_particles) / n_particles

        filtered_mean[t] = np.sum(weights * particles)
        filtered_particles_hist[t] = particles
        particles_lag2 = particles_lag1.copy()
        particles_lag1 = particles.copy()

    return dict(filtered_mean=filtered_mean, particles_hist=filtered_particles_hist,
                final_particles=particles, final_weights=weights,
                loglik_total=loglik_total, loglik_backcast=loglik_backcast)


def main():
    panel, monthly_cols, pib_qoq = panelmod.build_panel_agosto()
    params = panelmod.clean_params(panel, monthly_cols)
    phi, lam, R = params["phi"], params["lam"], params["R"]
    cols_available = [[c for c in panel.columns if not np.isnan(panel.iloc[t][c])] for t in range(len(panel))]

    with open(ROOT / "output" / "models" / "M0_fit.pkl", "rb") as f:
        m0 = pickle.load(f)
    normal_sd = 100 * np.sqrt(m0["fit"]["Sigma_hat"][0, 0])

    # s0 (LP) medido del residuo REAL de 2T20 bajo el modelo (proyeccion simple con phi)
    factor_hist_proxy = np.zeros(len(panel))  # proxy inicial con Q=1 constante, para medir el residuo de 2T20
    Q_flat = np.ones(len(panel))
    pf_proxy = particle_filter(panel, cols_available, lam, R, phi, g_tanh, 1e6, Q_flat, n_particles=500, seed=1)
    idx_2t20 = panel.index.get_loc(pd.Timestamp("2020-04-01"))
    resid_2t20 = pf_proxy["filtered_mean"][idx_2t20] - phi * pf_proxy["filtered_mean"][idx_2t20 - 1]
    s0 = max(abs(resid_2t20) / 1.0, 1.0) ** 2  # multiplicador de VARIANZA
    print(f"s0 (LP, medido del residuo de 2T20 en el factor) = {s0:.2f}")

    Q_series = np.ones(len(panel))
    idx_2t20_pos = panel.index.get_loc(pd.Timestamp("2020-04-01"))
    for j, t in enumerate(range(idx_2t20_pos, len(panel))):
        Q_series[t] = 1 + (s0 - 1) * RHO_QUARTERLY ** j

    print("\nComparando formas de g(x)...")
    results = {}
    for name, g_func in G_FORMS.items():
        pre_2020 = panel.loc[:"2019-12-01"]
        c = 1.5 * pre_2020.drop(columns=["PIB_q"]).std().mean() / np.mean(list(lam.values()))
        c = max(c, 1.0)
        res = particle_filter(panel, cols_available, lam, R, phi, g_func, c, Q_series, seed=7)
        results[name] = res
        print(f"  g-{name} (c={c:.2f}): loglik total={res['loglik_total']:.1f}  "
              f"loglik backcast ({BACKCAST_MONTHS}m)={res['loglik_backcast']:.1f}")

    best_g = max(results, key=lambda k: results[k]["loglik_backcast"])
    print(f"  -> Mejor forma no lineal: g-{best_g}")

    best = results[best_g]
    factor_hist = pd.Series(best["filtered_mean"], index=panel.index)
    factor_q = factor_hist.resample("QS").mean()
    factor_q_n = factor_hist.resample("QS").count()
    factor_q = factor_q[factor_q_n >= 2]
    bridge_df = pd.concat([pib_qoq.rename("pib"), factor_q.rename("f")], axis=1, sort=True).dropna()
    Xb = np.column_stack([np.ones(len(bridge_df)), bridge_df["f"].values])
    yb = bridge_df["pib"].values
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    sigma_b = (yb - Xb @ coef_b).std(ddof=2)

    h_months = pd.Timestamp("2020-07-01").to_period("M").ordinal - panel.index[-1].to_period("M").ordinal
    rng = np.random.default_rng(2101)
    final_particles = best["final_particles"]
    final_weights = best["final_weights"]
    growth = np.zeros(N_DRAWS)
    idx_particles = rng.choice(len(final_particles), size=N_DRAWS, p=final_weights)
    j_next = len(panel) - idx_2t20_pos
    for d in range(N_DRAWS):
        x0 = final_particles[idx_particles[d]]
        xs = [x0]
        for j in range(h_months):
            Q_j = 1 + (s0 - 1) * RHO_QUARTERLY ** (j_next + j)
            xs.append(phi * xs[-1] + np.sqrt(Q_j) * rng.standard_normal())
        f_q = np.mean(xs[-3:]) if len(xs) >= 3 else np.mean(xs)
        coef_d = rng.multivariate_normal(coef_b, sigma_b ** 2 * np.linalg.inv(Xb.T @ Xb))
        growth[d] = coef_d[0] + coef_d[1] * f_q + rng.normal(0, sigma_b)

    print(f"\n=== M2 (Particulas-{best_g} + LP, panel amplio): pronostico 3T20 ===")
    print(f"Mediana={np.median(growth):.2f}%  IC95%=[{np.percentile(growth,2.5):.2f}, {np.percentile(growth,97.5):.2f}]")

    np.save(OUTPUT / "M2_particulas_lp_growth_draws.npy", growth)
    with open(OUTPUT / "M2_particulas_lp_fit.pkl", "wb") as f:
        pickle.dump(dict(best_g=best_g, s0=s0, coef_b=coef_b, sigma_b=sigma_b,
                          factor_hist=factor_hist, final_particles=final_particles,
                          final_weights=final_weights, phi=phi), f)
    print(f"Guardado: {OUTPUT / 'M2_particulas_lp_growth_draws.npy'}")


if __name__ == "__main__":
    main()
