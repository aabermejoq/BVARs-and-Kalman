"""
M3 (parte 2/3): Gradient Boosting sobre indicadores de consumo con tarjetas.

Misma disciplina que el puente LP de M2: se predice el RESIDUO del VAR M0
(no IGAE_mom directo) a partir de TARJETAS, ANTAD, AUTOS, IMSS (conjunto de
consumo de alta frecuencia), y se valida por LOO-CV antes de usarse. Si el
LOO-CV R^2 no supera al puente lineal (M2), se reporta honestamente y se
pondera en consecuencia dentro del ensamble de M3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
vintage = import_module("01_vintage_calendar")

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

GBM_VARS = ["TARJETAS_mom", "ANTAD_mom", "AUTOS_mom", "IMSS_mom"]
NOWCAST_MONTHS = [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")]
CUTOFF = pd.Timestamp("2020-06-15")


def usable_gbm_vars(month, cutoff=CUTOFF):
    month_end = month + pd.offsets.MonthEnd(0)
    out = []
    for var in GBM_VARS:
        raw = var.replace("_mom", "")
        lag = vintage.VINTAGE_CALENDAR[raw].lag_dias
        if month_end + pd.Timedelta(days=lag) <= cutoff:
            out.append(var)
    return out


def build_residual_target(panel, fit0, window_end=pd.Timestamp("2019-12-31")):
    X_design, Y_design = m0.build_design(panel[m0.VARS].loc[:window_end].dropna(), m0.P)
    dates_design = panel[m0.VARS].loc[:window_end].dropna().index[m0.P:]
    i_igae = m0.VARS.index("IGAE_mom")
    b_igae = fit0["equations"]["IGAE_mom"]["b_post"]
    resid = Y_design[:, i_igae] - X_design @ b_igae
    return pd.Series(resid, index=dates_design)


def fit_gbm_bridge(panel, fit0, usable_vars, window_end=pd.Timestamp("2019-12-31")):
    resid_s = build_residual_target(panel, fit0, window_end)
    df_fast = panel[usable_vars].dropna()
    common = resid_s.index.intersection(df_fast.index)
    y = resid_s.loc[common].values
    Xf = df_fast.loc[common].values
    n = len(y)

    kf = KFold(n_splits=min(10, n), shuffle=True, random_state=0)
    grid = [
        dict(n_estimators=30, max_depth=2, learning_rate=0.05),
        dict(n_estimators=50, max_depth=2, learning_rate=0.03),
        dict(n_estimators=30, max_depth=1, learning_rate=0.05),
    ]
    best_score, best_params = -np.inf, grid[0]
    for params in grid:
        model = GradientBoostingRegressor(random_state=0, **params)
        oof = cross_val_predict(model, Xf, y, cv=kf)
        ss_res = np.sum((y - oof) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_score:
            best_score, best_params = r2, params

    model = GradientBoostingRegressor(random_state=0, **best_params).fit(Xf, y)
    print(f"GBM tarjetas: params={best_params}  CV10 R^2={best_score:.4f}  n={n}  "
          f"vars={usable_vars}")
    return model, best_score


def run():
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    df = panel[m0.VARS].dropna()
    est_df = df.loc[:m0.LAST_OFFICIAL_MONTH]
    fit0 = m0.fit_bvar_minnesota(est_df)

    results = {}
    for month in NOWCAST_MONTHS:
        uv = usable_gbm_vars(month)
        model, r2 = fit_gbm_bridge(panel, fit0, uv)
        x_now = panel.loc[[month], uv].values
        pred_resid = model.predict(x_now)[0]
        results[month] = dict(model=model, cv_r2=r2, usable_vars=uv, pred_resid=pred_resid)
        print(f"  Nowcast GBM residuo en {month.date()}: {pred_resid:.4f}")

    out = pd.DataFrame(
        {m.date(): dict(cv_r2=r["cv_r2"], pred_resid=r["pred_resid"]) for m, r in results.items()}
    ).T
    out.to_csv(PROC / "M3_gbm_tarjetas.csv")
    return results


if __name__ == "__main__":
    run()
