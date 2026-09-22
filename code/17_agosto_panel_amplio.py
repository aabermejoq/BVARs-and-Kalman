"""
17_agosto_panel_amplio.py

Panel amplio (9 series, sin IGAE) y parametros DFM "limpios" (estimados
solo con datos hasta dic-2019) para el corte de AGOSTO/3T20. Punto de
partida compartido por los nuevos M1, M2 y M3 (BVAR Minnesota + Kalman/
Particulas + SV/LP [+ escenarios]).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m2mod = import_module("07b_M2_dfm_robusto")
m3mod = import_module("08b_M3_dfm_robusto")

CLEAN_CUTOFF = pd.Timestamp("2019-12-01")
ASOF = pd.Timestamp("2020-08-15")


def build_panel_agosto():
    import pickle
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    idx = pd.date_range("1993-01-01", "2020-07-01", freq="MS")
    panel = pd.DataFrame(index=idx)
    panel["ActividadIndustrial"] = yoy(series["otros"]["ActividadIndustrial"]).reindex(idx)
    panel["FBCF"] = yoy(series["otros"]["FBCF"]).reindex(idx)
    panel["IMCP"] = yoy(series["Consumo"]["IMCP"]).reindex(idx)
    panel["Exportaciones"] = yoy(series["Balanza"]["Exportaciones"]).reindex(idx)
    panel["Importaciones"] = yoy(series["Balanza"]["Importaciones"]).reindex(idx)
    panel["IMSS"] = yoy(series["IMSS"]["IMSS_empleos"]).reindex(idx)
    panel["ANTAD"] = yoy(series["Consumo"]["ANTAD"]).reindex(idx)
    panel["AUTOS"] = yoy(series["Consumo"]["AUTOS"]).reindex(idx)
    fred = pd.read_csv(EXTERNAL / "fred_us_benchmarks.csv", parse_dates=["date"]).set_index("date")
    indpro = fred["INDPRO"].dropna()
    indpro = indpro[indpro.index <= pd.Timestamp("2020-06-01")]
    panel["INDPRO_EEUU"] = yoy(indpro).reindex(idx)

    panel.loc[panel.index > pd.Timestamp("2020-05-01"), ["ActividadIndustrial", "FBCF", "IMCP"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-06-01"), ["Exportaciones", "Importaciones", "INDPRO_EEUU"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-07-01"), ["IMSS", "ANTAD", "AUTOS"]] = np.nan

    monthly_cols = list(panel.columns)

    core_agosto_path = ROOT / "output" / "models_agosto" / "core_info.pkl"
    with open(core_agosto_path, "rb") as f:
        info_agosto = pickle.load(f)
    core = info_agosto["core"]
    pib_qoq = 100 * core["log_PIB"].diff(1)
    pib_series = pd.Series(index=panel.index, dtype=float)
    for dte, val in pib_qoq.items():
        close_month = dte + pd.DateOffset(months=2)
        if close_month in pib_series.index:
            pib_series[close_month] = val
    panel["PIB_q"] = pib_series

    return panel, monthly_cols, pib_qoq


def clean_params(panel, monthly_cols):
    """Cargas/phi/R estimados SOLO con datos hasta dic-2019 (mismo
    principio de 07c: evita que 2020 distorsione los parametros)."""
    panel_clean = panel.loc[:CLEAN_CUTOFF]
    fit_clean = m3mod.em_dfm_general(panel_clean, monthly_cols, verbose=False)
    return dict(phi=fit_clean["phi"], lam=fit_clean["lam"], R=fit_clean["R"])
