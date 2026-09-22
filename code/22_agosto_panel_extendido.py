"""
22_agosto_panel_extendido.py

Panel ampliado para agosto/3T20, incorporando los hallazgos de la
auditoria de variables faltantes:

  - Ya en la base pero NUNCA usadas: TasaDesempleo (cambio), TIIE (nivel),
    TC (depreciacion a/a). Historia completa verificada (2005/2006/1992 en
    adelante), disponibles ~1 mes de rezago (mismo trato que ANTAD/AUTOS)
    salvo TIIE/TC que son diarias, practicamente sin rezago.
  - Descargadas pero nunca incorporadas: ICSA (EEUU, solicitudes semanales
    de seguro de desempleo) -- re-descargada con el corte extendido a
    agosto (antes solo llegaba a mayo).
  - Nuevas: Google Trends (via pytrends, funciona en este entorno) para
    "reapertura" y "despidos" -- sustituye a GDELT como componente de
    mineria de texto/ML por decision explicita del usuario: mismo
    proposito, pero con HISTORIA REAL (2017 en adelante, se pudo pedir
    resolucion semanal) que permite validar con el mismo rigor (LOO) que
    el resto del ejercicio, cosa que GDELT no permite sin descargar >100GB
    de archivos crudos.
  - Mobility (retail_and_recreation) y OxCGRT (StringencyIndex): mismo
    tratamiento de 21_agosto_mobility_stringency.py.

TRATAMIENTO EN 2 NIVELES (importante, ver discusion con el usuario):
  Nivel 1 (entran a la estimacion EM conjunta "limpia", hasta dic-2019):
    las 9 series originales + TIIE + TC + TasaDesempleo_cambio + ICSA +
    Trends_reapertura/despidos -- TODAS tienen algo de traslape con la
    ventana limpia 2017-2019 (o mas atras), asi que su carga SI se puede
    estimar con el mismo principio de "parametros limpios" usado en todo
    el ejercicio.
  Nivel 2 (mobility_retail, stringency): NO tienen NINGUN dato antes de
    2020 (Mobility publica desde feb-2020, OxCGRT desde ene-2020) -- CERO
    traslape con la ventana limpia. No se pueden meter a la misma EM
    conjunta (no hay como estimar su carga con datos "limpios", literal no
    existen). Se calibran por separado, via una regresion auxiliar contra
    el factor YA extraido por el DFM de Nivel 1, usando SOLO el traslape
    disponible en 2020 -- una calibracion "dentro del episodio", declarada
    como tal, no una validacion fuera de muestra.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
m3mod = import_module("08b_M3_dfm_robusto")
panelmod_v1 = import_module("17_agosto_panel_amplio")

CLEAN_CUTOFF = pd.Timestamp("2019-12-01")
ASOF = pd.Timestamp("2020-08-15")

NIVEL1_COLS = ["ActividadIndustrial", "FBCF", "IMCP", "Exportaciones", "Importaciones",
               "IMSS", "ANTAD", "AUTOS", "INDPRO_EEUU",
               "TIIE", "TC_dep", "TasaDesempleo_chg", "ICSA",
               "Trends_reapertura", "Trends_despidos"]
NIVEL2_COLS = ["mobility_retail", "stringency"]


def build_panel_nivel1():
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

    # --- Nuevas series "duras", ya en la base, nunca usadas ---
    panel["TIIE"] = series["TIIE"]["TIIE"].resample("MS").mean().reindex(idx)
    tc_m = series["TC"]["TC"].resample("MS").mean()
    panel["TC_dep"] = yoy(tc_m).reindex(idx)
    panel["TasaDesempleo_chg"] = series["desempleo"]["TasaDesempleo"].diff().reindex(idx)

    icsa = pd.read_csv(EXTERNAL / "icsa_agosto.csv", parse_dates=["date"]).set_index("date")["ICSA"]
    icsa_m = icsa.resample("MS").mean()
    panel["ICSA"] = yoy(icsa_m).reindex(idx)

    # --- Google Trends (sustituye a GDELT, historia real desde 2017) ---
    trends = pd.read_csv(INTERIM / "trends_reapertura_despidos_2017_2020agosto.csv",
                          index_col=0, parse_dates=True)
    trends_m = trends.resample("MS").mean()
    panel["Trends_reapertura"] = trends_m["reapertura"].reindex(idx)
    panel["Trends_despidos"] = trends_m["despidos"].reindex(idx)

    # --- Rezagos de publicacion (real-time cutoff 15-ago-2020) ---
    panel.loc[panel.index > pd.Timestamp("2020-05-01"), ["ActividadIndustrial", "FBCF", "IMCP"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-06-01"), ["Exportaciones", "Importaciones", "INDPRO_EEUU"]] = np.nan
    panel.loc[panel.index > pd.Timestamp("2020-07-01"), ["IMSS", "ANTAD", "AUTOS", "TasaDesempleo_chg"]] = np.nan
    # TIIE/TC: diarias, practicamente sin rezago -> disponibles hasta el mes del corte (agosto, parcial)
    # ICSA: semanal, rezago ~5 dias -> disponible hasta agosto (parcial)
    # Trends: rezago minimo -> disponible hasta agosto (parcial, semana del 9-ago)
    for col in ["TIIE", "TC_dep", "ICSA", "Trends_reapertura", "Trends_despidos"]:
        panel.loc[panel.index > pd.Timestamp("2020-08-01"), col] = np.nan

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

    return panel, NIVEL1_COLS, pib_qoq


def clean_params_nivel1(panel, monthly_cols):
    panel_clean = panel.loc[:CLEAN_CUTOFF]
    fit_clean = m3mod.em_dfm_general(panel_clean, monthly_cols, verbose=False)
    return dict(phi=fit_clean["phi"], lam=fit_clean["lam"], R=fit_clean["R"])


def build_nivel2(panel_idx):
    """mobility_retail y stringency: SIN traslape con la ventana limpia,
    se cargan aparte (no entran a la EM conjunta de Nivel 1)."""
    mob = pd.read_csv(EXTERNAL / "google_mobility_benchmarks_agosto.csv", parse_dates=["date"])
    mob_mx = mob[mob["country_code"] == "MX"].set_index("date")
    ox = pd.read_csv(EXTERNAL / "oxcgrt_benchmarks_agosto.csv", parse_dates=["Date"])
    ox_mx = ox[ox["CountryCode"] == "MEX"].set_index("Date")

    out = pd.DataFrame(index=panel_idx)
    out["mobility_retail"] = mob_mx["retail_and_recreation_percent_change_from_baseline"].resample("MS").mean()
    out["stringency"] = ox_mx["StringencyIndex_Average"].resample("MS").mean()
    return out


def main():
    panel, cols, pib_qoq = build_panel_nivel1()
    print(f"Panel Nivel 1: {len(cols)} series -> {cols}")
    print("\nCobertura (primer/ultimo dato no nulo):")
    for c in cols:
        s = panel[c].dropna()
        print(f"  {c:20s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

    print("\nEstimando parametros limpios (EM, hasta dic-2019, panel Nivel 1)...")
    params = clean_params_nivel1(panel, cols)
    print("Cargas (lambda):")
    for k, v in sorted(params["lam"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:20s} {v:+.4f}")
    print(f"phi={params['phi']:.4f}")

    nivel2 = build_nivel2(panel.index)
    print("\nNivel 2 (sin traslape limpio, calibracion posterior):")
    for c in NIVEL2_COLS:
        s = nivel2[c].dropna()
        print(f"  {c:20s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

    with open(INTERIM / "panel_extendido_agosto.pkl", "wb") as f:
        pickle.dump(dict(panel=panel, cols=cols, pib_qoq=pib_qoq, params=params, nivel2=nivel2), f)
    print(f"\nGuardado: {INTERIM / 'panel_extendido_agosto.pkl'}")


if __name__ == "__main__":
    main()
