"""
24_panel_mensual_igae.py

Panel MENSUAL (no trimestral) con IGAE total como variable objetivo,
parametrizado por una fecha de corte arbitraria -- construido para el
ejercicio de 3 cortes de informacion sobre mayo-2020 (ver conversacion):

  Corte 1: 15-jun-2020 (15 dias despues de cerrado mayo)
  Corte 2: 30-jun-2020 (1 mes despues de cerrado mayo)
  Corte 3: 24-jul-2020 (justo ANTES de que salga el IGAE de mayo -- el
           corte usa toda la informacion disponible hasta el 23-jul; el
           IGAE de mayo publicado el 24-jul se usa DESPUES solo para
           comparar contra el pronostico, nunca como insumo)

Calendario de publicacion VERIFICADO con comunicados reales de INEGI/IMSS/
ANTAD/AMIA (no supuesto) -- ver 02_real_time_vintage.py (registro
actualizado) y EXACT_DATES abajo para los meses criticos de este ejercicio,
donde un rezago generico (en dias) podria fallar por 1-2 dias justo en la
fecha de corte que nos importa (en particular el corte 3, definido como
"justo antes de que salga el IGAE").

BUG CORREGIDO (encontrado por el usuario): FBCF e IMCP NO se publican el
mismo dia que IGAE -- van en un comunicado aparte (IMFBCF/IMCPMI), ~13 dias
DESPUES. El registro de 02_real_time_vintage.py ya fue corregido.
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
vintage = import_module("02_real_time_vintage")

TARGET_MONTH = pd.Timestamp("2020-05-01")
CORTE_1 = pd.Timestamp("2020-06-15")
CORTE_2 = pd.Timestamp("2020-06-30")
CORTE_3 = pd.Timestamp("2020-07-23")  # justo antes de la publicacion del 24-jul
IGAE_REAL_MAYO = -2.6  # publicado 24-jul-2020, comunicado 347/20 (verificado)

CLEAN_CUTOFF = pd.Timestamp("2019-12-01")

# Fechas EXACTAS verificadas (comunicados reales descargados de inegi.org.mx)
# para los meses criticos de este ejercicio -- mas precisas que el rezago
# generico del registro, que puede fallar por 1-2 dias en el limite exacto
# de un corte (en particular el corte 3).
EXACT_DATES = {
    "IGAE_ActInd": {  # mismo comunicado, misma fecha
        "2020-02-01": "2020-04-24", "2020-03-01": "2020-05-26",
        "2020-04-01": "2020-06-26", "2020-05-01": "2020-07-24",
        "2020-06-01": "2020-08-26", "2020-07-01": "2020-09-25",
    },
    "FBCF_IMCP": {  # mismo comunicado, misma fecha (13 dias despues de IGAE)
        "2020-02-01": "2020-05-06", "2020-03-01": "2020-06-05",
        "2020-04-01": "2020-07-06", "2020-05-01": "2020-08-06",
        "2020-06-01": "2020-09-07", "2020-07-01": "2020-10-06",
    },
    "Balanza": {
        "2020-04-01": "2020-05-26", "2020-05-01": "2020-06-26",
        "2020-06-01": "2020-07-27",
    },
    "ANTAD": {"2020-05-01": "2020-06-11"},
    "AUTOS": {"2020-05-01": "2020-06-05"},
    "IMSS": {"2020-05-01": "2020-06-12"},
}


def _exact_or_registry(group, ref_month, variable, asof):
    """Usa la fecha EXACTA verificada si esta disponible para ese mes de
    referencia; si no, cae al rezago generico del registro (para meses
    lejos de los cortes criticos, donde 1-2 dias de diferencia no cambia
    el resultado)."""
    key = ref_month.strftime("%Y-%m-01")
    exact = EXACT_DATES.get(group, {}).get(key)
    if exact is not None:
        return pd.Timestamp(exact) <= asof
    return vintage.is_available(variable, ref_month, asof=asof)


def build_monthly_panel(cutoff_date):
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    idx = pd.date_range("1993-01-01", TARGET_MONTH, freq="MS")
    panel = pd.DataFrame(index=idx)

    def apply_cutoff(raw, group, variable):
        s = raw.reindex(idx)
        mask = pd.Series([_exact_or_registry(group, d, variable, cutoff_date) for d in idx], index=idx)
        return s.where(mask)

    panel["ActividadIndustrial"] = apply_cutoff(yoy(series["otros"]["ActividadIndustrial"]),
                                                 "IGAE_ActInd", "ActividadIndustrial")
    panel["FBCF"] = apply_cutoff(yoy(series["otros"]["FBCF"]), "FBCF_IMCP", "FBCF")
    panel["IMCP"] = apply_cutoff(yoy(series["Consumo"]["IMCP"]), "FBCF_IMCP", "IMCP")
    panel["Exportaciones"] = apply_cutoff(yoy(series["Balanza"]["Exportaciones"]), "Balanza", "Exportaciones")
    panel["Importaciones"] = apply_cutoff(yoy(series["Balanza"]["Importaciones"]), "Balanza", "Importaciones")
    panel["IMSS"] = apply_cutoff(yoy(series["IMSS"]["IMSS_empleos"]), "IMSS", "IMSS_empleos")
    panel["ANTAD"] = apply_cutoff(yoy(series["Consumo"]["ANTAD"]), "ANTAD", "ANTAD")
    panel["AUTOS"] = apply_cutoff(yoy(series["Consumo"]["AUTOS"]), "AUTOS", "AUTOS")

    fred = pd.read_csv(EXTERNAL / "fred_us_benchmarks.csv", parse_dates=["date"]).set_index("date")
    indpro = fred["INDPRO"].dropna().resample("MS").mean()
    # INDPRO (FRED) rezago tipico EEUU ~1 mes -> asof - 30 dias como referencia conservadora
    indpro_avail = indpro.index[indpro.index + pd.Timedelta(days=45) <= cutoff_date]
    panel["INDPRO_EEUU"] = yoy(indpro).reindex(idx).where(pd.Series(idx, index=idx).isin(indpro_avail))

    panel["TIIE"] = series["TIIE"]["TIIE"].resample("MS").mean().reindex(idx).where(idx + pd.Timedelta(days=2) <= cutoff_date)
    tc_m = series["TC"]["TC"].resample("MS").mean()
    panel["TC_dep"] = yoy(tc_m).reindex(idx).where(idx + pd.Timedelta(days=2) <= cutoff_date)
    desem = series["desempleo"]["TasaDesempleo"].diff()
    panel["TasaDesempleo_chg"] = desem.reindex(idx).where(idx + pd.Timedelta(days=24) <= cutoff_date)

    icsa = pd.read_csv(EXTERNAL / "icsa_agosto.csv", parse_dates=["date"]).set_index("date")["ICSA"]
    icsa_m = icsa.resample("MS").mean()
    panel["ICSA"] = yoy(icsa_m).reindex(idx).where(idx + pd.Timedelta(days=10) <= cutoff_date)

    trends = pd.read_csv(INTERIM / "trends_reapertura_despidos_2017_2020agosto.csv",
                          index_col=0, parse_dates=True)
    trends_m = trends.resample("MS").mean()
    panel["Trends_reapertura"] = trends_m["reapertura"].reindex(idx).where(idx + pd.Timedelta(days=3) <= cutoff_date)
    panel["Trends_despidos"] = trends_m["despidos"].reindex(idx).where(idx + pd.Timedelta(days=3) <= cutoff_date)

    bmv = pd.read_csv(EXTERNAL / "bmv_agosto.csv", index_col=0, parse_dates=True).iloc[:, 0]
    bmv_m = bmv.resample("MS").mean()
    panel["BMV_yoy"] = yoy(bmv_m).reindex(idx).where(idx + pd.Timedelta(days=2) <= cutoff_date)

    lt_rate = pd.read_csv(EXTERNAL / "mx_ltrate_fred.csv", parse_dates=["observation_date"])
    lt_rate = lt_rate.set_index("observation_date")["IRLTLT01MXM156N"].resample("MS").mean().reindex(idx)
    panel["Spread_tasas"] = (lt_rate.where(idx + pd.Timedelta(days=30) <= cutoff_date) - panel["TIIE"])

    mob = pd.read_csv(EXTERNAL / "google_mobility_benchmarks_agosto.csv", parse_dates=["date"])
    mob_mx = mob[mob["country_code"] == "MX"].set_index("date")
    mob_m = mob_mx["retail_and_recreation_percent_change_from_baseline"].resample("MS").mean().reindex(idx)
    panel["mobility_retail"] = mob_m.where(idx + pd.Timedelta(days=32) <= cutoff_date)

    ox = pd.read_csv(EXTERNAL / "oxcgrt_benchmarks_agosto.csv", parse_dates=["Date"])
    ox_mx = ox[ox["CountryCode"] == "MEX"].set_index("Date")
    ox_m = ox_mx["StringencyIndex_Average"].resample("MS").mean().reindex(idx)
    panel["stringency"] = ox_m.where(idx + pd.Timedelta(days=2) <= cutoff_date)

    # --- IGAE total: variable OBJETIVO (m/m, igual transformacion que usan
    # los comunicados de INEGI para reportar la variacion mensual) ---
    igae_mom = 100 * np.log(series["otros"]["IGAE"] / series["otros"]["IGAE"].shift(1))
    panel["IGAE_target"] = apply_cutoff(igae_mom, "IGAE_ActInd", "IGAE")

    return panel


NIVEL1_COLS = ["ActividadIndustrial", "FBCF", "IMCP", "Exportaciones", "Importaciones",
               "IMSS", "ANTAD", "AUTOS", "INDPRO_EEUU", "TIIE", "TC_dep", "TasaDesempleo_chg",
               "ICSA", "Trends_reapertura", "Trends_despidos", "BMV_yoy", "Spread_tasas"]
NIVEL2_COLS = ["mobility_retail", "stringency"]


def clean_params_nivel1(panel):
    """IGAE_target entra como una serie MENSUAL MAS (H=[lam,0,0], igual que
    cualquier otra) -- NO como 'PIB_q' (esa forma especial de
    em_dfm_general hace el promedio 1/3+1/3+1/3 para BRIDGING trimestral,
    que ya no aplica: target y panel estan a la MISMA frecuencia mensual).
    Que el propio IGAE historico ayude a formar el factor es practica
    estandar de nowcasting (el propio NowTIM de Banxico usa el PIB/IGAE
    como insumo #1 de su factor), no la circularidad que se evito en
    08b_M3 (ahi el target era PIB TRIMESTRAL y se evitaba usar el IGAE
    MENSUAL, casi el mismo concepto a otra frecuencia, como insumo)."""
    m3mod = import_module("08b_M3_dfm_robusto")
    all_cols = NIVEL1_COLS + ["IGAE_target"]
    panel_clean = panel.loc[:CLEAN_CUTOFF, all_cols].copy()
    fit_clean = m3mod.em_dfm_general(panel_clean, all_cols, verbose=False)
    return dict(phi=fit_clean["phi"], lam=fit_clean["lam"], R=fit_clean["R"])


def main():
    for label, cutoff in [("Corte 1 (15-jun)", CORTE_1), ("Corte 2 (30-jun)", CORTE_2), ("Corte 3 (23-jul)", CORTE_3)]:
        panel = build_monthly_panel(cutoff)
        print(f"\n=== {label}: {cutoff.date()} ===")
        for c in NIVEL1_COLS + ["IGAE_target"]:
            s = panel[c].dropna()
            last = s.index.max() if len(s) else None
            print(f"  {c:20s} ultimo dato disponible: {last.date() if last is not None else 'NINGUNO'}")
        vintage.assert_no_future_reference_dates(
            panel[NIVEL1_COLS + ["IGAE_target"]].dropna(how="all"), asof=cutoff, label=label)

        with open(INTERIM / f"panel_mensual_igae_corte{cutoff.strftime('%Y%m%d')}.pkl", "wb") as f:
            pickle.dump(dict(panel=panel, cutoff=cutoff), f)


if __name__ == "__main__":
    main()
