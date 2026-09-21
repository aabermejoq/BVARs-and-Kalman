"""
01_audit_data.py

FASE 1 del ejercicio "México 2T20: la escalera M0-M4".

Objetivo: inspeccionar basededatos.xlsx (10 hojas descargadas de INEGI/Banxico/
fuentes académicas) y producir:
  (a) series limpias en formato largo/ancho (data/interim/*.csv)
  (b) una tabla de auditoria por variable (output/audit/audit_variables.csv)
  (c) una matriz de correlaciones para el analisis de redundancia
      (output/audit/correlations_vs_pib.csv)

Este script NO estima ningun modelo y NO aplica el corte de tiempo real.
El corte real-time (15-may-2020) se aplica en 02_real_time_vintage.py.

Fuente del archivo: subido por el usuario a GitHub (main, commits b32d7a8 y
e3ac6b8). Las hojas de INEGI/Banxico traen su propio bloque de metadatos
(fuente, unidad, periodo disponible) que se preserva en el diccionario de
datos; las hojas 'Consumo', 'Uncertainity' e 'IMSS' NO traen metadatos de
fuente dentro del archivo y se marcan como tales.
"""
import re
import pickle
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "basededatos.xlsx"
INTERIM = ROOT / "data" / "interim"
AUDIT_OUT = ROOT / "output" / "audit"
INTERIM.mkdir(parents=True, exist_ok=True)
AUDIT_OUT.mkdir(parents=True, exist_ok=True)


def parse_period_str(s):
    """INEGI BIE period labels look like '1993/01 p1' (monthly) or
    '1993/01 p1' with p1..p4 for quarters depending on the sheet."""
    m = re.match(r"(\d{4})/(\d{2})", str(s))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def load_workbook():
    return openpyxl.load_workbook(RAW_XLSX, data_only=True)


def extract_series(wb):
    """Hand-mapped extraction: each INEGI/Banxico sheet has a different
    metadata-block length before the actual data starts. Row offsets were
    identified by manual inspection (see audit notes) and are NOT guessed
    programmatically to avoid silently parsing the wrong block."""
    series = {}

    # --- PIB: quarterly, INEGI BIE, 'Serie desestacionalizada', precios 2018 ---
    ws = wb["PIB"]
    rows = list(ws.iter_rows(values_only=True))
    data = []
    for r in rows[6:]:
        if r[0] is None:
            continue
        y, q = parse_period_str(r[0])
        if y is None:
            continue
        month = (q - 1) * 3 + 1
        data.append((pd.Timestamp(year=y, month=month, day=1), r[2]))
    series["PIB"] = pd.DataFrame(data, columns=["date", "PIB"]).set_index("date")

    # --- otros: IGAE, Actividad Industrial, FBCF (monthly, base 2018=100, SA) ---
    ws = wb["otros"]
    rows = list(ws.iter_rows(values_only=True))
    data = []
    for r in rows[6:]:
        if r[0] is None:
            continue
        y, m = parse_period_str(r[0])
        if y is None:
            continue
        data.append((pd.Timestamp(year=y, month=m, day=1), r[2], r[3], r[4]))
    series["otros"] = pd.DataFrame(
        data, columns=["date", "IGAE", "ActividadIndustrial", "FBCF"]
    ).set_index("date")

    # --- desempleo: tasa de desocupación ENOE, SA (monthly) ---
    ws = wb["desempleo"]
    rows = list(ws.iter_rows(values_only=True))
    data = []
    for r in rows[6:]:
        if r[0] is None:
            continue
        y, m = parse_period_str(r[0])
        if y is None:
            continue
        data.append((pd.Timestamp(year=y, month=m, day=1), r[2]))
    series["desempleo"] = pd.DataFrame(
        data, columns=["date", "TasaDesempleo"]
    ).set_index("date")

    # --- INPC: monthly index, data starts row 18 ---
    ws = wb["INPC"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1]) for r in rows[18:] if r[0] is not None]
    series["INPC"] = pd.DataFrame(data, columns=["date", "INPC"]).set_index("date")

    # --- TIIE: daily, fondeo a 1 dia, data starts row 18 ---
    ws = wb["TIIE"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1]) for r in rows[18:] if r[0] is not None]
    series["TIIE"] = pd.DataFrame(data, columns=["date", "TIIE"]).set_index("date")

    # --- TC: monthly, pesos por dolar, data starts row 18 ---
    ws = wb["TC"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1]) for r in rows[18:] if r[0] is not None]
    series["TC"] = pd.DataFrame(data, columns=["date", "TC"]).set_index("date")

    # --- Balanza: exportaciones/importaciones totales SA, data starts row 18 ---
    ws = wb["Balanza"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1], r[2]) for r in rows[18:] if r[0] is not None]
    series["Balanza"] = pd.DataFrame(
        data, columns=["date", "Exportaciones", "Importaciones"]
    ).set_index("date")

    # --- Consumo: IMCP, ANTAD, AUTOS, TARJETAS (SIN metadatos de fuente) ---
    ws = wb["Consumo"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1], r[2], r[3], r[4]) for r in rows[1:] if r[0] is not None]
    series["Consumo"] = pd.DataFrame(
        data, columns=["date", "IMCP", "ANTAD", "AUTOS", "TARJETAS"]
    ).set_index("date")

    # --- Uncertainity: Mexican EPU index (Baker-Bloom-Davis style), sin metadatos ---
    ws = wb["Uncertainity"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1]) for r in rows[1:] if r[0] is not None]
    series["Uncertainity"] = pd.DataFrame(data, columns=["date", "EPU"]).set_index("date")

    # --- IMSS: empleos (trabajadores asegurados), sin metadatos ---
    ws = wb["IMSS"]
    rows = list(ws.iter_rows(values_only=True))
    data = [(r[0], r[1]) for r in rows[1:] if r[0] is not None]
    series["IMSS"] = pd.DataFrame(data, columns=["date", "IMSS_empleos"]).set_index("date")

    return series


def build_audit_table(series):
    """Variable-level audit: frequency, unit, coverage, missingness."""
    rows = []

    def add(var, sheet, freq, unit, source, note):
        df = series[sheet]
        col = var
        s = df[col]
        rows.append(
            dict(
                variable=var,
                hoja=sheet,
                frecuencia=freq,
                unidad=unit,
                fuente_declarada_en_archivo=source,
                primera_fecha=s.first_valid_index(),
                ultima_fecha=s.last_valid_index(),
                n_obs=int(s.notna().sum()),
                n_missing_en_rango=int(s.loc[s.first_valid_index():s.last_valid_index()].isna().sum()),
                nota=note,
            )
        )

    add("PIB", "PIB", "Trimestral", "Millones de pesos a precios de 2018",
        "INEGI - BIE (PIB trimestral, serie desestacionalizada, base 2018)",
        "PIB real desestacionalizado del enfoque de la producción/gasto agregado. "
        "NO es PNB: el titulo de la hoja confirma 'Producto interno bruto trimestral'.")

    add("IGAE", "otros", "Mensual", "Indice base 2018=100 (SA)",
        "INEGI - BIE (Indicador Global de la Actividad Economica, SA)",
        "Mejor proxy mensual de PIB (ver correlacion). Publicado con ~8 semanas de rezago.")

    add("ActividadIndustrial", "otros", "Mensual", "Indice base 2018=100 (SA)",
        "INEGI - BIE (Actividad industrial total, SA)",
        "Subcomponente de IGAE (correlacion ~0.96 con IGAE): alto riesgo de redundancia.")

    add("FBCF", "otros", "Mensual", "Indice base 2018=100 (SA)",
        "INEGI - BIE (Formacion bruta de capital fijo, SA)",
        "Proxy de inversion. Rezago de publicacion mayor (~10-11 semanas); "
        "en 2020 el ultimo dato solido disponible al corte es enero, con febrero "
        "en el limite (requiere verificar calendario 2020 exacto).")

    add("TasaDesempleo", "desempleo", "Mensual", "% de la PEA (SA, ENOE)",
        "INEGI - BIE (ENOE, tasa de desocupacion, SA)",
        "PROBLEMA DE MEDICION CONFIRMADO: INEGI suspendio la ENOE presencial el "
        "31-mar-2020 y CANCELO la publicacion de abril (prevista 27-may-2020), "
        "sustituyendola por la ETOE (encuesta telefonica, metodologia distinta) "
        "para abril-junio 2020. El valor de esta serie para abr/may-2020 proviene "
        "de un empalme posterior no disponible al 15-may-2020 y no comparable 1:1 "
        "con la ENOE regular.")

    add("INPC", "INPC", "Mensual", "Indice (sin unidad), base 2Q jul-2018",
        "Banco de Mexico (INPC, publicado ahora por INEGI desde jul-2011)",
        "Indice de precios, no revisado historicamente.")

    add("TIIE", "TIIE", "Diaria", "Porcentaje (tasa promedio)",
        "Banco de Mexico (TIIE de Fondeo a 1 dia)",
        "Tasa de mercado interbancario, disponible el mismo dia. Sin rezago de publicacion relevante.")

    add("TC", "TC", "Mensual (prom. de diario)", "Pesos por dolar",
        "Banco de Mexico (tipo de cambio FIX, promedio mensual)",
        "Precio de mercado, no revisado. Puede reconstruirse a frecuencia diaria si se requiere.")

    add("Exportaciones", "Balanza", "Mensual", "Miles de dolares (SA)",
        "Banco de Mexico (Balanza comercial de mercancias, SA)",
        "Solo bienes (no incluye servicios ni renta ni transferencias): NO es la balanza en cuenta corriente completa.")

    add("Importaciones", "Balanza", "Mensual", "Miles de dolares (SA)",
        "Banco de Mexico (Balanza comercial de mercancias, SA)",
        "Ver PIB=C+I+G+X-M: usar como indicador de actividad (insumos importados a manufactura de exportacion), NO como resta mecanica del PIB.")

    add("IMCP", "Consumo", "Mensual", "Indice (unidad no documentada en archivo)",
        "SIN METADATOS EN EL ARCHIVO - fuente no verificable internamente",
        "Probable Indice de Confianza del Consumidor (INEGI/Banxico) por escala y "
        "contexto de la hoja (Consumo), pero no se puede confirmar la identidad "
        "exacta de la serie sin metadatos. Requiere confirmacion del usuario antes de usarse.")

    add("ANTAD", "Consumo", "Mensual", "Indice (unidad no documentada en archivo)",
        "SIN METADATOS EN EL ARCHIVO - probablemente ANTAD (ventas mismas tiendas)",
        "Publicacion muy rapida (primeros dias del mes siguiente). Buen candidato de alta frecuencia.")

    add("AUTOS", "Consumo", "Mensual", "Indice (unidad no documentada en archivo)",
        "SIN METADATOS EN EL ARCHIVO - probablemente ventas/registro de vehiculos (AMIA/INEGI)",
        "Publicacion muy rapida (primeros dias del mes siguiente). Muy volatil en crisis (colapso de -60% abr-2020 en el nivel).")

    add("TARJETAS", "Consumo", "Mensual", "Indice (unidad no documentada en archivo)",
        "SIN METADATOS EN EL ARCHIVO",
        "Cobertura corta (inicia 2009): NO cubre la crisis Tequila (1994-95) y solo "
        "la cola de la crisis 2008-09. Correlacion baja con PIB (~0.28-0.30). "
        "Candidato debil para EXTENSION/M4 por falta de episodios historicos.")

    add("EPU", "Uncertainity", "Mensual", "Indice (base no documentada en archivo)",
        "SIN METADATOS EN EL ARCHIVO - estilo Baker-Bloom-Davis (Mexican EPU)",
        "Correlacion ~0 con el nivel de crecimiento del PIB: no es un indicador de "
        "actividad, sino de incertidumbre de politica. Rezago de publicacion tipico "
        "de este tipo de indices ~2-4 semanas (requiere verificar para 15-may-2020).")

    add("IMSS_empleos", "IMSS", "Mensual", "Numero de trabajadores asegurados",
        "SIN METADATOS EN EL ARCHIVO - coincide con IMSS (trabajadores asegurados)",
        "Publicacion muy rapida (primeros dias habiles del mes siguiente, IMSS "
        "publica su propio boletin). Excelente candidato de alta frecuencia para el "
        "choque COVID (abril 2020 disponible desde inicios de mayo 2020).")

    return pd.DataFrame(rows)


def correlation_analysis(series):
    """Redundancy check vs PIB q/q growth, using only pre-2020 data to avoid
    the COVID shock itself dominating a correlation used for variable
    *selection* (selection must not be informed by the COVID episode)."""
    pib_qoq = series["PIB"]["PIB"].pct_change(1) * 100

    otros = series["otros"]
    cons = series["Consumo"]
    imss = series["IMSS"][["IMSS_empleos"]]
    unc = series["Uncertainity"]
    desem = series["desempleo"]
    bal = series["Balanza"]
    tc = series["TC"]

    hf = pd.concat([otros, cons, imss, unc, desem, bal, tc], axis=1, sort=True)
    hf_q = hf.resample("QS").mean()
    hf_q_qoq = hf_q.pct_change(1) * 100

    df = pd.concat([pib_qoq.rename("PIB_qoq"), hf_q_qoq], axis=1, sort=True)
    df_pre = df.loc[:"2019-10-01"]
    corr = df_pre.corr()["PIB_qoq"].drop("PIB_qoq").sort_values(ascending=False)
    return corr.to_frame("corr_con_PIB_qoq_pre2020")


def main():
    wb = load_workbook()
    series = extract_series(wb)

    with open(INTERIM / "series_raw.pkl", "wb") as f:
        pickle.dump(series, f)

    for name, df in series.items():
        df.to_csv(INTERIM / f"{name}.csv")

    audit = build_audit_table(series)
    audit.to_csv(AUDIT_OUT / "audit_variables.csv", index=False)

    corr = correlation_analysis(series)
    corr.to_csv(AUDIT_OUT / "correlations_vs_pib.csv")

    print("Auditoria completa.")
    print(f"- Series guardadas en: {INTERIM}")
    print(f"- Tabla de auditoria: {AUDIT_OUT / 'audit_variables.csv'}")
    print(f"- Correlaciones vs PIB q/q (pre-2020): {AUDIT_OUT / 'correlations_vs_pib.csv'}")
    print("\n--- Resumen tabla de auditoria ---")
    print(audit[["variable", "hoja", "frecuencia", "primera_fecha", "ultima_fecha", "n_obs"]].to_string(index=False))
    print("\n--- Correlacion con PIB q/q (pre-2020) ---")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
