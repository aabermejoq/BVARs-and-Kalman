"""
Preparacion del panel mensual para el nowcast/forecast de IGAE.

Fuentes:
- data/raw/basededatos.xlsx (INEGI BIE / Banxico SIE, pull ya provisto en el repo)
- BMV IPC (^MXX) via Yahoo Finance chart API (Banxico SIE no trae el indice bursatil)
- INDPRO EEUU via FRED (fredgraph.csv, sin necesidad de API key)
- Casos COVID Mexico via Our World in Data (owid-covid-data.csv)

Regla de transformacion (metodologia tipo Banxico, ver docstring de TRANSFORM_RULES):
  indice / nivel de actividad -> log-diferencia mensual (log(x_t) - log(x_{t-1}))
  tasa (ya en %)              -> diferencia simple mensual
  balance / spread que puede ser negativo -> diferencia simple EN NIVEL (no log)
"""
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

XLSX = RAW / "basededatos.xlsx"


_PERIODO_RE = re.compile(r"^(\d{4})/(\d{2})")


def _is_periodo(v):
    return v is not None and _PERIODO_RE.match(str(v)) is not None


def _month_index_from_periodos(vals):
    """INEGI 'Periodos' column: 'YYYY/MM' o 'YYYY/MM p1' (para datos preliminares)."""
    out = []
    for v in vals:
        m = _PERIODO_RE.match(str(v))
        y, mo = m.groups()
        out.append(pd.Timestamp(int(y), int(mo), 1))
    return out


def load_otros():
    """Hoja 'otros': IGAE, ActividadIndustrial, FBCF (mensual, desestacionalizado, base 2018)."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["otros"]
    rows = [r for r in ws.iter_rows(min_row=7, values_only=True) if _is_periodo(r[0])]
    periodos = [r[0] for r in rows]
    igae = [r[2] for r in rows]
    ai = [r[3] for r in rows]
    fbcf = [r[4] for r in rows]
    idx = _month_index_from_periodos(periodos)
    df = pd.DataFrame(
        {"IGAE": igae, "ActividadIndustrial": ai, "FBCF": fbcf}, index=idx
    )
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_consumo():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Consumo"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    imcp = [r[1] for r in rows if r[0] is not None]
    antad = [r[2] for r in rows if r[0] is not None]
    autos = [r[3] for r in rows if r[0] is not None]
    tarjetas = [r[4] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame(
        {"IMCP": imcp, "ANTAD": antad, "AUTOS": autos, "TARJETAS": tarjetas}, index=idx
    )
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_desempleo():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["desempleo"]
    rows = [r for r in ws.iter_rows(min_row=7, values_only=True) if _is_periodo(r[0])]
    periodos = [r[0] for r in rows]
    tasa = [r[2] for r in rows]
    idx = _month_index_from_periodos(periodos)
    df = pd.DataFrame({"Desempleo": tasa}, index=idx)
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_imss():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["IMSS"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    empleos = [r[1] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame({"IMSS": empleos}, index=idx)
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_uncertainty():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Uncertainity"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    epu = [r[1] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame({"EPU": epu}, index=idx)
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_inpc():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["INPC"]
    rows = list(ws.iter_rows(min_row=19, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    inpc = [r[1] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame({"INPC": inpc}, index=idx)
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_tc():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["TC"]
    rows = list(ws.iter_rows(min_row=19, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    tc = [r[1] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame({"TC": tc}, index=idx)
    df.index.name = "fecha"
    return df.apply(pd.to_numeric, errors="coerce")


def load_tiie():
    """TIIE es diaria -> promedio mensual."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["TIIE"]
    rows = list(ws.iter_rows(min_row=19, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    tiie = [r[1] for r in rows if r[0] is not None]
    s = pd.Series(tiie, index=[pd.Timestamp(f) for f in fechas]).apply(
        pd.to_numeric, errors="coerce"
    )
    monthly = s.resample("MS").mean()
    monthly.name = "TIIE"
    df = monthly.to_frame()
    df.index.name = "fecha"
    return df


def load_balanza():
    """Exportaciones - Importaciones = balanza comercial (puede ser negativa)."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Balanza"]
    rows = list(ws.iter_rows(min_row=19, values_only=True))
    fechas = [r[0] for r in rows if r[0] is not None]
    exp = [r[1] for r in rows if r[0] is not None]
    imp = [r[2] for r in rows if r[0] is not None]
    idx = [pd.Timestamp(f) for f in fechas]
    df = pd.DataFrame({"Exportaciones": exp, "Importaciones": imp}, index=idx).apply(
        pd.to_numeric, errors="coerce"
    )
    df["Balanza"] = df["Exportaciones"] - df["Importaciones"]
    df.index.name = "fecha"
    return df[["Balanza"]]


def download_bmv():
    """BMV IPC (^MXX), mensual, via API publica de graficos de Yahoo Finance."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EMXX"
    r = requests.get(
        url,
        params={"range": "max", "interval": "1mo"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    idx = [pd.Timestamp(t, unit="s").normalize().replace(day=1) for t in ts]
    df = pd.DataFrame({"BMV": closes}, index=idx)
    df.index.name = "fecha"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_csv(RAW / "bmv_ipc.csv")
    return df


def download_indpro():
    """Industrial Production Index EEUU, via FRED (csv publico, sin API key)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": "INDPRO"}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["fecha", "INDPRO"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")
    df["INDPRO"] = pd.to_numeric(df["INDPRO"], errors="coerce")
    df.to_csv(RAW / "indpro.csv")
    return df


def download_covid_mexico():
    """Casos y muertes diarias COVID Mexico via Our World in Data."""
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    mx = df[df["iso_code"] == "MEX"].copy()
    mx["date"] = pd.to_datetime(mx["date"])
    cols = [
        "date",
        "new_cases",
        "new_cases_smoothed",
        "total_cases",
        "new_deaths",
        "new_deaths_smoothed",
        "total_deaths",
        "hosp_patients",
        "icu_patients",
        "stringency_index",
    ]
    cols = [c for c in cols if c in mx.columns]
    mx = mx[cols].set_index("date").sort_index()
    mx.to_csv(RAW / "covid_mexico.csv")
    return mx


def build_monthly_panel():
    pieces = [
        load_otros(),
        load_consumo(),
        load_desempleo(),
        load_imss(),
        load_uncertainty(),
        load_inpc(),
        load_tc(),
        load_tiie(),
        load_balanza(),
    ]
    try:
        bmv = download_bmv()
        pieces.append(bmv)
    except Exception as e:
        print(f"WARN: no se pudo descargar BMV ({e}); se omite del panel")
    try:
        indpro = download_indpro()
        pieces.append(indpro)
    except Exception as e:
        print(f"WARN: no se pudo descargar INDPRO ({e}); se omite del panel")

    panel = pieces[0]
    for p in pieces[1:]:
        panel = panel.join(p, how="outer")
    panel = panel.sort_index()
    panel.to_csv(PROC / "panel_monthly_levels.csv")
    return panel


# --- Reglas de transformacion ---------------------------------------------
# indice / nivel de actividad -> log-diferencia mensual
LOG_DIFF_VARS = [
    "IGAE",
    "ActividadIndustrial",
    "FBCF",
    "IMCP",
    "ANTAD",
    "AUTOS",
    "TARJETAS",
    "IMSS",
    "INPC",
    "TC",
    "BMV",
    "INDPRO",
]
# tasa (ya en %) -> diferencia simple mensual
SIMPLE_DIFF_VARS = ["Desempleo", "TIIE"]
# balance/indice que puede ser negativo o no es multiplicativo -> diferencia simple en nivel
LEVEL_DIFF_VARS = ["Balanza", "EPU"]


def transform_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    for c in panel.columns:
        if c in LOG_DIFF_VARS:
            s = panel[c]
            out[c + "_mom"] = np.log(s) - np.log(s.shift(1))
        elif c in SIMPLE_DIFF_VARS:
            out[c + "_mom"] = panel[c].diff(1)
        elif c in LEVEL_DIFF_VARS:
            out[c + "_mom"] = panel[c].diff(1)
        else:
            print(f"WARN: {c} sin regla de transformacion explicita, se omite")
    out.index.name = "fecha"
    return out


if __name__ == "__main__":
    panel = build_monthly_panel()
    print("Panel niveles:", panel.shape, panel.index.min(), "->", panel.index.max())
    transformed = transform_panel(panel)
    transformed.to_csv(PROC / "panel_monthly_mom.csv")
    print("Panel m/m:", transformed.shape)
    print(transformed.tail(10))

    try:
        covid = download_covid_mexico()
        print("COVID MX:", covid.shape, covid.index.min(), "->", covid.index.max())
    except Exception as e:
        print(f"WARN: no se pudo descargar COVID MX ({e})")
