"""
21_agosto_mobility_stringency.py

Prueba si Google Mobility y OxCGRT (StringencyIndex) -- YA DESCARGADOS en
data/external/ pero solo hasta el corte de MAYO (10-may-2020) y nunca
usados en ningun modelo -- aportan algo antes de construir la
infraestructura, mas pesada, de mineria de texto de GDELT.

Limitacion real, declarada de entrada: ambas series son productos DE LA
era-COVID (Mobility publica desde 15-feb-2020, OxCGRT trackea desde
ene-2020). No existe historia PRE-2020 para ninguna de las 2, asi que NO
se puede repetir el ejercicio de "estimar el puente en 2009-2019 limpio y
validar fuera de muestra" que se hizo con tarjetas -- aqui la evaluacion
es necesariamente distinta: correlacion CONTEMPORANEA/con rezagos dentro
de la propia ventana COVID (feb-jul 2020) contra los indicadores del panel
que YA se usan en M2/M3 (IMSS, ANTAD, ActividadIndustrial), para ver si
mobility podria servir como senal de "nowcast" (llenar el mes mas
reciente, sin publicar aun) en vez de como puente hacia el futuro.

Paso 1: re-descargar Mobility + OxCGRT con el corte extendido a
agosto/3T20 (los archivos fuente son historicos completos, no
incrementales -- solo hay que cambiar el corte de filtrado).
Paso 2: correlacion contemporanea y con rezagos +-2 meses.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
INTERIM = ROOT / "data" / "interim"

CUTOFF = pd.Timestamp("2020-08-15")
SAFE_CUTOFF_HF = pd.Timestamp("2020-08-10")  # margen de reporte, mismo criterio que 01b (mayo)

BENCHMARK_COUNTRIES = ["CN", "KR", "IT", "ES", "US", "MX"]
ISO2_TO_ISO3 = {"CN": "CHN", "KR": "KOR", "IT": "ITA", "ES": "ESP", "US": "USA", "MX": "MEX"}


def fetch_google_mobility_agosto():
    frames = []
    for cc in BENCHMARK_COUNTRIES:
        url = f"https://www.gstatic.com/covid19/mobility/2020_{cc}_Region_Mobility_Report.csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  [{cc}] Google Mobility no disponible: {e}")
            continue
        df = pd.read_csv(io.StringIO(r.text))
        nat = df[df["sub_region_1"].isna() & df["sub_region_2"].isna()].copy()
        if nat.empty:
            continue
        nat["date"] = pd.to_datetime(nat["date"])
        nat = nat[nat["date"] <= SAFE_CUTOFF_HF]
        nat["country_code"] = cc
        cols = [
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
        nat = nat.groupby(["country_code", "date"], as_index=False)[cols].mean()
        frames.append(nat[["country_code", "date"] + cols])
        print(f"  [{cc}] Google Mobility: {len(nat)} filas hasta {nat['date'].max().date()}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(EXTERNAL / "google_mobility_benchmarks_agosto.csv", index=False)
    return out


def fetch_oxcgrt_agosto():
    url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_national_v1.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), usecols=[
        "CountryName", "CountryCode", "Date",
        "StringencyIndex_Average", "GovernmentResponseIndex_Average",
        "EconomicSupportIndex",
    ], dtype={"Date": str})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df[df["CountryCode"].isin(ISO2_TO_ISO3.values())]
    df = df[df["Date"] <= SAFE_CUTOFF_HF]
    df.to_csv(EXTERNAL / "oxcgrt_benchmarks_agosto.csv", index=False)
    print(f"  OxCGRT: {len(df)} filas para {df['CountryCode'].nunique()} paises hasta {df['Date'].max().date()}")
    return df


def main():
    print("Re-descargando Google Mobility con corte extendido a agosto/3T20...")
    mob = fetch_google_mobility_agosto()
    print("\nRe-descargando OxCGRT con corte extendido a agosto/3T20...")
    ox = fetch_oxcgrt_agosto()

    mob_mx = mob[mob["country_code"] == "MX"].set_index("date")
    ox_mx = ox[ox["CountryCode"] == "MEX"].set_index("Date")

    mob_mx_m = mob_mx.resample("MS").mean(numeric_only=True)
    ox_mx_m = ox_mx[["StringencyIndex_Average"]].resample("MS").mean()

    with open(INTERIM / "series_raw.pkl", "rb") as f:
        import pickle
        series = pickle.load(f)

    def yoy(s):
        return 100 * np.log(s / s.shift(12))

    panel_existente = pd.DataFrame({
        "ActividadIndustrial": yoy(series["otros"]["ActividadIndustrial"]),
        "IMSS": yoy(series["IMSS"]["IMSS_empleos"]),
        "ANTAD": yoy(series["Consumo"]["ANTAD"]),
        "AUTOS": yoy(series["Consumo"]["AUTOS"]),
    })

    combo = pd.concat([
        mob_mx_m["retail_and_recreation_percent_change_from_baseline"].rename("mobility_retail"),
        mob_mx_m["workplaces_percent_change_from_baseline"].rename("mobility_work"),
        ox_mx_m["StringencyIndex_Average"].rename("stringency"),
        panel_existente,
    ], axis=1).loc["2020-02-01":"2020-07-01"]

    print("\n=== Panel mensual combinado (feb-jul 2020) ===")
    print(combo.round(1).to_string())

    print("\n=== Correlacion contemporanea y con rezagos (mobility_retail lidera si rezago>0) ===")
    targets = ["ActividadIndustrial", "IMSS", "ANTAD", "AUTOS"]
    drivers = ["mobility_retail", "mobility_work", "stringency"]
    for drv in drivers:
        for tgt in targets:
            for lag in [-1, 0, 1]:
                x = combo[drv].shift(-lag)  # lag>0: driver adelantado (lidera al target)
                y = combo[tgt]
                pair = pd.concat([x, y], axis=1).dropna()
                if len(pair) < 3:
                    continue
                c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                print(f"  {drv:16s} (rezago {lag:+d}m) vs {tgt:20s}  n={len(pair)}  corr={c:+.2f}")

    combo.to_csv(INTERIM / "mobility_stringency_mx_monthly_agosto.csv")
    print(f"\nGuardado: {INTERIM / 'mobility_stringency_mx_monthly_agosto.csv'}")


if __name__ == "__main__":
    main()
