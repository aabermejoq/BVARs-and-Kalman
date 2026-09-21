"""
01b_external_covid_benchmarks.py

Extension de la FASE 1/2 (a peticion explicita del usuario): amplia la base
de datos con fuentes EXTERNAS gratuitas, sin API key, para dar a M4 un
conjunto de entrenamiento mas grande que los ~2 episodios historicos de
Mexico (crisis Tequila 1994-95, GFC 2008-09).

Logica de por que esto NO viola la regla real-time (15-may-2020):
  Usamos la trayectoria COVID de OTROS PAISES que fueron golpeados ANTES
  que Mexico. Esa informacion es publica y observable el 15-may-2020 (no es
  el futuro de Mexico, es el pasado/presente de otros paises). Un analista
  en Ciudad de Mexico el 15-may-2020 SI podia ver como habian evolucionado
  la movilidad y las medidas de contencion en China (ene-2020), Corea del
  Sur (feb-2020), Italia y Espana (mar-2020) y Estados Unidos (mar-abr-2020).

Fuentes (todas publicas, sin necesidad de API key):
  1. Google COVID-19 Community Mobility Reports (gstatic.com/covid19/mobility)
     - Publicado por primera vez el 3-abr-2020, con historia desde 15-feb-2020.
     - Cobertura por pais, variables: retail_recreation, grocery_pharmacy,
       parks, transit_stations, workplaces, residential (% cambio vs. linea
       base pre-COVID).
     - LIMITACION: China (CN) NO tiene cobertura (servicios de Google
       bloqueados) - se cae de este dataset, se usa Oxford para caracterizar
       su choque.
  2. Oxford COVID-19 Government Response Tracker (OxCGRT, GitHub publico)
     - Indice de rigor de politicas de contencion (StringencyIndex_Average),
       diario, por pais, codificado casi en tiempo real desde ene-2020.
  3. FRED (Federal Reserve Economic Data), descarga CSV directa sin API key
     - Indicadores de EEUU (INDPRO: produccion industrial, ICSA: solicitudes
       iniciales de seguro de desempleo) como proxy del canal de demanda
       externa de EEUU hacia la manufactura de exportacion mexicana
       (~80% de las exportaciones de Mexico van a EEUU).

Se aplica un corte conservador: fecha de referencia <= 2020-05-10 para
mobility/Oxford (margen de ~5 dias vs. el corte de 15-may-2020 por rezago
tipico de reporte/codificacion) y se documentan los rezagos de FRED.
"""
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
AUDIT_OUT = ROOT / "output" / "audit"
EXTERNAL.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2020-05-15")
SAFE_CUTOFF_HF = pd.Timestamp("2020-05-10")  # margen conservador de reporte

# Paises "benchmark": golpeados por COVID antes o al mismo tiempo que Mexico,
# con perfiles de respuesta / velocidad de choque distintos.
BENCHMARK_COUNTRIES = {
    "CN": "China (choque mas temprano, ene-2020; sin cobertura de Google Mobility)",
    "KR": "Corea del Sur (contencion sin confinamiento estricto, feb-2020)",
    "IT": "Italia (confinamiento estricto y prolongado, mar-2020)",
    "ES": "Espana (confinamiento estricto y prolongado, mar-2020)",
    "US": "Estados Unidos (respuesta heterogenea por estado, mar-abr-2020; ademas socio comercial principal de Mexico)",
    "MX": "Mexico (caso de interes: choque a partir de mar-2020)",
}

# Google Mobility usa ISO 3166-1 alpha-2; OxCGRT usa ISO 3166-1 alpha-3.
ISO2_TO_ISO3 = {"CN": "CHN", "KR": "KOR", "IT": "ITA", "ES": "ESP", "US": "USA", "MX": "MEX"}


def fetch_google_mobility():
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
        # Solo nivel nacional (sin desagregacion sub-nacional)
        nat = df[df["sub_region_1"].isna() & df["sub_region_2"].isna()].copy()
        if nat.empty:
            print(f"  [{cc}] Google Mobility: sin filas de nivel nacional (probable sin cobertura).")
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
        # Nota de calidad de datos: Corea del Sur (KR) trae 2 place_id
        # distintos marcados como nivel nacional en el CSV de Google; se
        # promedian por fecha para obtener una sola serie nacional.
        nat = nat.groupby(["country_code", "date"], as_index=False)[cols].mean()
        frames.append(nat[["country_code", "date"] + cols])
        print(f"  [{cc}] Google Mobility: {len(nat)} filas hasta {nat['date'].max().date()}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(EXTERNAL / "google_mobility_benchmarks.csv", index=False)
    return out


def fetch_oxcgrt():
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
    df.to_csv(EXTERNAL / "oxcgrt_benchmarks.csv", index=False)
    print(f"  OxCGRT: {len(df)} filas para {df['CountryCode'].nunique()} paises hasta {df['Date'].max().date()}")
    return df


def fetch_fred(series_ids):
    frames = {}
    for sid in series_ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", sid]
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= CUTOFF]
        frames[sid] = df.set_index("date")[sid]
        print(f"  [FRED {sid}] {len(df)} filas hasta {df['date'].max().date()}")
    out = pd.concat(frames.values(), axis=1)
    out.to_csv(EXTERNAL / "fred_us_benchmarks.csv")
    return out


def main():
    print("Descargando Google COVID-19 Community Mobility Reports (benchmarks)...")
    mob = fetch_google_mobility()

    print("\nDescargando Oxford COVID-19 Government Response Tracker...")
    ox = fetch_oxcgrt()

    print("\nDescargando indicadores de EEUU via FRED (CSV directo, sin API key)...")
    # INDPRO: produccion industrial EEUU (mensual, canal de demanda externa)
    # ICSA: solicitudes iniciales de seguro de desempleo (semanal, muy rapido)
    fred = fetch_fred(["INDPRO", "ICSA"])

    dictionary_rows = [
        dict(variable="google_mobility_{6 subcategorias}", fuente="Google COVID-19 Community Mobility Reports",
             frecuencia="Diaria", cobertura="MX,KR,IT,ES,US (CN sin cobertura)",
             disponible_15may2020="Si (publicado desde 3-abr-2020, historia desde 15-feb-2020)",
             rol_propuesto="M2/M3 (Mexico, indicador de movilidad de altisima frecuencia) y M4 (analogos cross-country)"),
        dict(variable="StringencyIndex_Average", fuente="Oxford COVID-19 Government Response Tracker",
             frecuencia="Diaria", cobertura="CN,KR,IT,ES,US,MX",
             disponible_15may2020="Si (codificado casi en tiempo real desde ene-2020)",
             rol_propuesto="M4 (caracterizacion del choque X_t y alineacion temporal de analogos por rigor de politica, no por fecha calendario)"),
        dict(variable="INDPRO (EEUU)", fuente="FRED (Federal Reserve Bank of St. Louis)",
             frecuencia="Mensual", cobertura="Estados Unidos",
             disponible_15may2020="Ref. feb-2020 si (rezago tipico ~5-6 semanas); marzo-2020 en el limite, tratar con cautela",
             rol_propuesto="M3/M4 (proxy del canal de demanda externa de EEUU hacia manufactura de exportacion mexicana)"),
        dict(variable="ICSA (EEUU)", fuente="FRED (Federal Reserve Bank of St. Louis)",
             frecuencia="Semanal", cobertura="Estados Unidos",
             disponible_15may2020="Si (publicado con ~5 dias de rezago; la serie semanal llega hasta la semana del 9-may-2020)",
             rol_propuesto="M4 (senal muy rapida y muy dramatica del shock: records historicos de solicitudes en marzo-abril 2020, disponible en tiempo real)"),
    ]
    pd.DataFrame(dictionary_rows).to_csv(AUDIT_OUT / "external_data_dictionary.csv", index=False)

    print("\nListo. Archivos guardados en data/external/ y diccionario en "
          "output/audit/external_data_dictionary.csv")


if __name__ == "__main__":
    main()
