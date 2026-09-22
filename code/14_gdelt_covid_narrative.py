"""
Indice de narrativa COVID via GDELT GKG 2.1, filtrado por TEMA (no solo por
ubicacion Mexico). Periodo 2020-01 a 2022-12 (COVID no existia antes, no
hay linea base pre-pandemia posible). Muestreo: 2 slices/dia (00:00,
12:00 UTC), 1 dia/semana (miercoles), ~3 anios -> ~310 archivos reales.

Filtro: articulos con ubicacion Mexico (#MX# en V1Locations o dominio .mx)
Y al menos un tema de la lista COVID_THEMES en V1Themes.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "gdelt_covid"
RAW.mkdir(parents=True, exist_ok=True)
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

COVID_THEMES = [
    "CORONAVIRUS", "TAX_DISEASE_CORONAVIRUS", "HEALTH_PANDEMIC", "EPIDEMIC",
    "PANDEMIC", "SOC_QUARANTINE", "TAX_DISEASE_OUTBREAK", "TAX_DISEASE_INFECTIOUS",
]
MX_LOC = "#MX#"

# sub-temas para desglosar la "composicion" de la narrativa, no solo el tono
SUBTHEME_GROUPS = {
    "salud_publica": ["GENERAL_HEALTH", "MEDICAL", "WB_635_PUBLIC_HEALTH", "HEALTH_PANDEMIC"],
    "miedo_crisis": ["EPU_CATS_MIGRATION_FEAR", "CRISISLEX", "FEAR"],
    "cuarentena_restricciones": ["SOC_QUARANTINE", "CRISISLEX_T01_CAUTION_ADVICE"],
    "economia": ["ECON_", "WB_2670_JOBS", "UNEMPLOYMENT"],
    "gobierno_politica": ["EPU_POLICY", "GENERAL_GOVERNMENT", "TAX_FNCACT_AUTHORITIES"],
}


def sample_dates(start="2020-01-01", end="2022-12-31", weekday=2):
    """weekday=2 -> miercoles. 1 dia/semana."""
    all_days = pd.date_range(start, end, freq="D")
    return [d for d in all_days if d.weekday() == weekday]


def slice_urls_for_date(date: pd.Timestamp):
    base = "https://data.gdeltproject.org/gdeltv2/"
    stamps = [date.strftime("%Y%m%d") + h for h in ("000000", "120000")]
    return [f"{base}{s}.gkg.csv.zip" for s in stamps]


def download_and_extract(url: str):
    fname = RAW / Path(url).name
    csv_path = fname.with_suffix("")
    if not csv_path.exists():
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                return None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(RAW)
        except Exception as e:
            print(f"  WARN descarga fallo {url}: {e}")
            return None
    try:
        df = pd.read_csv(csv_path, sep="\t", header=None, usecols=[3, 7, 9, 15],
                          names=["domain", "themes", "locations", "tone"], dtype=str,
                          on_bad_lines="skip")
    except Exception as e:
        print(f"  WARN parseo fallo {csv_path}: {e}")
        return None
    finally:
        try:
            csv_path.unlink()  # borra el .csv extraido de inmediato para no acumular espacio
        except FileNotFoundError:
            pass
    return df


def covid_mx_records(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None
    loc = df["locations"].fillna("")
    dom = df["domain"].fillna("")
    themes = df["themes"].fillna("")
    is_mx = loc.str.contains(MX_LOC, regex=False) | dom.str.endswith(".mx")
    is_covid = themes.str.contains("|".join(COVID_THEMES), case=False, regex=True)
    return df[is_mx & is_covid]


def build():
    dates = sample_dates()
    rows = []
    for d in dates:
        urls = slice_urls_for_date(d)
        recs = []
        for url in urls:
            df = download_and_extract(url)
            r = covid_mx_records(df)
            if r is not None and len(r) > 0:
                recs.append(r)
        if not recs:
            rows.append(dict(fecha=d, n_articulos=0, tono_promedio=np.nan))
            print(f"  {d.date()}: 0 articulos covid+MX")
            continue
        all_recs = pd.concat(recs)
        tones = []
        subtheme_counts = {k: 0 for k in SUBTHEME_GROUPS}
        for _, row in all_recs.iterrows():
            t = row["tone"]
            if isinstance(t, str):
                parts = t.split(",")
                if len(parts) >= 1:
                    try:
                        tones.append(float(parts[0]))
                    except ValueError:
                        pass
            th = row["themes"] or ""
            for grp, keys in SUBTHEME_GROUPS.items():
                if any(k in th for k in keys):
                    subtheme_counts[grp] += 1
        row_out = dict(
            fecha=d, n_articulos=len(all_recs),
            tono_promedio=np.mean(tones) if tones else np.nan,
        )
        for grp, cnt in subtheme_counts.items():
            row_out[f"pct_{grp}"] = cnt / len(all_recs)
        rows.append(row_out)
        print(f"  {d.date()}: {len(all_recs)} articulos covid+MX, tono={row_out['tono_promedio']:.2f}"
              if tones else f"  {d.date()}: {len(all_recs)} articulos, sin tono valido")

    daily = pd.DataFrame(rows).set_index("fecha")
    daily.to_csv(PROC / "gdelt_covid_daily.csv")

    monthly = daily.resample("MS").agg(
        n_articulos=("n_articulos", "sum"),
        tono_promedio=("tono_promedio", "mean"),
        **{f"pct_{g}": (f"pct_{g}", "mean") for g in SUBTHEME_GROUPS},
    )
    monthly.to_csv(PROC / "gdelt_covid_monthly.csv")
    print("\nGuardado data/processed/gdelt_covid_monthly.csv")
    print(monthly.round(3))
    return monthly


if __name__ == "__main__":
    build()
