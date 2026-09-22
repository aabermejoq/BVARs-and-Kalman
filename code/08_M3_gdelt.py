"""
M3 (parte 3/3): canal de noticias GDELT GKG 2.1.

Descarga real de slices GKG 2.1 (cada 15 min, formato tab-separated, 27
columnas: col 9=V1Locations "tipo#nombre#pais#adm1#lat#lon#featureid;...",
col 15=V1.5Tone "Tone,Positive,Negative,Polarity,ActivityRefDensity,
SelfGroupRefDensity,WordCount"). Se filtra a Mexico via '#MX#' en
V1Locations o dominio '.mx' en SourceCommonName.

Muestreo REDUCIDO respecto al plan original (4-8 slices/dia) por costo de
computo/tiempo en esta sesion: 2 slices/dia (00:00 y 12:00 UTC), 1 dia por
mes, de enero 2017 a junio 2020 (42 meses x 2 = 84 archivos reales, no
fabricados). Se documenta esta reduccion honestamente: el resultado es una
muestra dispersa (una ventana de 15 min x2 representando cada mes), mas
ruidosa que un muestreo diario completo.

Agregado mensual: tono promedio ponderado por conteo de articulos, y
volumen (num. articulos Mexico-relacionados) por mes. Se valida por LOO/CV
contra IGAE_mom y contra |residuo del VAR| (canal de varianza) ANTES de
usarse en el ensamble de M3.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "gdelt"
RAW.mkdir(parents=True, exist_ok=True)
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

MX_RE_LOC = "#MX#"


def sample_dates(start="2017-01-01", end="2020-06-01"):
    months = pd.date_range(start, end, freq="MS")
    return [m + pd.Timedelta(days=0) for m in months]  # dia 1 de cada mes


def slice_urls_for_date(date: pd.Timestamp):
    base = "https://data.gdeltproject.org/gdeltv2/"
    stamps = [date.strftime("%Y%m%d") + h for h in ("000000", "120000")]
    return [f"{base}{s}.gkg.csv.zip" for s in stamps]


def download_and_extract(url: str) -> pd.DataFrame | None:
    fname = RAW / Path(url).name
    csv_name = fname.with_suffix("").name  # quita .zip
    csv_path = RAW / csv_name
    if csv_path.exists():
        pass
    else:
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
        df = pd.read_csv(csv_path, sep="\t", header=None, usecols=[1, 3, 9, 15],
                          names=["date", "domain", "locations", "tone"], dtype=str,
                          on_bad_lines="skip")
    except Exception as e:
        print(f"  WARN parseo fallo {csv_path}: {e}")
        return None
    return df


def mexico_tone_from_slice(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None
    loc = df["locations"].fillna("")
    dom = df["domain"].fillna("")
    is_mx = loc.str.contains(MX_RE_LOC, regex=False) | dom.str.endswith(".mx")
    mx = df[is_mx]
    if len(mx) == 0:
        return dict(n_articles=0, avg_tone=np.nan, avg_polarity=np.nan)
    tones = mx["tone"].dropna().str.split(",", expand=False)
    tone_vals, polarity_vals = [], []
    for t in tones:
        if t is None or len(t) < 4:
            continue
        try:
            tone_vals.append(float(t[0]))
            polarity_vals.append(float(t[3]))
        except (ValueError, IndexError):
            continue
    return dict(
        n_articles=len(mx),
        avg_tone=np.mean(tone_vals) if tone_vals else np.nan,
        avg_polarity=np.mean(polarity_vals) if polarity_vals else np.nan,
    )


def build_monthly_gdelt():
    dates = sample_dates()
    rows = []
    for d in dates:
        urls = slice_urls_for_date(d)
        agg_articles, agg_tone_sum, agg_pol_sum = 0, 0.0, 0.0
        for url in urls:
            df = download_and_extract(url)
            res = mexico_tone_from_slice(df)
            if res is None or res["n_articles"] == 0:
                continue
            agg_articles += res["n_articles"]
            if not np.isnan(res["avg_tone"]):
                agg_tone_sum += res["avg_tone"] * res["n_articles"]
            if not np.isnan(res["avg_polarity"]):
                agg_pol_sum += res["avg_polarity"] * res["n_articles"]
        row = dict(
            fecha=pd.Timestamp(d.year, d.month, 1),
            n_articulos_mx=agg_articles,
            tono_promedio=agg_tone_sum / agg_articles if agg_articles > 0 else np.nan,
            polaridad_promedio=agg_pol_sum / agg_articles if agg_articles > 0 else np.nan,
        )
        rows.append(row)
        print(f"  {d.date()}: {agg_articles} articulos MX, tono={row['tono_promedio']:.3f}"
              if agg_articles > 0 else f"  {d.date()}: 0 articulos MX")

    out = pd.DataFrame(rows).set_index("fecha")
    out.to_csv(PROC / "gdelt_monthly.csv")
    return out


if __name__ == "__main__":
    monthly = build_monthly_gdelt()
    print(monthly.describe())
