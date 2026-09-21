"""
12_tarjetas_daily_audit.py

Auditoria y preparacion de data/raw/tarjetas_diario.xlsx (Banxico, diario,
32 categorias x {Total, Debito, Credito}), para su uso EXCLUSIVO en el
ejercicio de agosto/3T20 (M4), segun instruccion del usuario.

Supuestos declarados (dados por el usuario, no verificados independiente-
mente por mi):
  - Fuente: Banxico.
  - Publicacion: semanal (aunque el archivo trae granularidad diaria).
  - Series ORIGINALES (no desestacionalizadas) -- a diferencia del resto
    de la base, que si trae series SA cuando corresponde.
  - Disponibilidad real-time: rezago de 1 dia (supuesto de trabajo).

Tratamiento:
  - Se usa la columna "Total" de cada categoria (no el desglose
    debito/credito, para no multiplicar la dimensionalidad sin necesidad).
  - Para evitar el efecto de dia-de-la-semana (un problema real en datos
    diarios de consumo: martes de 2019 no es comparable a martes de 2020
    porque el numero de fin de semana en el rango cambia), se usa una
    comparacion interanual de VENTANA MOVIL de 7 dias (suma de los ultimos
    7 dias vs. la suma de los mismos 7 dias 364 dias antes) en vez de una
    variacion dia a dia. Esto es una forma simple y defendible de
    "desestacionalizar" sin recurrir a X-13, consistente con el
    tratamiento ya usado para ANTAD/AUTOS/IMSS (variacion interanual).
  - Clustering (k-means) de las categorias por la FORMA de su trayectoria
    de caida/recuperacion en 2020, para identificar arquetipos
    (contacto-intensivo/discrecional vs. esencial/sustituible) -- la parte
    de "ML genuino" pedida por el usuario. Esto es una CARACTERIZACION del
    choque (para el filtro bayesiano de M4-agosto), no se usa para nada en
    M2/M3.
"""
import pickle
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tarjetas_diario.xlsx"
INTERIM = ROOT / "data" / "interim"
AUDIT_OUT = ROOT / "output" / "audit"

LAG_DAYS = 1  # supuesto de disponibilidad real-time dado por el usuario


def load_daily():
    wb = openpyxl.load_workbook(RAW, data_only=True)
    ws = wb["Hoja1"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = [r for r in rows[1:] if r[0] is not None]
    df = pd.DataFrame(data, columns=header)
    df["fecha"] = pd.to_datetime(dict(year=df["year"], month=df["mes"], day=df["dia"]))
    df = df.set_index("fecha").sort_index()
    total_cols = [c for c in df.columns if c.startswith("Total de monto operado")]
    df = df[total_cols]
    df.columns = [c.replace("Total de monto operado a través de tarjetas en ", "")
                   .replace("Total de monto operado a través de tarjetas", "AGREGADO")
                  for c in df.columns]
    return df


def rolling_yoy(df):
    roll7 = df.rolling(7).sum()
    yoy = 100 * np.log(roll7 / roll7.shift(364))
    return yoy


def apply_real_time_cutoff(df, asof, lag_days=LAG_DAYS):
    last_allowed = asof - pd.Timedelta(days=lag_days)
    return df.loc[:last_allowed]


def cluster_categories(yoy, window=("2020-03-01", "2020-08-14"), k=3, exclude=("AGREGADO", "No definido")):
    cats = [c for c in yoy.columns if c not in exclude]
    traj = yoy.loc[window[0]:window[1], cats].dropna(axis=1, how="any")
    if traj.shape[1] < k:
        traj = yoy.loc[window[0]:window[1], cats].dropna(axis=1, thresh=int(0.9 * len(traj)))
        traj = traj.ffill().bfill()
    X = StandardScaler().fit_transform(traj.T.values)
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
    labels = pd.Series(km.labels_, index=traj.columns)
    cluster_means = traj.T.groupby(labels).mean().T  # trayectoria promedio de cada cluster
    return labels, cluster_means


def main():
    df = load_daily()
    print(f"Categorias: {len(df.columns)}  Rango: {df.index.min().date()} -> {df.index.max().date()}")

    yoy = rolling_yoy(df)

    asof = pd.Timestamp("2020-08-15")
    yoy_rt = apply_real_time_cutoff(yoy, asof)
    print(f"\nCorte real-time (rezago {LAG_DAYS} dia): ultima fecha disponible = {yoy_rt.index.max().date()}")

    print("\nUltimos 5 dias del agregado (a/a, ventana movil 7 dias):")
    print(yoy_rt["AGREGADO"].tail(5).round(2).to_string())

    labels, cluster_means = cluster_categories(yoy_rt)
    print("\n=== Clustering de categorias (k=3) por forma de trayectoria mar-ago 2020 ===")
    for cl in sorted(labels.unique()):
        miembros = labels[labels == cl].index.tolist()
        print(f"Cluster {cl}: {miembros}")

    # Identificar cual cluster es el mas golpeado (menor promedio en el periodo)
    peor_cluster = cluster_means.mean().idxmin()
    print(f"\nCluster mas golpeado: {peor_cluster} "
          f"(promedio a/a mar-ago 2020 = {cluster_means[peor_cluster].mean():.1f}%)")
    print("Categorias en el cluster mas golpeado:", labels[labels == peor_cluster].index.tolist())

    # Participacion de gasto del cluster mas golpeado en el gasto total (base pre-crisis)
    base = df.loc["2019-01-01":"2019-12-31"]
    miembros_peor = labels[labels == peor_cluster].index.tolist()
    participacion = base[miembros_peor].sum().sum() / base["AGREGADO"].sum()
    print(f"Participacion del cluster mas golpeado en el gasto total (base 2019): {participacion:.1%}")

    INTERIM.mkdir(parents=True, exist_ok=True)
    yoy.to_csv(INTERIM / "tarjetas_daily_yoy.csv")
    with open(INTERIM / "tarjetas_clusters.pkl", "wb") as f:
        pickle.dump(dict(labels=labels, cluster_means=cluster_means, peor_cluster=peor_cluster,
                          participacion=participacion), f)

    dict_rows = [dict(
        variable="tarjetas_diario (32 categorias + agregado)", fuente="Banxico (declarado por el usuario)",
        frecuencia="Diaria (agregada a ventana movil 7 dias para des-estacionalizar)",
        disponible_real_time="Supuesto: rezago de 1 dia (declarado por el usuario, no verificado independientemente)",
        cobertura="2009-01-01 a 2026-07-31, sin huecos",
        transformacion="Variacion interanual (a/a) de la suma movil de 7 dias",
        uso="EXCLUSIVO de M4-agosto (filtro bayesiano de escenarios); no se usa en M2/M3/M0/M1",
    )]
    pd.DataFrame(dict_rows).to_csv(AUDIT_OUT / "tarjetas_diario_dictionary.csv", index=False)
    print(f"\nGuardado: {INTERIM / 'tarjetas_daily_yoy.csv'}, {INTERIM / 'tarjetas_clusters.pkl'}")


if __name__ == "__main__":
    main()
