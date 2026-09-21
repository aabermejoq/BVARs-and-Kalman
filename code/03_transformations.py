"""
03_transformations.py

Construye el panel TRIMESTRAL del conjunto CORE = {PIB, INPC, TIIE, TC} con
seguridad real-time, usando el registro de disponibilidad de la Fase 2.

Regla de "trimestre completo": un trimestre solo se incluye para una
variable si TODAS sus observaciones constitutivas (3 meses, o todos los
dias habiles del trimestre para TIIE) estan dentro de lo disponible al
corte `asof`. Esto excluye automaticamente 2020Q2 de las 4 variables CORE
sin necesidad de reglas ad hoc por variable.

Transformaciones:
  - PIB: log(nivel) -- estandar en BVARs con prior Minnesota (RW en logs).
  - INPC: log(nivel) -- igual tratamiento que el resto de la literatura de
    BVARs grandes (Banbura-Giannone-Reichlin 2010 mantienen precios en logs,
    no en inflacion, y dejan que el prior Minnesota implique la dinamica).
  - TIIE: nivel (tasa de interes, no se transforma).
  - TC: log(nivel) -- estandar para tipos de cambio.

HALLAZGO IMPORTANTE (reportado explicitamente, no oculto): la TIIE de
Fondeo a 1 dia solo existe desde 2006-01-02 en el archivo. Por tanto el
panel CORE completo (las 4 variables juntas) solo puede empezar en 2006Q1,
NO en 1993Q1 como PIB/INPC/TC individualmente permitirian. Esto recorta la
muestra de M0 a 57 trimestres (2006Q1-2020Q1) y EXCLUYE la crisis Tequila
(1994-95) de la estimacion de M0 -- se documenta como limitacion, no se
inventa una tasa de interes previa a 2006 que no esta en la base.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

sys.path.insert(0, str(ROOT / "code"))
from importlib import import_module
rtv = import_module("02_real_time_vintage")

CUTOFF = rtv.CUTOFF_DATE


def quarterly_with_completeness(raw_series, variable, asof=CUTOFF, agg="mean"):
    """Trunca `raw_series` a lo disponible en `asof` (Fase 2) y agrega a
    trimestral, manteniendo SOLO los trimestres cuyo conteo de obs. dentro
    del trimestre coincide con el conteo en la serie SIN truncar (es decir,
    el trimestre ya habia terminado y se conocia por completo)."""
    raw = raw_series.dropna()
    rt = rtv.apply_real_time_cutoff(raw, variable, asof)

    raw_q_count = raw.resample("QS").count()
    rt_q_count = rt.resample("QS").count()
    rt_q_agg = rt.resample("QS").agg(agg)

    complete = (rt_q_count.reindex(raw_q_count.index, fill_value=0) == raw_q_count)
    complete = complete.reindex(rt_q_agg.index, fill_value=False)
    return rt_q_agg[complete]


def main():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    pib_q = quarterly_with_completeness(series["PIB"]["PIB"], "PIB", agg="mean")
    inpc_q = quarterly_with_completeness(series["INPC"]["INPC"], "INPC", agg="mean")
    tiie_q = quarterly_with_completeness(series["TIIE"]["TIIE"], "TIIE", agg="mean")
    tc_q = quarterly_with_completeness(series["TC"]["TC"], "TC", agg="mean")

    print("Ultimo trimestre completo disponible al 15-may-2020, por variable:")
    for name, s in [("PIB", pib_q), ("INPC", inpc_q), ("TIIE", tiie_q), ("TC", tc_q)]:
        print(f"  {name}: {s.index.min().date()} -> {s.index.max().date()}  (n={len(s)})")

    core = pd.DataFrame({
        "log_PIB": np.log(pib_q),
        "log_INPC": np.log(inpc_q),
        "TIIE": tiie_q,
        "log_TC": np.log(tc_q),
    })
    core_full_history = core.dropna(subset=["log_PIB", "log_INPC", "log_TC"])
    core_with_tiie = core.dropna()

    print(f"\nPanel CORE sin TIIE (log_PIB, log_INPC, log_TC): "
          f"{core_full_history.index.min().date()} -> {core_full_history.index.max().date()} "
          f"(n={len(core_full_history)})")
    print(f"Panel CORE completo (+TIIE): "
          f"{core_with_tiie.index.min().date()} -> {core_with_tiie.index.max().date()} "
          f"(n={len(core_with_tiie)})  <-- este es el usado en M0 (TIIE inicia 2006-01-02)")

    assert core_with_tiie.index.max() == pd.Timestamp("2020-01-01"), (
        "El ultimo trimestre CORE deberia ser 2020Q1 (2020-04-01 = 2T20 debe "
        "quedar excluido por la regla real-time)."
    )
    rtv.assert_no_future_reference_dates(core_with_tiie, CUTOFF, label="Panel CORE trimestral")

    core_with_tiie.to_csv(INTERIM / "core_quarterly_panel.csv")
    print(f"\nGuardado: {INTERIM / 'core_quarterly_panel.csv'}")
    print("\nUltimas 6 filas:")
    print(core_with_tiie.tail(6))


if __name__ == "__main__":
    main()
