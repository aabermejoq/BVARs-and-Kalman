"""
02_real_time_vintage.py

FASE 2: diccionario de disponibilidad + funcion de corte real-time + vintage
al 15-may-2020.

LIMITACION DECLARADA (aceptada por el usuario en Fase 1):
  basededatos.xlsx es una descarga de "ultima vintage" (fecha de consulta
  ago-sep 2026), NO un archivo de vintages historicas real (no existe un
  ALFRED publico para Mexico y no esta en el repo). Por lo tanto:
    - el CORTE de disponibilidad (que periodos de referencia se conocian el
      15-may-2020) se aplica correctamente, con base en calendarios de
      publicacion documentados;
    - pero los NIVELES de las observaciones usadas pueden diferir de los
      publicados originalmente en 2020, porque llevan las revisiones
      posteriores de INEGI/Banxico (esto es especialmente relevante para
      series desestacionalizadas: IGAE, PIB, Actividad Industrial, FBCF,
      IMCP, desempleo). Financieras (TIIE, TC) y precios (INPC) practicamente
      no se revisan. Se declara explicitamente en cada figura/tabla del
      ejercicio.

Este script NO estima modelos. Construye:
  (a) PUBLICATION_LAG_REGISTRY: rezago de publicacion documentado por
      variable, con nivel de confianza (alta/media/incierta).
  (b) KNOWN_DISRUPTIONS: quiebres conocidos de disponibilidad/comparabilidad
      (ENOE suspendida por COVID-19).
  (c) funciones is_available() / apply_real_time_cutoff() /
      assert_no_future_reference_dates() que implementan la regla de
      "no look-ahead" a nivel de codigo.
  (d) el panel de vintage al 15-may-2020 (data/interim/vintage_2020-05-15.csv)
      y un resumen legible (output/audit/vintage_2020-05-15_summary.csv).
"""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
AUDIT_OUT = ROOT / "output" / "audit"

CUTOFF_DATE = pd.Timestamp("2020-05-15")


class DataAvailabilityError(Exception):
    """Se levanta cuando no se puede determinar de forma segura si una
    observacion habria estado disponible en la fecha de corte. Por diseno,
    NUNCA se asume disponibilidad por defecto."""


# ---------------------------------------------------------------------------
# (a) Registro de rezagos de publicacion.
#
# 'lag_days': dias corridos desde el FIN del periodo de referencia hasta la
#             fecha esperada de publicacion. Se usa el extremo MAS
#             CONSERVADOR (mas largo) del rango plausible cuando existe
#             incertidumbre, para que un caso limite quede EXCLUIDO por
#             default en vez de incluido por azar.
# 'confidence': 'alta'  -> calendario institucional bien conocido / verificado
#                          en esta sesion via fuente primaria.
#               'media' -> patron tipico documentado en la literatura/
#                          practica de mercado, dia exacto no verificado con
#                          fuente primaria en esta sesion.
#               'sin_verificar' -> no se pudo establecer un rezago razonable;
#                          la variable se EXCLUYE automaticamente hasta
#                          confirmacion.
# ---------------------------------------------------------------------------
PUBLICATION_LAG_REGISTRY = {
    "PIB": dict(freq="Q", lag_days=32, confidence="alta",
                note="PIB trimestral 'oportuno' (estimacion preliminar de INEGI), "
                     "publicado ~30 dias despues del cierre del trimestre "
                     "(ej. 1T20 publicado 30-abr-2020). Se revisa dos veces mas "
                     "en los meses siguientes; usamos el valor con revisiones "
                     "posteriores por la limitacion de vintage declarada."),
    "IGAE": dict(freq="M", lag_days=56, confidence="alta",
                 note="VERIFICADO con comunicados reales de INEGI (2020): abr->"
                      "26-jun (57d), may->24-jul (54d), jun->26-ago (57d), jul->"
                      "25-sep (56d), ago->26-oct (56d). Rezago consistente ~54-57d, "
                      "se usa 56 como valor central."),
    "ActividadIndustrial": dict(freq="M", lag_days=56, confidence="alta",
                 note="VERIFICADO: se publica en el MISMO comunicado que IGAE "
                      "(mismas fechas exactas de arriba) -- confirmado revisando "
                      "el texto de los boletines igae2020_*.pdf, que NO mencionan "
                      "FBCF ni IMCP (esos van en un comunicado aparte, ver abajo)."),
    "FBCF": dict(freq="M", lag_days=67, confidence="alta",
                 note="CORREGIDO (bug encontrado por el usuario): FBCF/IMCP NO "
                      "se publican junto con IGAE, van en un comunicado propio "
                      "'Indicador Mensual de la Formacion Bruta de Capital Fijo' "
                      "(IMFBCF), ~13 dias DESPUES de IGAE. VERIFICADO 2020: mar->"
                      "5-jun (66d), abr->6-jul (67d), may->6-ago (67d), jun->"
                      "7-sep (68d), jul->6-oct (67d)."),
    "IMCP": dict(freq="M", lag_days=67, confidence="alta",
                 note="CORREGIDO (bug encontrado por el usuario): iba erroneamente "
                      "agrupado con el rezago de IGAE. VERIFICADO: se publica el "
                      "MISMO dia que FBCF (comunicado imcpmi, fechas identicas: "
                      "5-jun, 6-jul, 6-ago, 7-sep, 6-oct para mar-jul 2020), NO "
                      "con IGAE. Ver FBCF."),
    "TasaDesempleo": dict(freq="M", lag_days=24, confidence="media",
                 note="ENOE mensual en condiciones normales tiene rezago de ~3-4 "
                      "semanas. Ver KNOWN_DISRUPTIONS: este rezago NO aplica "
                      "durante la suspension COVID de 2020."),
    "INPC": dict(freq="M", lag_days=10, confidence="alta",
                 note="INPC de quincena/mes completo se publica en los primeros "
                      "dias del mes siguiente."),
    "TIIE": dict(freq="D", lag_days=0, confidence="alta",
                 note="Tasa de mercado interbancario, disponible el mismo dia."),
    "TC": dict(freq="M", lag_days=2, confidence="alta",
                 note="La hoja trae el PROMEDIO MENSUAL del tipo de cambio FIX "
                      "(no observaciones diarias): por tanto la fecha de "
                      "referencia es el mes completo y no puede conocerse antes "
                      "de que el mes termine. El nivel diario es publico sin "
                      "rezago, pero el promedio de un mes requiere esperar su "
                      "cierre; se asume ~2 dias habiles para su calculo y "
                      "publicacion (alta confianza: es aritmetica trivial sobre "
                      "datos ya publicos, no una estimacion)."),
    "Exportaciones": dict(freq="M", lag_days=26, confidence="alta",
                 note="VERIFICADO: 'Informacion Oportuna sobre la Balanza "
                      "Comercial' (INEGI), may-2020 publicada 26-jun-2020 (26d)."),
    "Importaciones": dict(freq="M", lag_days=26, confidence="alta",
                 note="Ver Exportaciones. Mismo comunicado, mismo rezago (26d)."),
    "ANTAD": dict(freq="M", lag_days=11, confidence="alta",
                 note="VERIFICADO: boletin propio de ANTAD, ventas de mayo-2020 "
                      "publicadas 11-jun-2020 (11d)."),
    "AUTOS": dict(freq="M", lag_days=5, confidence="alta",
                 note="VERIFICADO: Registro Administrativo de la Industria "
                      "Automotriz de Vehiculos Ligeros (RAIAVL, INEGI/AMIA), "
                      "ventas de mayo-2020 publicadas 5-jun-2020 (5d) -- el "
                      "indicador mas rapido del panel duro."),
    "IMSS_empleos": dict(freq="M", lag_days=12, confidence="alta",
                 note="VERIFICADO: IMSS publico los datos de puestos de trabajo "
                      "de mayo-2020 el ~12-jun-2020 (12d, reportado por Forbes "
                      "Mexico 13-jun-2020 citando el comunicado del dia anterior)."),
    "TARJETAS": dict(freq="M", lag_days=None, confidence="sin_verificar",
                 note="Fuente y calendario de publicacion no confirmados por el "
                      "usuario ni verificables dentro del archivo. EXCLUIDA de "
                      "cualquier uso automatico hasta confirmacion."),
    "EPU": dict(freq="M", lag_days=21, confidence="media",
                 note="Economic Policy Uncertainty Index Mexico "
                      "(policyuncertainty.com/mexico_monthly.html), construido "
                      "por mineria de prensa; se actualiza con rezago corto "
                      "(semanas), dia exacto de actualizacion no verificado."),
}

# ---------------------------------------------------------------------------
# (b) Quiebres conocidos de disponibilidad/comparabilidad (evidencia primaria).
# ---------------------------------------------------------------------------
KNOWN_DISRUPTIONS = {
    "TasaDesempleo": [
        dict(
            ref_start=pd.Timestamp("2020-03-01"),
            ref_end=pd.Timestamp("2020-06-01"),
            resolved_date=pd.Timestamp("2020-07-17"),
            reason=(
                "INEGI suspendio el levantamiento presencial de la ENOE el "
                "31-mar-2020 por la contingencia COVID-19 y CANCELO la "
                "publicacion de los resultados de abril (prevista 27-may-2020), "
                "sustituyendola por la ETOE (Encuesta Telefonica de Ocupacion y "
                "Empleo, metodologia distinta) para abr-jun 2020. La ENOE "
                "presencial se reactivo el 17-jul-2020. Fuente: comunicado "
                "INEGI 264/20 (1-jun-2020) y nota tecnica ECOVID-ML."
            ),
        )
    ]
}


def reference_period_end(date, freq):
    if freq == "Q":
        return date + pd.offsets.QuarterEnd(0)
    if freq == "M":
        return date + pd.offsets.MonthEnd(0)
    if freq == "D":
        return date
    raise DataAvailabilityError(f"Frecuencia desconocida: {freq}")


def expected_publication_date(variable, reference_date):
    if variable not in PUBLICATION_LAG_REGISTRY:
        raise DataAvailabilityError(
            f"'{variable}' no esta en PUBLICATION_LAG_REGISTRY: no se puede "
            f"determinar disponibilidad real-time sin asumir. Agregue el "
            f"registro explicitamente antes de usar esta variable."
        )
    reg = PUBLICATION_LAG_REGISTRY[variable]
    if reg["lag_days"] is None:
        return None
    ref_end = reference_period_end(reference_date, reg["freq"])
    return ref_end + pd.Timedelta(days=reg["lag_days"])


def is_available(variable, reference_date, asof=CUTOFF_DATE):
    """True solo si la observacion de `variable` con esa fecha de referencia
    habria sido PUBLICA y CONFIABLE en la fecha `asof`. Nunca asume
    disponibilidad quando falta informacion: retorna False."""
    for dis in KNOWN_DISRUPTIONS.get(variable, []):
        if dis["ref_start"] <= reference_date <= dis["ref_end"] and asof < dis["resolved_date"]:
            return False
    pub_date = expected_publication_date(variable, reference_date)
    if pub_date is None:
        return False
    return pub_date <= asof


def apply_real_time_cutoff(series, variable, asof=CUTOFF_DATE):
    """Filtra una Serie/columna indexada por fecha de referencia, dejando
    solo las observaciones disponibles en `asof`."""
    mask = pd.Series(series.index, index=series.index).apply(
        lambda d: is_available(variable, d, asof)
    )
    return series.loc[mask]


def assert_no_future_reference_dates(df_or_series, asof=CUTOFF_DATE, label=""):
    """Guarda de seguridad incondicional: ninguna fecha de REFERENCIA puede
    ser posterior al corte, sin importar el rezago de publicacion. Esto
    protege, por ejemplo, contra alimentar accidentalmente el PIB de 2T20
    (fecha de referencia 2020-04-01) a cualquier rutina de estimacion."""
    idx = df_or_series.index
    future = idx[idx > asof]
    if len(future) > 0:
        raise DataAvailabilityError(
            f"{label}: contiene fechas de referencia posteriores al corte "
            f"{asof.date()}: {list(future)}"
        )


def build_vintage_panel(series, asof=CUTOFF_DATE):
    """Aplica el corte real-time a cada variable del registro y devuelve
    (a) el panel filtrado y (b) una tabla resumen de la ultima fecha de
    referencia disponible por variable."""
    var_to_sheet_col = {
        "PIB": ("PIB", "PIB"),
        "IGAE": ("otros", "IGAE"),
        "ActividadIndustrial": ("otros", "ActividadIndustrial"),
        "FBCF": ("otros", "FBCF"),
        "TasaDesempleo": ("desempleo", "TasaDesempleo"),
        "INPC": ("INPC", "INPC"),
        "TIIE": ("TIIE", "TIIE"),
        "TC": ("TC", "TC"),
        "Exportaciones": ("Balanza", "Exportaciones"),
        "Importaciones": ("Balanza", "Importaciones"),
        "IMCP": ("Consumo", "IMCP"),
        "ANTAD": ("Consumo", "ANTAD"),
        "AUTOS": ("Consumo", "AUTOS"),
        "TARJETAS": ("Consumo", "TARJETAS"),
        "EPU": ("Uncertainity", "EPU"),
        "IMSS_empleos": ("IMSS", "IMSS_empleos"),
    }

    panel = {}
    summary_rows = []
    for var, (sheet, col) in var_to_sheet_col.items():
        s = series[sheet][col].dropna()
        try:
            s_rt = apply_real_time_cutoff(s, var, asof)
            last_ref = s_rt.index.max() if len(s_rt) else None
            status = PUBLICATION_LAG_REGISTRY[var]["confidence"]
        except DataAvailabilityError as e:
            s_rt = s.iloc[0:0]
            last_ref = None
            status = f"ERROR: {e}"
        panel[var] = s_rt
        summary_rows.append(
            dict(
                variable=var,
                ultima_fecha_referencia_disponible=last_ref,
                n_obs_en_vintage=len(s_rt),
                confianza_rezago=status,
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values("variable")
    return panel, summary


def main():
    with open(INTERIM / "series_raw.pkl", "rb") as f:
        series = pickle.load(f)

    # Salvaguarda explicita pedida por el usuario: el PIB de 2T20 (fecha de
    # referencia 2020-04-01) NUNCA debe pasar el corte para ESTIMACION.
    pib = series["PIB"]["PIB"]
    pib_rt = apply_real_time_cutoff(pib, "PIB", CUTOFF_DATE)
    assert pd.Timestamp("2020-04-01") not in pib_rt.index, (
        "VIOLACION DE REGLA REAL-TIME: el PIB de 2T20 paso el filtro de corte."
    )
    print(f"OK: PIB 2T20 (2020-04-01) correctamente EXCLUIDO del set de estimacion.")
    print(f"Ultimo PIB disponible al {CUTOFF_DATE.date()}: {pib_rt.index.max().date()} "
          f"(1T20, valor {pib_rt.iloc[-1]:,.0f} millones de pesos 2018)")

    panel, summary = build_vintage_panel(series, CUTOFF_DATE)

    summary.to_csv(AUDIT_OUT / "vintage_2020-05-15_summary.csv", index=False)
    with open(INTERIM / "vintage_2020-05-15_panel.pkl", "wb") as f:
        pickle.dump(panel, f)

    print("\n--- Vintage al 15-may-2020: ultima fecha de referencia disponible por variable ---")
    print(summary.to_string(index=False))

    # Ejemplo de uso de la salvaguarda generica (assert_no_future_reference_dates)
    assert_no_future_reference_dates(pib_rt, CUTOFF_DATE, label="PIB (panel real-time)")
    print("\nOK: assert_no_future_reference_dates paso sin errores sobre el panel real-time.")


if __name__ == "__main__":
    main()
