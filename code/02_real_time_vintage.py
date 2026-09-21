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
    "IGAE": dict(freq="M", lag_days=58, confidence="media",
                 note="Indicador Global de la Actividad Economica: rezago tipico "
                      "de ~8 semanas (se publica ~3a semana del 2o mes siguiente)."),
    "ActividadIndustrial": dict(freq="M", lag_days=58, confidence="media",
                 note="Se publica junto con IGAE en el mismo comunicado del "
                      "Sistema de Cuentas Nacionales; se trata con el mismo "
                      "rezago conservador que IGAE."),
    "FBCF": dict(freq="M", lag_days=70, confidence="media",
                 note="Formacion Bruta de Capital Fijo tiene rezago de publicacion "
                      "mayor al de IGAE (tipicamente 10-11 semanas)."),
    "IMCP": dict(freq="M", lag_days=58, confidence="media",
                 note="Indicador Mensual del Consumo Privado en el Mercado "
                      "Interior (IMCPMI): forma parte del mismo paquete de "
                      "difusion que IGAE, mismo rezago."),
    "TasaDesempleo": dict(freq="M", lag_days=24, confidence="media",
                 note="ENOE mensual en condiciones normales tiene rezago de ~3-4 "
                      "semanas. Ver KNOWN_DISRUPTIONS: este rezago NO aplica "
                      "durante la suspension COVID de 2020."),
    "INPC": dict(freq="M", lag_days=10, confidence="alta",
                 note="INPC de quincena/mes completo se publica en los primeros "
                      "dias del mes siguiente."),
    "TIIE": dict(freq="D", lag_days=0, confidence="alta",
                 note="Tasa de mercado interbancario, disponible el mismo dia."),
    "TC": dict(freq="D", lag_days=1, confidence="alta",
                 note="Tipo de cambio FIX, disponible el mismo dia habil; el "
                      "promedio de un mes completo requiere esperar al cierre "
                      "de ese mes (+1 dia)."),
    "Exportaciones": dict(freq="M", lag_days=25, confidence="media",
                 note="Balanza comercial 'oportuna' de Banxico/INEGI, publicada "
                      "~25 dias despues del cierre del mes de referencia."),
    "Importaciones": dict(freq="M", lag_days=25, confidence="media",
                 note="Ver Exportaciones. Mismo comunicado, mismo rezago."),
    "ANTAD": dict(freq="M", lag_days=7, confidence="media",
                 note="Boletin propio de ANTAD (fuente privada), publicado en "
                      "los primeros dias habiles del mes siguiente. Calendario "
                      "exacto no verificado con fuente primaria en esta sesion."),
    "AUTOS": dict(freq="M", lag_days=10, confidence="media",
                 note="Venta/registro de vehiculos ligeros (INEGI/AMIA), "
                      "publicado en la primera quincena del mes siguiente."),
    "IMSS_empleos": dict(freq="M", lag_days=8, confidence="media",
                 note="IMSS publica su propio boletin de trabajadores asegurados "
                      "en la primera semana del mes siguiente: es de los "
                      "indicadores mas rapidos de la base."),
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
