"""
Calendario de publicacion (vintage) por variable: cuantos dias despues del
cierre del mes de referencia esta disponible el dato.

Se usa para construir, dado un corte de informacion (ej. 15 dias despues de
cerrado mayo/junio 2020), el "ragged edge" real: que variables ya se conocen
para el mes t, cuales solo hasta t-1, t-2, etc.

Confianza de cada rezago:
  "alta_usuario"   -> dado explicitamente por el usuario como ya verificado
                      contra comunicados reales en una sesion anterior.
  "alta_sesion"    -> reverificado en ESTA sesion contra fuente primaria.
  "media_generica" -> conocimiento estandar del calendario de difusion
                       INEGI/Banxico, no reverificado linea por linea en esta
                       sesion (riesgo bajo pero no cero).
  "supuesto"       -> no verificado, tratar con cautela.

Caso INDPRO: el prompt original marcaba un supuesto de 45 dias "nunca
confirmado con fuente primaria". Se verifico en esta sesion: la Fed publica
el reporte G.17 (Industrial Production and Capacity Utilization) alrededor
de 15 dias despues del cierre del mes de referencia, no 45.
Fuente: https://www.federalreserve.gov/releases/g17/ (calendario de
publicacion 2026: 16-ene, 18-feb, 16-mar, 16-abr, 15-may, 15-jun, ...).
Se corrige aqui de 45 -> 16 dias (dato preliminar; se usa 16 por ser el
numero de dias tipico observado en el calendario 2026, redondeado al alza
sobre el ~15 citado en el G.17 Technical Q&A).

Caso IGAE: el usuario dio 56 dias (alta_usuario). Una busqueda independiente
en esta sesion sobre el calendario de difusion INEGI arrojo ~53 dias
naturales. No se pudo extraer el boletin PDF original para zanjar la
diferencia exacta (herramientas de PDF rotas en este entorno -- ver nota en
el codigo de 00_data_prep). Se usa el valor del usuario (56) como central
pero se documenta el rango [53, 56] como incertidumbre de +/-3 dias.
"""
from dataclasses import dataclass


@dataclass
class VintageEntry:
    variable: str
    lag_dias: int
    confianza: str
    nota: str


VINTAGE_CALENDAR = {
    "IGAE": VintageEntry("IGAE", 56, "alta_usuario",
        "INEGI. Dado por el usuario como verificado; chequeo independiente "
        "esta sesion via busqueda web sugiere ~53 dias (calendario de "
        "difusion INEGI). Rango de incertidumbre [53,56]."),
    "ActividadIndustrial": VintageEntry("ActividadIndustrial", 56, "alta_usuario",
        "INEGI, mismo comunicado que IGAE (se publican el mismo dia)."),
    "FBCF": VintageEntry("FBCF", 67, "alta_usuario",
        "INEGI, comunicado APARTE del de IGAE (no el mismo dia)."),
    "IMCP": VintageEntry("IMCP", 67, "alta_usuario",
        "Indicador Mensual del Consumo Privado, INEGI, mismo comunicado que FBCF."),
    "ANTAD": VintageEntry("ANTAD", 11, "alta_usuario",
        "Ventas mismas tiendas ANTAD, asociacion privada, publicacion rapida."),
    "TARJETAS": VintageEntry("TARJETAS", 11, "supuesto",
        "Indicador de consumo con tarjetas. Sin fuente primaria verificada "
        "esta sesion; se asume el mismo rezago que ANTAD (privado, rubro "
        "similar de consumo) como cota razonable -- tratar con cautela."),
    "AUTOS": VintageEntry("AUTOS", 5, "alta_usuario",
        "Registro Administrativo de la Industria Automotriz de Vehiculos "
        "Ligeros (INEGI/AMIA), de las publicaciones mas rapidas."),
    "IMSS": VintageEntry("IMSS", 12, "alta_usuario",
        "Puestos de trabajo afiliados al IMSS, publicacion administrativa rapida."),
    "Balanza": VintageEntry("Balanza", 26, "alta_usuario",
        "Balanza comercial, INEGI/Banxico."),
    "INDPRO": VintageEntry("INDPRO", 16, "alta_sesion",
        "Corregido en esta sesion: Federal Reserve G.17, ~15-17 dias segun "
        "calendario 2026 (fuente: federalreserve.gov/releases/g17/). "
        "El supuesto previo de 45 dias NO estaba confirmado y era incorrecto."),
    "BMV": VintageEntry("BMV", 0, "alta_sesion",
        "Indice bursatil, dato de mercado disponible el mismo dia (t+0)."),
    "TC": VintageEntry("TC", 1, "alta_sesion",
        "Tipo de cambio FIX, Banxico publica con 1 dia habil de rezago."),
    "TIIE": VintageEntry("TIIE", 1, "alta_sesion",
        "Banxico publica la tasa de referencia con 1 dia habil de rezago."),
    "Desempleo": VintageEntry("Desempleo", 25, "media_generica",
        "ENOE mensual, INEGI. Calendario tipico ~25 dias; NO reverificado "
        "contra un comunicado especifico en esta sesion -- tratar como "
        "estimacion, no como hecho verificado."),
    "INPC": VintageEntry("INPC", 9, "media_generica",
        "INPC quincenal/mensual, INEGI publica cifra mensual ~los primeros "
        "9-10 dias del mes siguiente. NO reverificado contra comunicado "
        "especifico en esta sesion."),
    "EPU": VintageEntry("EPU", 30, "supuesto",
        "Mexican Economic Policy Uncertainty Index (policyuncertainty.com). "
        "Sin verificar calendario real de actualizacion; se asume ~1 mes "
        "de rezago como cota conservadora."),
    "COVID_MX": VintageEntry("COVID_MX", 1, "alta_sesion",
        "Our World in Data agrega reportes diarios de Secretaria de Salud "
        "con ~1 dia de rezago."),
    "GDELT": VintageEntry("GDELT", 0, "alta_sesion",
        "GDELT GKG 2.1 se actualiza cada 15 minutos; disponible casi en "
        "tiempo real (t+0)."),
}


def info_set_at_cutoff(cutoff_date, reference_month_end):
    """
    Dado un corte (fecha en la que se genera el pronostico) y el fin del mes
    de referencia t, regresa para cada variable el ULTIMO mes m<=t para el
    cual el dato ya estaria publicado en el corte (ragged edge real).
    """
    import pandas as pd

    out = {}
    for var, entry in VINTAGE_CALENDAR.items():
        period = pd.Timestamp(reference_month_end).to_period("M")
        while True:
            month_end = period.to_timestamp("M")
            fecha_publicacion = month_end + pd.Timedelta(days=entry.lag_dias)
            if fecha_publicacion <= cutoff_date:
                out[var] = period
                break
            period = period - 1
            if period.year < 2015:
                out[var] = None
                break
    return out


if __name__ == "__main__":
    import pandas as pd

    cutoff = pd.Timestamp("2020-06-15")  # 15 dias despues de cerrado mayo 2020
    ref_month_end = pd.Timestamp("2020-05-31")
    disponible = info_set_at_cutoff(cutoff, ref_month_end)
    print(f"Corte: {cutoff.date()} (15 dias post-cierre de mayo 2020)")
    print(f"{'variable':<22}{'ultimo mes disponible':<25}{'confianza del rezago'}")
    for var, per in disponible.items():
        conf = VINTAGE_CALENDAR[var].confianza
        print(f"{var:<22}{str(per):<25}{conf}")
