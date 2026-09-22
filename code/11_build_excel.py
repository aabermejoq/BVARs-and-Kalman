"""
Construye el entregable en Excel: datos crudos -> transformaciones -> escenarios
epidemiologicos -> GDELT -> modelos M0-M3, con FORMULAS EN VIVO (no solo
valores pegados) donde es genuinamente posible, y graficas nativas de Excel
ligadas a esas celdas (si se cambia un dato de entrada, la grafica se mueve).

Que es formula en vivo y que es valor fijo, y por que:
- Transformaciones_MoM: formula en vivo (=LN(B3)-LN(B2), etc.) sobre
  Datos_Niveles.
- COVID_Escenarios: formula en vivo completa (g(t) por escenario, pesos de
  backcast, presion mensual acumulada) a partir de g0 y los parametros de
  cada escenario, todos en celdas editables.
- GDELT: formula en vivo (z-score contra la linea base 2017-2019).
- M0/M1/M2: VALORES pegados. Son salida de una simulacion de Monte Carlo
  (BVAR Minnesota con draws posteriores, Kalman+SV, filtro de particulas)
  que no es razonable reproducir como formulas de hoja de calculo. Se
  documenta esto explicitamente en la hoja.
- M3: formula en vivo = M2 (valor) + inclinacion/varianza de escenarios
  epidemiologicos (formula, referenciando COVID_Escenarios) + inclinacion/
  varianza de GDELT (formula, referenciando GDELT) -- ESTA es la capa que
  el usuario pidio que se mueva si se cambian los datos, y la que fuerza la
  entrada de GDELT sin importar su validacion cruzada (documentada aparte).
"""
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.chart import AreaChart, LineChart, Reference, Series
from openpyxl.chart.marker import Marker
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

m0 = import_module("03_M0_bvar_minnesota")
epi = import_module("06_M3_epidemic_scenarios")
m3c = import_module("09_M3_combine")

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
NOTE_FONT = Font(italic=True, color="595959", size=9)
TITLE_FONT = Font(bold=True, size=16)
SUB_FONT = Font(size=11, color="404040")
PCT_FMT = "0.00%"
NUM_FMT = "#,##0.00"


def style_header(ws, row, ncols, start_col=1):
    for c in range(start_col, start_col + ncols):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ---------------------------------------------------------------------------
def build():
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws_port = wb.create_sheet("Portada")
    ws_niv = wb.create_sheet("Datos_Niveles")
    ws_mom = wb.create_sheet("Transformaciones_MoM")
    ws_epi = wb.create_sheet("COVID_Escenarios")
    ws_gdelt = wb.create_sheet("GDELT")
    ws_mod = wb.create_sheet("Modelos")
    ws_graf = wb.create_sheet("Graficas")
    ws_meta = wb.create_sheet("Metodologia")

    build_portada(ws_port)
    build_niveles(ws_niv)
    build_transformaciones(ws_mom)
    build_covid_escenarios(ws_epi)
    build_gdelt(ws_gdelt)
    build_modelos(ws_mod, ws_epi, ws_gdelt)
    build_graficas(ws_graf, ws_mod)
    build_metodologia(ws_meta)

    out_path = OUT / "IGAE_escalera_2020.xlsx"
    wb.save(out_path)
    print(f"Guardado: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
def build_portada(ws):
    ws["B2"] = "IGAE México — Escalera de pronóstico 2020"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "M0 (BVAR Minnesota) → M1 (+Kalman/SV) → M2 (+Partículas/LP) → M3 (+Escenarios epi. + GBM + GDELT)"
    ws["B3"].font = SUB_FONT
    ws["B5"] = "Corte de información:"
    ws["C5"] = "15 de junio de 2020 (15 días tras el cierre de mayo)"
    ws["B6"] = "Último IGAE oficial conocido al corte:"
    ws["C6"] = "marzo 2020 (vintage real, rezago de publicación 56 días)"
    ws["B7"] = "Horizonte de pronóstico:"
    ws["C7"] = "abril → diciembre 2020"

    notes = [
        "Cómo leer este archivo:",
        "1) Datos_Niveles: series mensuales reales (INEGI/Banxico/Yahoo Finance/FRED), en niveles.",
        "2) Transformaciones_MoM: fórmulas EN VIVO (log-diferencia o diferencia simple según la naturaleza de cada serie) sobre Datos_Niveles.",
        "3) COVID_Escenarios: 4 curvas de crecimiento semanal de casos (fórmulas en vivo desde g0 real y parámetros editables por escenario), con pesos bayesianos de backcast.",
        "4) GDELT: tono/volumen mensual de noticias sobre México (dato real, GKG 2.1), con z-score en vivo contra su línea base 2017-2019.",
        "5) Modelos: M0/M1/M2 son SALIDA DE SIMULACIÓN (Monte Carlo bayesiano en Python, no reproducible como fórmulas de hoja de cálculo) y se pegan como valores. M3 SÍ es fórmula en vivo: M2 + ajuste por escenarios epidemiológicos + ajuste por GDELT — cambia si usted cambia g0, los parámetros de escenario, los pesos, o cualquier dato de GDELT.",
        "6) Graficas: 4 gráficas de banda (una por capa) más una gráfica comparativa de medianas, todas ligadas a la hoja Modelos.",
        "7) Metodologia: validación cruzada de variables, calendario de publicación (vintage), y hallazgos honestos — incluida la validación NEGATIVA de GDELT y del puente lineal, que se incluyeron en el modelo por instrucción expresa a pesar de no validar en CV.",
    ]
    r = 10
    for n in notes:
        ws.cell(row=r, column=2, value=n)
        if r == 10:
            ws.cell(row=r, column=2).font = Font(bold=True)
        r += 1

    autosize(ws, [3, 14, 90])
    for row in range(11, r):
        ws.cell(row=row, column=2).alignment = Alignment(wrap_text=False)


# ---------------------------------------------------------------------------
LEVELS_START = "2015-01-01"
LEVELS_END = "2020-12-01"
LEVEL_COLS = ["IGAE", "ActividadIndustrial", "FBCF", "IMCP", "ANTAD", "AUTOS",
              "TARJETAS", "Desempleo", "IMSS", "EPU", "INPC", "TC", "TIIE",
              "Balanza", "BMV", "INDPRO"]


def build_niveles(ws):
    panel = pd.read_csv(PROC / "panel_monthly_levels.csv", index_col=0, parse_dates=True)
    sub = panel.loc[LEVELS_START:LEVELS_END, LEVEL_COLS]

    ws.cell(row=1, column=1, value="Fecha")
    for j, col in enumerate(LEVEL_COLS, start=2):
        ws.cell(row=1, column=j, value=col)
    style_header(ws, 1, len(LEVEL_COLS) + 1)

    for i, (date, row) in enumerate(sub.iterrows(), start=2):
        ws.cell(row=i, column=1, value=date.strftime("%Y-%m"))
        for j, col in enumerate(LEVEL_COLS, start=2):
            ws.cell(row=i, column=j, value=float(row[col]))

    ws.freeze_panes = "B2"
    autosize(ws, [10] + [14] * len(LEVEL_COLS))
    ws.cell(row=sub.shape[0] + 3, column=1,
            value="Fuente: INEGI BIE, Banxico SIE, Yahoo Finance (^MXX), FRED (INDPRO). Dato real, sin transformar.")
    ws.cell(row=sub.shape[0] + 3, column=1).font = NOTE_FONT


# ---------------------------------------------------------------------------
def build_transformaciones(ws):
    from importlib import import_module
    dp = import_module("00_data_prep")

    ws.cell(row=1, column=1, value="Fecha")
    for j, col in enumerate(LEVEL_COLS, start=2):
        rule = ("log-diff" if col in dp.LOG_DIFF_VARS else
                "diff simple" if col in dp.SIMPLE_DIFF_VARS else "diff nivel")
        ws.cell(row=1, column=j, value=f"{col}_mom ({rule})")
    style_header(ws, 1, len(LEVEL_COLS) + 1)

    n_rows = 72  # debe coincidir con Datos_Niveles
    for i in range(2, n_rows + 2):
        ws.cell(row=i, column=1, value=f"=Datos_Niveles!A{i}")
        for j, col in enumerate(LEVEL_COLS, start=2):
            col_letter = get_column_letter(j)
            if i == 2:
                continue  # sin mes previo
            if col in dp.LOG_DIFF_VARS:
                f = f"=LN(Datos_Niveles!{col_letter}{i})-LN(Datos_Niveles!{col_letter}{i - 1})"
            else:
                f = f"=Datos_Niveles!{col_letter}{i}-Datos_Niveles!{col_letter}{i - 1}"
            cell = ws.cell(row=i, column=j, value=f)
            cell.number_format = PCT_FMT

    ws.freeze_panes = "B2"
    autosize(ws, [10] + [16] * len(LEVEL_COLS))
    ws.cell(row=n_rows + 3, column=1,
            value="Regla: índices/niveles de actividad → log-diferencia mensual; tasas ya en % → diferencia simple; "
                  "balanza/EPU (pueden ser negativos) → diferencia simple en nivel. Fórmulas EN VIVO sobre Datos_Niveles.")
    ws.cell(row=n_rows + 3, column=1).font = NOTE_FONT


# ---------------------------------------------------------------------------
CUTOFF = pd.Timestamp("2020-06-15")


def build_covid_escenarios(ws):
    weekly_new, g = epi.observed_weekly_growth(n_weeks=5)
    g_recent = g.iloc[-4:]
    g0_val = float(g.iloc[-1])

    ws["A1"] = "Escenarios epidemiológicos — crecimiento semanal de casos, México"
    ws["A1"].font = Font(bold=True, size=13)

    ws["A3"] = "g0 (tasa de crecimiento semanal real al corte, dato observado)"
    ws["D3"] = g0_val
    ws["D3"].number_format = PCT_FMT
    ws["D3"].font = Font(bold=True)
    ws.cell(row=4, column=1, value="Fuente: Our World in Data, semana terminada 2020-06-14, log(casos_t/casos_t-1).")
    ws.cell(row=4, column=1).font = NOTE_FONT

    # --- parametros de escenario (editables) ---
    ws["A6"] = "Parámetros de escenario (editables)"
    ws["A6"].font = Font(bold=True)
    headers = ["Escenario", "g_inf (tasa largo plazo)", "tau (semanas)", "bump_semana", "bump_amplitud", "bump_ancho"]
    for j, h in enumerate(headers, start=1):
        ws.cell(row=7, column=j, value=h)
    style_header(ws, 7, len(headers))
    scen_names = list(epi.SCENARIOS.keys())
    scen_row0 = 8
    for i, name in enumerate(scen_names):
        p = epi.SCENARIOS[name]
        r = scen_row0 + i
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=p["g_inf"])
        ws.cell(row=r, column=3, value=p["tau"])
        ws.cell(row=r, column=4, value=p["bump_week"] if p["bump_week"] is not None else "")
        ws.cell(row=r, column=5, value=p["bump_amp"])
        ws.cell(row=r, column=6, value=p["bump_width"])
    # named refs for g0
    g0_ref = "$D$3"

    # --- tabla semanal: t=-4..28, g(t) por escenario (formula en vivo) ---
    tbl_header_row = scen_row0 + len(scen_names) + 2
    ws.cell(row=tbl_header_row, column=1, value="Semana (t, 0=corte)").font = Font(bold=True)
    for i, name in enumerate(scen_names):
        ws.cell(row=tbl_header_row, column=2 + i, value=name)
    ws.cell(row=tbl_header_row, column=2 + len(scen_names), value="g_observado (real, solo t<0)")
    style_header(ws, tbl_header_row, len(scen_names) + 2)

    week_start, week_end = -4, 28
    first_week_row = tbl_header_row + 1
    for wi, t in enumerate(range(week_start, week_end + 1)):
        r = first_week_row + wi
        ws.cell(row=r, column=1, value=t)
        for si, name in enumerate(scen_names):
            prow = scen_row0 + si
            col = get_column_letter(2 + si)
            ginf = f"$B${prow}"
            tau = f"$C${prow}"
            bumpwk = f"$D${prow}"
            bumpamp = f"$E${prow}"
            bumpwidth = f"$F${prow}"
            base = f"{ginf}+({g0_ref}-{ginf})*EXP(-A{r}/{tau})"
            bump = f"+IF({bumpwk}=\"\",0,{bumpamp}*EXP(-0.5*((A{r}-{bumpwk})/{bumpwidth})^2))"
            f = f"=ROUND({base}{bump},6)"
            cell = ws.cell(row=r, column=2 + si, value=f)
            cell.number_format = "0.0000"
        if t < 0:
            gi = t + 4  # index into g_recent (0..3)
            obs_val = float(g_recent.iloc[gi])
            ws.cell(row=r, column=2 + len(scen_names), value=obs_val).number_format = "0.0000"

    last_week_row = first_week_row + (week_end - week_start)

    # --- pesos de backcast (formula en vivo, verosimilitud gaussiana t<0) ---
    weight_hdr_row = last_week_row + 2
    ws.cell(row=weight_hdr_row, column=1, value="Verosimilitud de backcast (t=-4..-1) y peso bayesiano").font = Font(bold=True)
    ws.cell(row=weight_hdr_row + 1, column=1, value="tau_verosimilitud")
    ws.cell(row=weight_hdr_row + 1, column=2, value=0.04)
    tau_lik_ref = f"$B${weight_hdr_row + 1}"

    ll_row = weight_hdr_row + 3
    ws.cell(row=ll_row, column=1, value="Escenario")
    ws.cell(row=ll_row, column=2, value="logLik backcast")
    ws.cell(row=ll_row, column=3, value="peso bayesiano")
    style_header(ws, ll_row, 3)
    obs_col_letter = get_column_letter(2 + len(scen_names))
    backcast_rows = f"{first_week_row}:{first_week_row + 3}"  # t=-4..-1
    for si, name in enumerate(scen_names):
        r = ll_row + 1 + si
        scen_col = get_column_letter(2 + si)
        scen_range = f"{scen_col}{first_week_row}:{scen_col}{first_week_row + 3}"
        obs_range = f"{obs_col_letter}{first_week_row}:{obs_col_letter}{first_week_row + 3}"
        f = f"=-0.5*SUMPRODUCT((({obs_range})-({scen_range}))^2)/{tau_lik_ref}^2"
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=f)
    ll_first, ll_last = ll_row + 1, ll_row + len(scen_names)
    max_ll_ref = f"MAX($B${ll_first}:$B${ll_last})"
    for si in range(len(scen_names)):
        r = ll_row + 1 + si
        f = f"=EXP(B{r}-{max_ll_ref})/SUMPRODUCT(EXP($B${ll_first}:$B${ll_last}-{max_ll_ref}))"
        ws.cell(row=r, column=3, value=f).number_format = PCT_FMT

    weights_range = f"$C${ll_first}:$C${ll_last}"
    scen_name_range = f"$A${ll_first}:$A${ll_last}"

    # --- presion mensual acumulada por escenario (formula en vivo) ---
    month_hdr_row = ll_row + len(scen_names) + 3
    ws.cell(row=month_hdr_row, column=1, value="Presión de casos mensual (log-nivel acumulado relativo al corte)").font = Font(bold=True)
    ws.cell(row=month_hdr_row + 1, column=1, value="Mes")
    for si, name in enumerate(scen_names):
        ws.cell(row=month_hdr_row + 1, column=2 + si, value=name)
    ws.cell(row=month_hdr_row + 1, column=2 + len(scen_names), value="semana efectiva (MAX(0,·))")
    style_header(ws, month_hdr_row + 1, len(scen_names) + 2)

    months = pd.date_range("2020-04-01", "2020-12-01", freq="MS")
    first_month_row = month_hdr_row + 2
    for mi, mth in enumerate(months):
        r = first_month_row + mi
        month_end = mth + pd.offsets.MonthEnd(0)
        weeks_since_cutoff = (month_end - CUTOFF).days / 7.0
        ws.cell(row=r, column=1, value=mth.strftime("%Y-%m"))
        eff_col = get_column_letter(2 + len(scen_names))
        ws.cell(row=r, column=2 + len(scen_names), value=round(weeks_since_cutoff, 3))
        for si, name in enumerate(scen_names):
            scen_col = get_column_letter(2 + si)
            fwd_range = f"{scen_col}${first_week_row + 4}:{scen_col}${last_week_row}"  # solo t>=0
            fwd_weeks = f"$A${first_week_row + 4}:$A${last_week_row}"
            # meses cuyo fin de mes cae ANTES del corte (abril, mayo) no tienen
            # presion epidemica futura todavia -> 0; de lo contrario, suma
            # acumulada de g(t) desde la semana 0 hasta la semana efectiva.
            f = (f"=IF({eff_col}{r}<0,0,SUMIF({fwd_weeks},\"<=\"&{eff_col}{r},{fwd_range}))")
            ws.cell(row=r, column=2 + si, value=f).number_format = "0.0000"

    last_month_row = first_month_row + len(months) - 1

    autosize(ws, [30] + [16] * (len(scen_names) + 1))

    ws._refs = dict(
        weights_range=weights_range, scen_name_range=scen_name_range,
        month_first=first_month_row, month_last=last_month_row,
        scen_names=scen_names, month_col0=2,
        eff_week_col=2 + len(scen_names),
    )


# ---------------------------------------------------------------------------
def build_gdelt(ws):
    g = pd.read_csv(PROC / "gdelt_monthly.csv", index_col=0, parse_dates=True)

    ws["A1"] = "GDELT GKG 2.1 — tono de noticias sobre México (dato real, muestreo 2 slices/día, 1 día/mes)"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = ("Muestreo reducido respecto al plan original (4-8 slices/día) por costo de tiempo de cómputo en la sesión. "
                "CV5 R² contra el residuo del VAR: -0.21 (media) / -0.19 (varianza) — NO validó, pero se incluye en el "
                "modelo por instrucción expresa (ver Modelos!M3 y Metodologia).")
    ws["A2"].font = NOTE_FONT

    headers = ["Fecha", "n_articulos_MX", "tono_promedio", "polaridad_promedio"]
    hdr_row = 4
    for j, h in enumerate(headers, start=1):
        ws.cell(row=hdr_row, column=j, value=h)
    style_header(ws, hdr_row, len(headers))

    first_row = hdr_row + 1
    for i, (date, row) in enumerate(g.iterrows()):
        r = first_row + i
        ws.cell(row=r, column=1, value=date.strftime("%Y-%m"))
        ws.cell(row=r, column=2, value=float(row["n_articulos_mx"]))
        ws.cell(row=r, column=3, value=float(row["tono_promedio"]))
        ws.cell(row=r, column=4, value=float(row["polaridad_promedio"]))
    last_row = first_row + len(g) - 1

    baseline_mask = (g.index >= "2017-01-01") & (g.index <= "2019-12-31")
    baseline_first = first_row + np.argmax(baseline_mask)
    baseline_last = first_row + len(baseline_mask) - 1 - np.argmax(baseline_mask[::-1])

    ws.cell(row=last_row + 2, column=1, value="Línea base 2017-2019 (media)")
    ws.cell(row=last_row + 2, column=3, value=f"=AVERAGE(C{baseline_first}:C{baseline_last})")
    ws.cell(row=last_row + 3, column=1, value="Línea base 2017-2019 (desv. estándar)")
    ws.cell(row=last_row + 3, column=3, value=f"=STDEV(C{baseline_first}:C{baseline_last})")
    mean_ref = f"$C${last_row + 2}"
    std_ref = f"$C${last_row + 3}"

    ws.cell(row=hdr_row, column=5, value="z_tono (vs. línea base)")
    style_header(ws, hdr_row, 1, start_col=5)
    for i in range(len(g)):
        r = first_row + i
        f = f"=(C{r}-{mean_ref})/{std_ref}"
        ws.cell(row=r, column=5, value=f).number_format = "0.00"

    autosize(ws, [10, 16, 16, 18, 16])

    ws._refs = dict(first_row=first_row, last_row=last_row, mean_ref=mean_ref, std_ref=std_ref)


# ---------------------------------------------------------------------------
def build_modelos(ws, ws_epi, ws_gdelt):
    months = pd.date_range("2020-04-01", "2020-12-01", freq="MS")
    layers = ["M0", "M1", "M2", "M3"]
    stats = ["p05", "p25", "mediana", "p75", "p95"]

    summaries = {L: pd.read_csv(PROC / f"{L}_summary.csv", index_col=0, parse_dates=True) for L in ["M0", "M1", "M2"]}
    meta = json.loads((PROC / "M3_meta.json").read_text())
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    igae_obs = panel["IGAE_mom"]

    ws["A1"] = "Modelos M0-M3 — mediana y bandas de incertidumbre, abril-diciembre 2020"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = ("M0/M1/M2 = salida de simulación Monte Carlo (Python, valores pegados). "
                "M3 = fórmula EN VIVO: M2 + ajuste por escenarios epidemiológicos (hoja COVID_Escenarios) "
                "+ ajuste por GDELT (hoja GDELT), con coeficientes editables abajo.")
    ws["A2"].font = NOTE_FONT

    ws["A4"] = "Coeficientes de M3 (editables)"
    ws["A4"].font = Font(bold=True)
    coef_labels = ["TILT_COEF (inclinación por escenario epi, fracción de sqrt(s0))",
                   "VAR_COEF (varianza por dispersión de escenarios)",
                   "GDELT_TILT_COEF (inclinación por z-score GDELT)",
                   "GDELT_VAR_COEF (varianza por |z-score| GDELT)",
                   "s0 (varianza del residuo más reciente del VAR, marzo 2020 — ancla real)"]
    coef_vals = [meta["tilt_coef"], meta["var_coef"], meta["gdelt_tilt_coef"], meta["gdelt_var_coef"], meta["s0"]]
    coef_cells = {}
    for i, (lbl, val) in enumerate(zip(coef_labels, coef_vals)):
        r = 5 + i
        ws.cell(row=r, column=1, value=lbl)
        ws.cell(row=r, column=3, value=val)
        coef_cells[lbl.split(" ")[0]] = f"$C${r}"

    hdr_row = 11
    ws.cell(row=hdr_row, column=1, value="Fecha")
    col = 2
    layer_cols = {}
    for L in layers:
        layer_cols[L] = col
        for s in stats:
            ws.cell(row=hdr_row, column=col, value=f"{L}_{s}")
            col += 1
    obs_col = col
    ws.cell(row=hdr_row, column=obs_col, value="IGAE_observado")
    style_header(ws, hdr_row, obs_col)

    first_row = hdr_row + 1
    epi_refs = ws_epi._refs
    gdelt_refs = ws_gdelt._refs

    for mi, mth in enumerate(months):
        r = first_row + mi
        ws.cell(row=r, column=1, value=mth.strftime("%Y-%m"))
        for L in ["M0", "M1", "M2"]:
            base = layer_cols[L]
            row_data = summaries[L].loc[mth]
            for si, s in enumerate(stats):
                cell = ws.cell(row=r, column=base + si, value=float(row_data[s]))
                cell.number_format = PCT_FMT

        # --- M3: formula en vivo ---
        epi_month_row = epi_refs["month_first"] + mi
        n_scen = len(epi_refs["scen_names"])
        pressures_range = f"COVID_Escenarios!B{epi_month_row}:{get_column_letter(1 + n_scen)}{epi_month_row}"
        ref_pressure = f"MIN({pressures_range})"
        # weights_range en COVID_Escenarios es vertical (un escenario por fila);
        # pressures_range es horizontal (un escenario por columna) -- TRANSPOSE
        # para que SUMPRODUCT multiplique elemento a elemento sin error de forma.
        weights_range = f"TRANSPOSE(COVID_Escenarios!{epi_refs['weights_range']})"

        # localizar la fila de GDELT por fecha (MATCH), no por offset fijo -- robusto a cambios
        gdelt_row_formula_lookup = f"MATCH(A{r},GDELT!$A${gdelt_refs['first_row']}:$A${gdelt_refs['last_row']},0)"
        z_gdelt = f"INDEX(GDELT!$E${gdelt_refs['first_row']}:$E${gdelt_refs['last_row']},{gdelt_row_formula_lookup})"

        tilt_epi = f"-{coef_cells['TILT_COEF']}*SQRT({coef_cells['s0']})*SUMPRODUCT({weights_range},TANH({pressures_range}))"
        var_epi = f"{coef_cells['VAR_COEF']}*SQRT({coef_cells['s0']})*SUMPRODUCT({weights_range},TANH(ABS({pressures_range}-{ref_pressure})))"
        tilt_gdelt = f"-{coef_cells['GDELT_TILT_COEF']}*SQRT({coef_cells['s0']})*TANH({z_gdelt})"
        var_gdelt = f"{coef_cells['GDELT_VAR_COEF']}*SQRT({coef_cells['s0']})*TANH(ABS({z_gdelt}))"
        total_tilt = f"({tilt_epi})+({tilt_gdelt})"
        total_sd = f"SQRT(({var_epi})^2+({var_gdelt})^2)"

        m2_base = layer_cols["M2"]
        m2_col = {s: get_column_letter(m2_base + si) for si, s in enumerate(stats)}
        m3_formulas = {
            "p05": f"={m2_col['p05']}{r}+{total_tilt}-1.645*{total_sd}",
            "p25": f"={m2_col['p25']}{r}+{total_tilt}-0.675*{total_sd}",
            "mediana": f"={m2_col['mediana']}{r}+{total_tilt}",
            "p75": f"={m2_col['p75']}{r}+{total_tilt}+0.675*{total_sd}",
            "p95": f"={m2_col['p95']}{r}+{total_tilt}+1.645*{total_sd}",
        }
        base = layer_cols["M3"]
        for si, s in enumerate(stats):
            cell = ws.cell(row=r, column=base + si, value=m3_formulas[s])
            cell.number_format = PCT_FMT

        obs_val = igae_obs.get(mth, None)
        if obs_val is not None and pd.notna(obs_val):
            ws.cell(row=r, column=obs_col, value=float(obs_val)).number_format = PCT_FMT

    ws.freeze_panes = get_column_letter(2) + str(first_row)
    autosize(ws, [10] + [11] * (obs_col - 1))
    ws._refs = dict(hdr_row=hdr_row, first_row=first_row, last_row=first_row + len(months) - 1,
                     layer_cols=layer_cols, obs_col=obs_col, stats=stats)


def build_graficas(ws, ws_mod):
    refs = ws_mod._refs
    hdr_row, first_row, last_row = refs["hdr_row"], refs["first_row"], refs["last_row"]
    layer_cols, obs_col, stats = refs["layer_cols"], refs["obs_col"], refs["stats"]

    ws["A1"] = "Escalera de pronóstico — M0 a M3 (todas ligadas a la hoja Modelos)"
    ws["A1"].font = Font(bold=True, size=13)

    # --- Grafica 1: comparacion de medianas + observado (todas las capas) ---
    lc = LineChart()
    lc.title = "Mediana por capa vs. IGAE observado"
    lc.y_axis.numFmt = "0%"
    lc.height = 9
    lc.width = 22
    cats = Reference(ws_mod, min_col=1, min_row=first_row, max_row=last_row)
    for L in ["M0", "M1", "M2", "M3"]:
        med_col = layer_cols[L] + 2  # orden stats: p05,p25,mediana,p75,p95 -> mediana es indice 2
        ref = Reference(ws_mod, min_col=med_col, min_row=hdr_row, max_row=last_row)
        lc.add_data(ref, titles_from_data=True)
    obs_ref = Reference(ws_mod, min_col=obs_col, min_row=hdr_row, max_row=last_row)
    lc.add_data(obs_ref, titles_from_data=True)
    lc.set_categories(cats)
    for i, s in enumerate(lc.series):
        s.smooth = False
        s.marker = Marker(symbol="circle", size=5)
    lc.series[-1].graphicalProperties.line.width = 28000
    lc.series[-1].graphicalProperties.line.solidFill = "000000"
    lc.series[-1].marker.graphicalProperties.solidFill = "000000"
    ws.add_chart(lc, "A3")

    # --- Graficas 2-5: banda por capa (p05-p95 sombreado, mediana, observado) ---
    layer_titles = {
        "M0": "M0 · BVAR Minnesota",
        "M1": "M0+M1 · + Kalman / volatilidad estocástica",
        "M2": "M0+M1+M2 · + Partículas / Local Projections",
        "M3": "M0+M1+M2+M3 · + Escenarios epi. / GBM / GDELT",
    }
    anchors = ["A22", "L22", "A40", "L40"]
    fills = ["BDD7EE", "9DC3E6", "5B9BD5", "2E5395"]
    for idx, L in enumerate(["M0", "M1", "M2", "M3"]):
        base = layer_cols[L]
        lc2 = LineChart()
        lc2.title = layer_titles[L]
        lc2.y_axis.numFmt = "0%"
        lc2.height = 9
        lc2.width = 16
        cats2 = Reference(ws_mod, min_col=1, min_row=first_row, max_row=last_row)
        for si, s in enumerate(["p05", "p95", "mediana"]):
            col = base + si if s != "mediana" else base + 2
            col = base + {"p05": 0, "p95": 4, "mediana": 2}[s]
            ref = Reference(ws_mod, min_col=col, min_row=hdr_row, max_row=last_row)
            lc2.add_data(ref, titles_from_data=True)
        obs_ref2 = Reference(ws_mod, min_col=obs_col, min_row=hdr_row, max_row=last_row)
        lc2.add_data(obs_ref2, titles_from_data=True)
        lc2.set_categories(cats2)
        colors = ["A6A6A6", "A6A6A6", fills[idx], "000000"]
        widths = [10000, 10000, 26000, 26000]
        dashed = [True, True, False, False]
        for si, s in enumerate(lc2.series):
            s.smooth = False
            s.graphicalProperties.line.solidFill = colors[si]
            s.graphicalProperties.line.width = widths[si]
            if dashed[si]:
                s.graphicalProperties.line.dashStyle = "dash"
            s.marker = Marker(symbol="none")
        lc2.series[-1].marker = Marker(symbol="circle", size=5)
        lc2.series[-1].marker.graphicalProperties.solidFill = "000000"
        ws.add_chart(lc2, anchors[idx])

    ws["A58"] = ("Nota: por limitación de la librería usada para graficar (openpyxl), las bandas se muestran como "
                 "P5 y P95 en líneas punteadas (no como área sombreada continua); los valores exactos de P25/P75 "
                 "están en la hoja Modelos. Mediana en línea gruesa de color, IGAE observado en negro.")
    ws["A58"].font = NOTE_FONT


# ---------------------------------------------------------------------------
def build_metodologia(ws):
    ws["A1"] = "Metodología, validaciones y hallazgos honestos"
    ws["A1"].font = Font(bold=True, size=13)

    r = 3
    ws.cell(row=r, column=1, value="1) Selección de variables — CV10 R² por rezago (0/1/2 meses)").font = Font(bold=True)
    r += 1
    vs = pd.read_csv(PROC / "variable_selection_full.csv")
    piv = vs.pivot(index="variable", columns="rezago_meses", values="cv10_r2")
    ws.cell(row=r, column=1, value="Variable")
    ws.cell(row=r, column=2, value="Rezago 0")
    ws.cell(row=r, column=3, value="Rezago 1")
    ws.cell(row=r, column=4, value="Rezago 2")
    style_header(ws, r, 4)
    r += 1
    for var, row in piv.iterrows():
        ws.cell(row=r, column=1, value=var)
        for j, lag in enumerate([0, 1, 2]):
            c = ws.cell(row=r, column=2 + j, value=float(row[lag]))
            c.number_format = "0.0000"
        r += 1
    r += 1
    ws.cell(row=r, column=1, value=("Hallazgo: solo ActividadIndustrial (R²≈0.50, pero mismo rezago de publicación "
                                     "que IGAE — no sirve para nowcasting) e IMSS (R²≈0.06, rezago 12 días) superan "
                                     "cero. El resto no muestra señal lineal univariada positiva."))
    ws.cell(row=r, column=1).font = NOTE_FONT
    r += 3

    ws.cell(row=r, column=1, value="2) Calendario de publicación (vintage)").font = Font(bold=True)
    r += 1
    vintage = import_module("01_vintage_calendar")
    ws.cell(row=r, column=1, value="Variable")
    ws.cell(row=r, column=2, value="Rezago (días)")
    ws.cell(row=r, column=3, value="Confianza")
    ws.cell(row=r, column=4, value="Nota")
    style_header(ws, r, 4)
    r += 1
    for var, entry in vintage.VINTAGE_CALENDAR.items():
        ws.cell(row=r, column=1, value=var)
        ws.cell(row=r, column=2, value=entry.lag_dias)
        ws.cell(row=r, column=3, value=entry.confianza)
        ws.cell(row=r, column=4, value=entry.nota)
        r += 1
    r += 1

    ws.cell(row=r, column=1, value="3) Validación de puentes (LOO/CV) — todas NEGATIVAS, incluidas igual por instrucción").font = Font(bold=True)
    r += 1
    meta = json.loads((PROC / "M3_meta.json").read_text())
    lines = [
        f"Puente LP (abril, ridge): LOO-CV R² = -0.0148",
        f"Puente LP (mayo, ridge): LOO-CV R² = -0.0129",
        f"GBM tarjetas (ANTAD/AUTOS/IMSS/TARJETAS): CV10 R² = -0.0477",
        f"GDELT (media del residuo VAR): CV5 R² = {meta['gdelt_r2_mean']:.4f}",
        f"GDELT (varianza del residuo VAR): CV5 R² = {meta['gdelt_r2_var']:.4f}",
        "Ninguno de estos modelos supera a la media histórica del residuo en validación honesta. "
        "El puente LP y GBM se dejan con su corrección conservadora (M2). GDELT se fuerza al ensamble "
        "de M3 por instrucción expresa del usuario, vía z-score de tono contra línea base 2017-2019, "
        "documentado aquí sin maquillar el resultado de su validación.",
    ]
    for line in lines:
        ws.cell(row=r, column=1, value=line)
        ws.cell(row=r, column=1).font = NOTE_FONT
        r += 1
    r += 1

    ws.cell(row=r, column=1, value="4) Pesos bayesianos de backcast de escenarios epidemiológicos").font = Font(bold=True)
    r += 1
    for k, v in meta["scenario_weights"].items():
        ws.cell(row=r, column=1, value=k)
        ws.cell(row=r, column=2, value=v).number_format = "0.00%"
        r += 1
    r += 1
    ws.cell(row=r, column=1, value=("Al corte (15-jun-2020) el crecimiento semanal de casos aún no desaceleraba "
                                     "(11-21% semanal en las 4 semanas previas). El escenario 'rebote_rapido' "
                                     "queda con peso ≈0 por ser inconsistente con esa trayectoria ya observada."))
    ws.cell(row=r, column=1).font = NOTE_FONT
    r += 3

    ws.cell(row=r, column=1, value="5) Corrección de vintage").font = Font(bold=True)
    r += 1
    ws.cell(row=r, column=1, value=("INDPRO de EEUU se recodificó de 45 a ~16 días de rezago, verificado contra el "
                                     "calendario real de publicación G.17 de la Reserva Federal "
                                     "(federalreserve.gov/releases/g17/). El supuesto anterior de 45 días nunca "
                                     "había sido confirmado con fuente primaria."))
    ws.cell(row=r, column=1).font = NOTE_FONT

    autosize(ws, [95, 16, 16, 60])


if __name__ == "__main__":
    build()
