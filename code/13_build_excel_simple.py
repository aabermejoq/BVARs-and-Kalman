"""
Excel simplificado: fecha, dato real, cada pronostico (mediana), y graficas
nativas de Excel ligadas a esos datos -- una comparativa sin intervalos y
4 graficas de abanico (una por capa) con banda de incertidumbre real
(P5-P95 y P25-P75 como area apilada, no lineas punteadas).

Todo se escribe como VALORES (no formulas) para evitar el problema de
caché vacío de openpyxl que rompio la version anterior en Excel real
--"vinculada a los datos" se cumple porque las graficas de Excel siempre
leen las celdas en vivo (si usted edita un numero en la hoja Datos, la
grafica se redibuja sola; eso no depende de formulas).
"""
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.chart import AreaChart, LineChart, Reference
from openpyxl.chart.legend import LegendEntry
from openpyxl.chart.marker import Marker
from openpyxl.chart.series import SeriesLabel
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

LAYERS = ["M0", "M1", "M2", "M3"]
LAYER_LABELS = {
    "M0": "M0 · BVAR Minnesota",
    "M1": "M0+M1 · + Lenza-Primiceri (2022)",
    "M2": "M0+M1+M2 · + Partículas (Gordon 1993)",
    "M3": "M0+M1+M2+M3 · + Waggoner-Zha (1999)",
}
# rampa secuencial azul (mismo hue, oscureciendo por capa) + tinta para "observado"
LAYER_COLOR = {"M0": "9EC5F4", "M1": "5598E7", "M2": "2A78D6", "M3": "104281"}
INK = "0B0D10"
BAND_LIGHT = {"M0": "DCEAFC", "M1": "C7DEF8", "M2": "B8D3F3", "M3": "A9C6E8"}

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=15)
NOTE_FONT = Font(italic=True, color="595959", size=9)
PCT_FMT = "0.00%"


def style_header(ws, row, ncols, start_col=1):
    for c in range(start_col, start_col + ncols):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def load_data():
    months = pd.date_range("2020-04-01", "2020-12-01", freq="MS")
    panel = pd.read_csv(PROC / "panel_monthly_mom.csv", index_col=0, parse_dates=True)
    observed = panel["IGAE_mom"].reindex(months)

    summaries = {L: pd.read_csv(PROC / f"{L}_summary.csv", index_col=0, parse_dates=True).reindex(months)
                 for L in LAYERS}
    return months, observed, summaries


def build_datos_sheet(ws, months, observed, summaries):
    ws["A1"] = "Serie principal (para la gráfica comparativa sin intervalos)"
    ws["A1"].font = Font(bold=True, size=12)
    hdr = ["Fecha", "IGAE observado", "M0", "M1", "M2", "M3"]
    for j, h in enumerate(hdr, start=1):
        ws.cell(row=2, column=j, value=h)
    style_header(ws, 2, len(hdr))

    for i, mth in enumerate(months):
        r = 3 + i
        ws.cell(row=r, column=1, value=mth.strftime("%Y-%m"))
        obs = observed.loc[mth]
        c = ws.cell(row=r, column=2, value=float(obs) if pd.notna(obs) else None)
        c.number_format = PCT_FMT
        for j, L in enumerate(LAYERS, start=3):
            c = ws.cell(row=r, column=j, value=float(summaries[L].loc[mth, "mediana"]))
            c.number_format = PCT_FMT

    last_row = 2 + len(months)
    autosize(ws, [10, 15, 11, 11, 11, 11])

    # --- seccion 2: percentiles y segmentos apilados por capa (para los abanicos) ---
    sec2_row = last_row + 3
    ws.cell(row=sec2_row, column=1, value="Percentiles por capa (para las gráficas de abanico)").font = Font(bold=True, size=12)
    sec2_row += 1

    layer_blocks = {}
    col = 1
    ws.cell(row=sec2_row, column=col, value="Fecha")
    col += 1
    for L in LAYERS:
        start_col = col
        headers = ["p05", "seg_25_05", "seg_75_25", "seg_95_75", "mediana", "p25", "p75", "p95"]
        for h in headers:
            ws.cell(row=sec2_row, column=col, value=f"{L}_{h}")
            col += 1
        layer_blocks[L] = start_col
    obs_col2 = col
    ws.cell(row=sec2_row, column=obs_col2, value="IGAE observado")
    style_header(ws, sec2_row, obs_col2)

    first_data_row = sec2_row + 1
    for i, mth in enumerate(months):
        r = first_data_row + i
        ws.cell(row=r, column=1, value=mth.strftime("%Y-%m"))
        for L in LAYERS:
            s = summaries[L].loc[mth]
            base = layer_blocks[L]
            vals = [
                s["p05"],
                s["p25"] - s["p05"],
                s["p75"] - s["p25"],
                s["p95"] - s["p75"],
                s["mediana"],
                s["p25"],
                s["p75"],
                s["p95"],
            ]
            for k, v in enumerate(vals):
                c = ws.cell(row=r, column=base + k, value=float(v))
                c.number_format = PCT_FMT
        obs = observed.loc[mth]
        if pd.notna(obs):
            ws.cell(row=r, column=obs_col2, value=float(obs)).number_format = PCT_FMT

    last_row2 = first_data_row + len(months) - 1
    return dict(main_header_row=2, main_first=3, main_last=last_row, obs_col=2, layer_cols_main={L: 3 + i for i, L in enumerate(LAYERS)},
                sec2_header_row=sec2_row, sec2_first=first_data_row, sec2_last=last_row2,
                layer_blocks=layer_blocks, obs_col2=obs_col2)


def make_fan_chart(ws_data, title, color_light, color_dark, ink, refs, layer_key):
    base = refs["layer_blocks"][layer_key]
    hdr = refs["sec2_header_row"]
    first, last = refs["sec2_first"], refs["sec2_last"]

    area = AreaChart()
    area.grouping = "stacked"
    area.overlap = 100
    area.title = title
    area.style = None
    area.y_axis.numFmt = "0%"
    area.y_axis.title = None
    area.x_axis.title = None
    area.height = 9.5
    area.width = 15.5
    area.gapWidth = 0

    cats = Reference(ws_data, min_col=1, min_row=first, max_row=last)

    # base invisible = p05 (sin titulo, se oculta de la leyenda mas abajo)
    p05_ref = Reference(ws_data, min_col=base, min_row=hdr, max_row=last)
    area.add_data(p05_ref, titles_from_data=False)
    # seg 25-05 y seg 75-25 y seg 95-75 (3 franjas visibles, nombres limpios)
    seg_ref = Reference(ws_data, min_col=base + 1, max_col=base + 3, min_row=hdr, max_row=last)
    area.add_data(seg_ref, titles_from_data=False)
    area.set_categories(cats)

    s0 = area.series[0]
    s0.graphicalProperties.noFill = True
    s0.graphicalProperties.ln = LineProperties(noFill=True)

    band_fills = [color_light, color_dark, color_light]
    band_names = ["P5–P95", "P25–P75 (RIC)", "P5–P95"]
    for k in range(1, 4):
        s = area.series[k]
        s.graphicalProperties.solidFill = band_fills[k - 1]
        s.graphicalProperties.ln = LineProperties(noFill=True)
        s.tx = SeriesLabel(v=band_names[k - 1])

    # mediana + observado como lineas superpuestas
    line = LineChart()
    med_ref = Reference(ws_data, min_col=base + 4, min_row=hdr, max_row=last)
    line.add_data(med_ref, titles_from_data=False)
    line.series[-1].tx = SeriesLabel(v="Mediana del modelo")
    obs_ref = Reference(ws_data, min_col=refs["obs_col2"], min_row=hdr, max_row=last)
    line.add_data(obs_ref, titles_from_data=False)
    line.series[-1].tx = SeriesLabel(v="IGAE observado")
    line.set_categories(cats)

    med_series = line.series[0]
    med_series.smooth = False
    med_series.graphicalProperties.line.solidFill = color_dark
    med_series.graphicalProperties.line.width = 26000
    med_series.marker = Marker(symbol="circle", size=6)
    med_series.marker.graphicalProperties.solidFill = color_dark
    med_series.marker.graphicalProperties.ln.solidFill = "FFFFFF"

    obs_series = line.series[1]
    obs_series.smooth = False
    obs_series.graphicalProperties.line.solidFill = ink
    obs_series.graphicalProperties.line.width = 26000
    obs_series.marker = Marker(symbol="circle", size=6)
    obs_series.marker.graphicalProperties.solidFill = ink
    obs_series.marker.graphicalProperties.ln.solidFill = "FFFFFF"

    area += line
    area.y_axis.majorGridlines.spPr = None
    # ocultar de la leyenda: la serie base invisible (0) y la franja P75-P95
    # duplicada (2, mismo color/nombre que la franja P5-P25 en el indice 1)
    area.legend.legendEntry = [LegendEntry(idx=0, delete=True), LegendEntry(idx=3, delete=True)]
    return area


def make_comparison_chart(ws_data, refs):
    lc = LineChart()
    lc.title = "Mediana por capa vs. IGAE observado (sin intervalos)"
    lc.y_axis.numFmt = "0%"
    lc.height = 10
    lc.width = 24

    hdr, first, last = refs["main_header_row"], refs["main_first"], refs["main_last"]
    cats = Reference(ws_data, min_col=1, min_row=first, max_row=last)
    for L in LAYERS:
        ref = Reference(ws_data, min_col=refs["layer_cols_main"][L], min_row=hdr, max_row=last)
        lc.add_data(ref, titles_from_data=True)
    obs_ref = Reference(ws_data, min_col=refs["obs_col"], min_row=hdr, max_row=last)
    lc.add_data(obs_ref, titles_from_data=True)
    lc.set_categories(cats)

    colors = [LAYER_COLOR[L] for L in LAYERS] + [INK]
    widths = [22000, 22000, 22000, 26000, 30000]
    for i, s in enumerate(lc.series):
        s.smooth = False
        s.graphicalProperties.line.solidFill = colors[i]
        s.graphicalProperties.line.width = widths[i]
        s.marker = Marker(symbol="circle", size=6)
        s.marker.graphicalProperties.solidFill = colors[i]
        s.marker.graphicalProperties.ln.solidFill = "FFFFFF"
    lc.series[-1].graphicalProperties.line.dashStyle = "sysDot"
    return lc


def build():
    months, observed, summaries = load_data()

    wb = openpyxl.Workbook()
    ws_data = wb.active
    ws_data.title = "Datos"
    refs = build_datos_sheet(ws_data, months, observed, summaries)

    ws_graf = wb.create_sheet("Gráficas")
    ws_graf["A1"] = "IGAE México 2020 — Escalera de pronóstico M0→M3"
    ws_graf["A1"].font = TITLE_FONT
    ws_graf["A2"] = "Corte de información: 15-jun-2020 · Horizonte: abril–diciembre 2020"
    ws_graf["A2"].font = Font(color="595959", size=10)

    comp = make_comparison_chart(ws_data, refs)
    ws_graf.add_chart(comp, "A4")

    anchors = ["A24", "K24", "A43", "K43"]
    for i, L in enumerate(LAYERS):
        chart = make_fan_chart(
            ws_data, LAYER_LABELS[L], BAND_LIGHT[L], LAYER_COLOR[L], INK, refs, L
        )
        ws_graf.add_chart(chart, anchors[i])

    ws_graf["A62"] = ("Banda: P5–P95 (claro) y P25–P75 (oscuro, rango intercuartílico). "
                       "Línea de color = mediana del modelo. Línea negra punteada = IGAE m/m observado.")
    ws_graf["A62"].font = NOTE_FONT

    autosize(ws_graf, [12] * 20)

    out_path = OUT / "IGAE_escalera_2020.xlsx"
    wb.save(out_path)
    print(f"Guardado: {out_path}")
    return out_path


if __name__ == "__main__":
    build()
