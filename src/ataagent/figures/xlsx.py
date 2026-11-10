"""Phase-duration workbook with a native Excel chart (requires the ``figures`` extra: openpyxl).

One xy-scatter chart object holds, per group, the chest and head phases as thick horizontal bar
segments (mean, cumulative), SEM error bars, one jittered dot per animal at each segment end and
the group names as data labels. Every number on the chart sheet is an Excel formula over the
``data_raw`` sheet, so editing ``data_raw`` updates the chart. The workbook uses the Office
2013-2022 theme, Arial text, 0.25 pt axis lines and a transparent plot area.
"""

from __future__ import annotations

import logging
import re
import shutil
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import ooxml

LOG = logging.getLogger("ataagent")
A_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
PT_PER_EMU = 12700
RAW, CHT = "data_raw", "chart"
SEG_LABELS = ["chest phase", "head phase"]
SEG_COLS = ["chest_phase_min", "head_phase_min"]
CUM_COLS = ["head_on_min", "end_min"]
RAW_COLS = [
    "label",
    "animal",
    "group",
    "chest_phase_min",
    "head_phase_min",
    "head_on_min",
    "end_min",
    "checkpoint_tag",
    "y_jitter",
]
FONT, FONT_PT = "Arial", 9
X_MIN, X_UNIT = 0, 1
BAR_THICK_PT, ERR_COLOR, ERR_LINE_EMU, AXIS_LINE_EMU = 17, "404040", 9525, 3175
MARKER_SIZE, MARKER_LINE_EMU = 5, 9525
PLOT_INNER = (0.21, 0.17, 0.76, 0.74)  # x, y, w, h of the inner plot area (fractions of the chart frame)
CHART_H_CM = 4.0568
CHART_ANCHOR = "F4"


def theme_xml() -> bytes:
    """The Office 2013-2022 theme shipped as package data."""
    return resources.files("ataagent").joinpath("data/theme_office2013-2022.xml").read_bytes()


def _txt_props():
    from openpyxl.chart.text import RichText
    from openpyxl.drawing.text import CharacterProperties, Paragraph, ParagraphProperties
    from openpyxl.drawing.text import Font as DrawFont

    cp = CharacterProperties(latin=DrawFont(typeface=FONT), sz=int(FONT_PT * 100), b=False)
    return RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])


def build_workbook(table: pd.DataFrame, path: str | Path, cfg: Config, xmax_min: float | None = None) -> Path:
    """Write the phase-duration workbook.

    Parameters
    ----------
    table : DataFrame
        Per-animal phase table (``ata phase`` output) with a non-empty ``group``; an optional ``label`` column names the dots.
    path : str or Path
        Output ``.xlsx``.
    cfg : Config
        ``figures.group_order``, ``xlsx_bar_colors``, ``xlsx_plot_w_in``, ``xlsx_jitter_span``, ``jitter_seed``.
    xmax_min : float, optional
        x axis maximum (min); defaults to ``figures.stack_xmax_min`` so the chart lines up with the stacked plot.

    Returns
    -------
    Path
        The workbook.
    """
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series
    from openpyxl.chart.data_source import NumDataSource, NumRef
    from openpyxl.chart.error_bar import ErrorBars
    from openpyxl.chart.layout import Layout, ManualLayout
    from openpyxl.chart.marker import Marker
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter as col

    from .bars import group_order

    f = cfg.figures
    xmax = f.stack_xmax_min if xmax_min is None else xmax_min
    groups = group_order(table, cfg)
    t = table[table.group.isin(groups)].copy()
    t["_o"] = t.group.map({g: i for i, g in enumerate(groups)})
    t = t.sort_values(["_o", "animal"]).drop(columns="_o").reset_index(drop=True)
    if "label" not in t:
        t["label"] = t.animal
    t["y_jitter"] = np.random.default_rng(f.jitter_seed).uniform(0, 1, len(t))
    raw_rows = t[RAW_COLS].values.tolist()
    ns, ng = len(SEG_LABELS), len(groups)
    wb = Workbook()
    ws = wb.active
    ws.title = RAW
    ws.append(RAW_COLS)
    for r in raw_rows:
        ws.append(r)
    for c in range(1, len(RAW_COLS) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)
    gi = RAW_COLS.index("group")
    bounds = []
    for g in groups:
        rows = [i for i, r in enumerate(raw_rows) if r[gi] == g]
        bounds.append((rows[0] + 2, rows[-1] + 2))
    c_seg = [col(RAW_COLS.index(n) + 1) for n in SEG_COLS]
    c_cum = [col(RAW_COLS.index(n) + 1) for n in CUM_COLS]
    c_jit = col(RAW_COLS.index("y_jitter") + 1)

    wc = wb.create_sheet(CHT)

    def put(r: int, c: int, v: object, bold: bool = False) -> None:
        cell = wc.cell(row=r, column=c, value=v)
        if bold:
            cell.font = Font(bold=True)

    put(1, 1, "Phase durations: mean +- SEM with one dot per animal", True)
    put(2, 1, "All numbers below are Excel formulas over data_raw (minutes). Edit data_raw and the chart follows.")
    h = 4
    put(h, 1, "group", True)
    put(h, 2, "n", True)
    for j, lab in enumerate(SEG_LABELS):
        put(h, 3 + j, f"{lab} mean (min)", True)
        put(h, 3 + ns + j, f"{lab} SEM (min)", True)
    for i, (r0, r1) in enumerate(bounds):
        r = h + 1 + i
        put(r, 1, groups[i])
        put(r, 2, f"=COUNT({RAW}!${c_seg[0]}${r0}:${c_seg[0]}${r1})")
        for j in range(ns):
            rng = f"{RAW}!${c_seg[j]}${r0}:${c_seg[j]}${r1}"
            put(r, 3 + j, f"=AVERAGE({rng})")
            put(r, 3 + ns + j, f"=STDEV({rng})/SQRT(COUNT({rng}))")
    s0, s1 = h + 1, h + ng
    ypos = [ng - i for i in range(ng)]
    b = s1 + 2
    put(b, 1, "cumulative phase boundaries (min) and the y position of each group band", True)
    b += 1
    put(b, 1, "group", True)
    for j in range(ns):
        put(b, 2 + j, f"end of {SEG_LABELS[j]} (min)", True)
    put(b, 2 + ns, "y (category units)", True)
    for i in range(ng):
        r, s = b + 1 + i, s0 + i
        put(r, 1, f"=$A${s}")
        for j in range(ns):
            put(r, 2 + j, "=" + "+".join(f"${col(3 + k)}${s}" for k in range(j + 1)))
        put(r, 2 + ns, ypos[i])
    b0, b1 = b + 1, b + ng
    yc = col(2 + ns)
    g = b1 + 2
    put(g, 1, "bar segments: each phase of each group as a 2-point thick line", True)
    g += 1
    put(g, 1, "series", True)
    put(g, 2, "x (min)", True)
    put(g, 3, "y (category units)", True)
    seg_rows, r = {}, g + 1
    for i in range(ng):
        for j in range(ns):
            start = "0" if j == 0 else f"${col(1 + j)}${b0 + i}"
            put(r, 1, f"{groups[i]} / {SEG_LABELS[j]}")
            put(r, 2, f"={start}")
            put(r, 3, f"=${yc}${b0 + i}")
            put(r + 1, 2, f"=${col(2 + j)}${b0 + i}")
            put(r + 1, 3, f"=${yc}${b0 + i}")
            seg_rows[(i, j)] = (r, r + 1)
            r += 2
    lab = r + 1
    put(lab, 1, "group label anchors (invisible series, data labels only)", True)
    lab += 1
    for i in range(ng):
        put(lab + 1 + i, 1, groups[i])
        put(lab + 1 + i, 2, 0)
        put(lab + 1 + i, 3, ypos[i])
    l0, l1 = lab + 1, lab + ng
    p = l1 + 2
    put(p, 1, "per-animal points (one dot per animal at each segment end)", True)
    p += 1
    put(p, 1, "label", True)
    for j in range(ns):
        put(p, 2 + j, f"{SEG_LABELS[j]} end x (min)", True)
    put(p, 2 + ns, "y (category units)", True)
    for i in range(len(raw_rows)):
        src, dst = i + 2, p + 1 + i
        put(dst, 1, f"={RAW}!$A${src}")
        for j in range(ns):
            put(dst, 2 + j, f"={RAW}!${c_cum[j]}${src}")
        pos = next(ypos[k] for k, (r0, r1) in enumerate(bounds) if r0 <= src <= r1)
        put(dst, 2 + ns, f"={pos}+({RAW}!${c_jit}${src}-0.5)*{f.xlsx_jitter_span}")
    p0, p1 = p + 1, p + len(raw_rows)
    for c, w in zip("ABCDE", (30, 10, 24, 24, 20), strict=True):
        wc.column_dimensions[c].width = w

    def ser(xcol: int, ycol: int, r0: int, r1: int, title: str) -> Series:
        return Series(
            Reference(wc, min_col=ycol, min_row=r0, max_row=r1), Reference(wc, min_col=xcol, min_row=r0, max_row=r1), title=title
        )

    sc = ScatterChart()
    sc.title = None
    sc.legend = None
    sc.graphical_properties = GraphicalProperties(noFill=True, ln=LineProperties(noFill=True))
    pal = f.xlsx_bar_colors
    for i in range(ng):
        for j in range(ns):
            s = ser(2, 3, *seg_rows[(i, j)], f"{groups[i]} {SEG_LABELS[j]}")
            s.marker = Marker(symbol="none")
            s.graphicalProperties = GraphicalProperties(
                ln=LineProperties(solidFill=pal[j], w=BAR_THICK_PT * PT_PER_EMU, cap="flat")
            )
            sc.series.append(s)
    for j in range(ns):
        s = ser(2 + j, 2 + ns, b0, b1, f"{SEG_LABELS[j]} SEM")
        s.marker = Marker(symbol="none")
        s.graphicalProperties = GraphicalProperties(ln=LineProperties(noFill=True))
        e = col(3 + ns + j)
        ref = f"{CHT}!${e}${s0}:${e}${s1}"
        s.errBars = ErrorBars(
            errDir="x",
            errBarType="both",
            errValType="cust",
            noEndCap=False,
            plus=NumDataSource(numRef=NumRef(f=ref)),
            minus=NumDataSource(numRef=NumRef(f=ref)),
            spPr=GraphicalProperties(ln=LineProperties(solidFill=ERR_COLOR, w=ERR_LINE_EMU)),
        )
        sc.series.append(s)
    for j in range(ns):
        s = ser(2 + j, 2 + ns, p0, p1, f"{SEG_LABELS[j]} points")
        s.marker = Marker(
            symbol="circle",
            size=MARKER_SIZE,
            spPr=GraphicalProperties(solidFill=pal[j], ln=LineProperties(solidFill="FFFFFF", w=MARKER_LINE_EMU)),
        )
        s.graphicalProperties = GraphicalProperties(ln=LineProperties(noFill=True))
        sc.series.append(s)
    s = ser(2, 3, l0, l1, "group labels")
    s.marker = Marker(symbol="none")
    s.graphicalProperties = GraphicalProperties(ln=LineProperties(noFill=True))
    sc.series.append(s)
    label_idx = len(sc.series) - 1
    sc.x_axis.scaling.min, sc.x_axis.scaling.max, sc.x_axis.majorUnit = X_MIN, xmax, X_UNIT
    sc.x_axis.delete = False
    sc.x_axis.majorTickMark, sc.x_axis.minorTickMark = "out", "none"
    sc.x_axis.majorGridlines = None
    sc.x_axis.txPr = _txt_props()
    sc.x_axis.title = None
    sc.y_axis.scaling.min, sc.y_axis.scaling.max = 0.5, ng + 0.5
    sc.y_axis.majorTickMark, sc.y_axis.minorTickMark = "out", "none"
    sc.y_axis.majorGridlines = None
    sc.y_axis.delete = False
    px, py, pw, ph = PLOT_INNER
    sc.layout = Layout(manualLayout=ManualLayout(layoutTarget="inner", xMode="edge", yMode="edge", x=px, y=py, w=pw, h=ph))
    sc.width = f.xlsx_plot_w_in * 2.54 / pw
    sc.height = CHART_H_CM
    wc.add_chart(sc, CHART_ANCHOR)
    out = Path(path)
    wb.save(out)
    _postprocess(out, label_idx, groups)
    LOG.info("wrote %s (%d animals, %d groups)", out.name, len(raw_rows), ng)
    return out


def _style_axes(x: str) -> str:
    """Draw both axis lines (0.25 pt), Arial x labels, hide the y-axis numbers (category coordinates)."""
    ln = f'<spPr {A_NS}><a:ln w="{AXIS_LINE_EMU}"/></spPr>'
    hidden = (
        f"<txPr {A_NS}><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr><a:noFill/></a:defRPr></a:pPr>"
        '<a:endParaRPr lang="en-US"/></a:p></txPr>'
    )
    xtxt = (
        f'<txPr {A_NS}><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="{FONT_PT * 100}" b="0">'
        f'<a:latin typeface="{FONT}"/></a:defRPr></a:pPr><a:endParaRPr lang="en-US"/></a:p></txPr>'
    )
    out, n = [], 0
    for blk in re.split(r"(<valAx>.*?</valAx>)", x, flags=re.S):
        if not blk.startswith("<valAx>"):
            out.append(blk)
            continue
        n += 1
        if "<numFmt" not in blk:
            blk = blk.replace("<majorTickMark", '<numFmt formatCode="General" sourceLinked="1"/><majorTickMark', 1)
        if "<tickLblPos" not in blk:
            blk = blk.replace("<crossAx", '<tickLblPos val="nextTo"/><crossAx', 1)
        blk = blk.replace("<crossAx", ln + "<crossAx", 1)
        if "<crosses " not in blk:
            cr = '<crosses val="max"/>' if n == 1 else '<crosses val="autoZero"/>'
            blk = re.sub(r"(<crossAx[^>]*/>)", r"\1" + cr + '<crossBetween val="midCat"/>', blk, count=1)
        blk = re.sub(r"<txPr[ >].*?</txPr>", "", blk, flags=re.S)
        blk = blk.replace("<crossAx", (xtxt if n == 1 else hidden) + "<crossAx", 1)
        out.append(blk)
    return "".join(out)


def _normalise(x: str) -> str:
    """Bring openpyxl's chart XML to the form Excel writes.

    chartSpace header, varyColors, smooth, bg1 marker lines, chart-level dLbls, print settings.
    """
    x = re.sub(
        r"(<chartSpace[^>]*>)", r'\1<date1904 val="0"/><lang val="en-US"/><roundedCorners val="1"/><style val="2"/>', x, count=1
    )
    if "<scatterStyle" not in x:
        x = x.replace("<scatterChart>", '<scatterChart><scatterStyle val="lineMarker"/>', 1)
    x = re.sub(r"(<scatterStyle[^>]*/>)", r'\1<varyColors val="1"/>', x, count=1)
    x = re.sub(r'(<marker><symbol val="none"\s*/>)<spPr>.*?</spPr>(</marker>)', r"\1\2", x, flags=re.S)
    x = re.sub(r"</yVal>", '</yVal><smooth val="1"/>', x)
    x = re.sub(r'<a:srgbClr val="FFFFFF"\s*/>', '<a:schemeClr val="bg1"/>', x)
    x = re.sub(r'<[wh]Mode val="factor"\s*/>', "", x)
    x = re.sub(
        r"</ser>(<axId)",
        r'</ser><dLbls><showLegendKey val="0"/><showVal val="0"/><showCatName val="0"/>'
        r'<showSerName val="0"/><showPercent val="0"/><showBubbleSize val="0"/></dLbls>\1',
        x,
        count=1,
    )
    x = re.sub(r"(<dispBlanksAs[^>]*/>)", r'\1<showDLblsOverMax val="1"/>', x, count=1)
    head, tail = x.rsplit("</chart>", 1)
    tail = re.sub(r'<a:prstDash val="solid"\s*/>', "", tail)
    tail = tail.replace(
        "</spPr>",
        '</spPr><printSettings><headerFooter/><pageMargins b="0.75" l="0.7" r="0.7" t="0.75" '
        'header="0.3" footer="0.3"/><pageSetup/></printSettings>',
        1,
    )
    return head + "</chart>" + tail


def _postprocess(path: Path, label_idx: int, labels: list[str]) -> None:
    """Group-name data labels, axis styling, transparent areas, theme and schema order (XML edits openpyxl cannot do)."""
    from xml.sax.saxutils import escape

    def dlbl(idx: int, text: str) -> str:
        return (
            f'<dLbl><idx val="{idx}"/><tx><rich {A_NS}><a:bodyPr/><a:lstStyle/><a:p><a:r>'
            f'<a:rPr lang="en-US" sz="{FONT_PT * 100}">'
            f'<a:latin typeface="{FONT}"/></a:rPr><a:t>{escape(text)}</a:t></a:r></a:p></rich></tx><dLblPos val="l"/>'
            '<showLegendKey val="0"/><showVal val="0"/><showCatName val="0"/><showSerName val="0"/><showPercent val="0"/>'
            '<showBubbleSize val="0"/></dLbl>'
        )

    dlbls = (
        "<dLbls>"
        + "".join(dlbl(i, t) for i, t in enumerate(labels))
        + f'<spPr {A_NS}><a:noFill/><a:ln><a:noFill/></a:ln><a:effectLst/></spPr><dLblPos val="l"/><showLegendKey val="0"/>'
        '<showVal val="1"/><showCatName val="0"/><showSerName val="0"/><showPercent val="0"/><showBubbleSize val="0"/>'
        '<showLeaderLines val="0"/></dLbls>'
    )
    nofill = f"<spPr {A_NS}><a:noFill/><a:ln><a:noFill/></a:ln></spPr>"
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/theme/theme1.xml":
                data = theme_xml()
            elif item.filename.startswith("xl/drawings/drawing"):
                d = data.decode("utf-8")
                m = re.search(r'<ext cx="(\d+)" cy="(\d+)"/>', d)
                if m and re.search(r"<xfrm\s*/>", d):  # an empty xfrm makes Excel drop the chart
                    ns = A_NS.split('"')[1]
                    d = re.sub(
                        r"<xfrm\s*/>",
                        f'<xfrm xmlns:a="{ns}"><a:off x="0" y="0"/><a:ext cx="{m.group(1)}" cy="{m.group(2)}"/></xfrm>',
                        d,
                        count=1,
                    )
                data = d.encode("utf-8")
            elif re.fullmatch(r"xl/charts/chart\d+\.xml", item.filename):
                x = data.decode("utf-8")
                x = re.sub(r'(<valAx>.*?)<axPos val="[^"]*"/>', r'\1<axPos val="t"/>', x, count=1, flags=re.S)
                x = _style_axes(x)
                sers = re.findall(r"<ser>.*?</ser>", x, re.S)
                tgt = sers[label_idx]
                x = x.replace(tgt, tgt.replace("<xVal>", dlbls + "<xVal>", 1), 1)
                if "<autoTitleDeleted" not in x:
                    x = x.replace("<plotArea>", '<autoTitleDeleted val="1"/><plotArea>', 1)
                x = x.replace("</plotArea>", nofill + "</plotArea>", 1)
                if "</chart><spPr" not in x:
                    x = x.replace("</chart>", "</chart>" + nofill, 1)
                data = _normalise(x).encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    if ooxml.check(path):
        ooxml.repair(path)
    left = ooxml.check(path)
    if left:
        raise RuntimeError(f"chart XML order problems remain: {left}")
