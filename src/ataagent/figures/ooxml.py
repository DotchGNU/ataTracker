"""Check and repair the child-element order of DrawingML chart parts in xlsx / pptx files.

OOXML schemas are ``xsd:sequence`` types, so child order is part of the specification. Excel
offers to repair a file whose chart elements are out of order, while other readers may render
it without complaint; rendering is therefore not a validity check, this module is.
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

SER = {
    "barChart": [
        "idx",
        "order",
        "tx",
        "spPr",
        "invertIfNegative",
        "pictureOptions",
        "dPt",
        "dLbls",
        "trendline",
        "errBars",
        "cat",
        "val",
        "shape",
        "extLst",
    ],
    "scatterChart": [
        "idx",
        "order",
        "tx",
        "spPr",
        "marker",
        "dPt",
        "dLbls",
        "trendline",
        "errBars",
        "xVal",
        "yVal",
        "smooth",
        "extLst",
    ],
    "lineChart": [
        "idx",
        "order",
        "tx",
        "spPr",
        "marker",
        "dPt",
        "dLbls",
        "trendline",
        "errBars",
        "cat",
        "val",
        "smooth",
        "extLst",
    ],
}
_AX = [
    "axId",
    "scaling",
    "delete",
    "axPos",
    "majorGridlines",
    "minorGridlines",
    "title",
    "numFmt",
    "majorTickMark",
    "minorTickMark",
    "tickLblPos",
    "spPr",
    "txPr",
    "crossAx",
    "crosses",
    "crossesAt",
]
ORDER = {
    "valAx": _AX + ["crossBetween", "majorUnit", "minorUnit", "dispUnits", "extLst"],
    "catAx": _AX + ["auto", "lblAlgn", "lblOffset", "tickLblSkip", "tickMarkSkip", "noMultiLvlLbl", "extLst"],
    "errBars": ["errDir", "errBarType", "errValType", "noEndCap", "plus", "minus", "val", "spPr", "extLst"],
    "marker": ["symbol", "size", "spPr", "extLst"],
    "dPt": ["idx", "invertIfNegative", "marker", "bubble3D", "explosion", "spPr", "pictureOptions", "extLst"],
    "dLbl": [
        "idx",
        "delete",
        "layout",
        "tx",
        "numFmt",
        "spPr",
        "txPr",
        "dLblPos",
        "showLegendKey",
        "showVal",
        "showCatName",
        "showSerName",
        "showPercent",
        "showBubbleSize",
        "separator",
        "extLst",
    ],
    "dLbls": [
        "dLbl",
        "delete",
        "numFmt",
        "spPr",
        "txPr",
        "dLblPos",
        "showLegendKey",
        "showVal",
        "showCatName",
        "showSerName",
        "showPercent",
        "showBubbleSize",
        "separator",
        "showLeaderLines",
        "leaderLines",
        "extLst",
    ],
    "scatterChart": ["scatterStyle", "varyColors", "ser", "dLbls", "axId", "extLst"],
    "barChart": ["barDir", "grouping", "varyColors", "ser", "dLbls", "gapWidth", "overlap", "serLines", "axId", "extLst"],
    "lineChart": [
        "grouping",
        "varyColors",
        "ser",
        "dLbls",
        "dropLines",
        "hiLowLines",
        "upDownBars",
        "marker",
        "smooth",
        "axId",
        "extLst",
    ],
    "chart": [
        "title",
        "autoTitleDeleted",
        "pivotFmts",
        "view3D",
        "floor",
        "sideWall",
        "backWall",
        "plotArea",
        "legend",
        "plotVisOnly",
        "dispBlanksAs",
        "showDLblsOverMax",
        "extLst",
    ],
    "chartSpace": [
        "date1904",
        "lang",
        "roundedCorners",
        "style",
        "clrMapOvr",
        "pivotSource",
        "protection",
        "chart",
        "spPr",
        "txPr",
        "externalData",
        "printSettings",
        "userShapes",
        "extLst",
    ],
    "numRef": ["f", "numCache", "extLst"],
    "strRef": ["f", "strCache", "extLst"],
    "xVal": ["multiLvlStrRef", "numRef", "numLit", "strRef", "strLit"],
    "yVal": ["numRef", "numLit"],
}
PLOTAREA_RANK = {"layout": 0, "dTable": 2, "spPr": 3, "extLst": 4}
_AXES = {"valAx", "catAx", "dateAx", "serAx"}
REPEATABLE = {
    "ser",
    "dPt",
    "dLbl",
    "trendline",
    "errBars",
    "ext",
    "axId",
    "valAx",
    "catAx",
    "dateAx",
    "serAx",
    "barChart",
    "lineChart",
    "scatterChart",
    "f",
}
REQUIRED_FIRST = {"scatterChart": "scatterStyle", "barChart": "barDir", "lineChart": "grouping"}
DEFAULTS = {"scatterStyle": "lineMarker"}


def _name(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _is_c(el: ET.Element) -> bool:
    return el.tag.startswith("{" + C + "}")


def _check_el(el: ET.Element, parent: str | None, part: str, problems: list[str]) -> None:
    tag = _name(el)
    names = [_name(k) for k in el if _is_c(k)]
    req = REQUIRED_FIRST.get(tag)
    if req and req not in names:
        problems.append(f"{part}: <{tag}> missing required first child <{req}>")
    if tag == "plotArea":
        last = -1.0
        for n in names:
            r = PLOTAREA_RANK.get(n, 1.5 if n in _AXES else 1)
            if r < last:
                problems.append(f"{part}: <plotArea> child <{n}> is out of place")
            last = max(last, r)
        return
    spec = SER.get(parent or "") if tag == "ser" else ORDER.get(tag)
    if not spec:
        return
    pos, seen = -1, {}
    for n in names:
        if n == "extLst":
            continue
        if n not in spec:
            problems.append(f"{part}: <{tag}> unexpected child <{n}>")
            continue
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1 and n not in REPEATABLE:
            problems.append(f"{part}: <{tag}> has {seen[n]}x <{n}> (max 1)")
        i = spec.index(n)
        if i < pos:
            problems.append(f"{part}: <{tag}> child <{n}> comes after <{spec[pos]}>")
        pos = max(pos, i)


def _walk(el: ET.Element, parent: str | None, part: str, problems: list[str]) -> None:
    if _is_c(el):
        _check_el(el, parent, part, problems)
    for k in el:
        _walk(k, _name(el), part, problems)


def check(path: str | Path, part_glob: str = "charts/chart") -> list[str]:
    """List child-order problems of every chart part of an xlsx / pptx file (empty list = valid order)."""
    problems: list[str] = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(x for x in z.namelist() if part_glob in x and x.endswith(".xml")):
            _walk(ET.fromstring(z.read(n)), None, n, problems)
    return problems


def _reorder(el: ET.Element, parent: str | None) -> None:
    tag = _name(el)
    for k in list(el):
        _reorder(k, tag)
    if not _is_c(el) or tag == "plotArea":
        return
    spec = SER.get(parent or "") if tag == "ser" else ORDER.get(tag)
    if not spec:
        return
    kids = list(el)
    req = REQUIRED_FIRST.get(tag)
    if req and not any(k.tag == f"{{{C}}}{req}" for k in kids):
        new = ET.Element(f"{{{C}}}{req}")
        if req in DEFAULTS:
            new.set("val", DEFAULTS[req])
        kids.insert(0, new)

    def key(k: ET.Element) -> int:
        if not _is_c(k):
            return len(spec) + 1
        n = _name(k)
        return spec.index(n) if n in spec else len(spec)

    kids = sorted(kids, key=key)
    for k in list(el):
        el.remove(k)
    for k in kids:
        el.append(k)


def repair(path: str | Path, part_glob: str = "charts/chart") -> list[str]:
    """Rewrite the file with every chart part's children in schema order; returns the rewritten part names."""
    for pre, uri in (("c", C), ("a", A_NS), ("r", R_NS)):
        ET.register_namespace(pre, uri)
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    import os

    os.close(fd)
    changed = []
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if part_glob in item.filename and item.filename.endswith(".xml"):
                root = ET.fromstring(data)
                _reorder(root, None)
                data = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(root, encoding="utf-8")
                changed.append(item.filename)
            zout.writestr(item, data)
    shutil.move(tmp, path)
    return changed
