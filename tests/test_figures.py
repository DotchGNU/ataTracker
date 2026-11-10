import zipfile

import numpy as np
import pandas as pd
import pytest

from atatracker.config import config_from_dict
from atatracker.figures.stack import binmax, thin


def test_thin_keeps_run_ends():
    x = np.arange(8.0)
    v = np.array([0, 1, 1, 1, 1, 2, 2, 0.0])
    xt, vt = thin(x, v)
    assert xt.tolist() == [0, 1, 4, 5, 6, 7]
    assert vt.tolist() == [0, 1, 1, 2, 2, 0]


def test_binmax_preserves_per_bin_max():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 6.0, 5000))
    v = rng.uniform(0, 1, 5000)
    xb, vb = binmax(x, v, n_bins=60, xmax=6.0)
    edges = np.linspace(0, 6.0, 61)
    for k in range(60):
        m = (x >= edges[k]) & (x < edges[k + 1])
        if m.any():
            c = (edges[k] + edges[k + 1]) / 2
            j = np.where(np.isclose(xb, c))[0]
            if len(j):  # interior points of equal runs are dropped by thin(); the values are the same
                assert vb[j[0]] == v[m].max()
    assert vb.max() == v.max()


def test_binmax_zero_bins_is_full_resolution():
    x = np.linspace(0, 1, 11)
    v = np.linspace(0, 1, 11)
    xb, vb = binmax(x, v, 0, 1.0)
    assert np.array_equal(xb, x) and np.array_equal(vb, v)


def _table():
    return pd.DataFrame(
        dict(
            animal=["a1", "a2", "a3", "a4"],
            group=["group A", "group A", "group B", "group B"],
            chest_phase_min=[1.0, 1.2, 0.8, 0.9],
            head_phase_min=[2.0, 2.2, 1.5, 1.7],
            head_on_min=[1.0, 1.2, 0.8, 0.9],
            end_min=[3.0, 3.4, 2.3, 2.6],
            checkpoint_tag="m-1k",
        )
    )


def test_bar_plot_writes_png_svg_source(tmp_path):
    from atatracker.figures.bars import plot_phase_bars
    from atatracker.figures.style import use_style

    use_style()
    files = plot_phase_bars(_table(), tmp_path, "bars", config_from_dict({}))
    assert [f.name for f in files] == ["bars.png", "bars.svg", "source_bars.csv"]
    src = pd.read_csv(files[2])
    assert set(src.columns) >= {"panel", "group", "n", "mean", "sem", "animal", "x", "y"}
    assert len(src) == 8


def test_workbook_is_valid(tmp_path):
    pytest.importorskip("openpyxl")
    from atatracker.figures import ooxml
    from atatracker.figures.xlsx import build_workbook, theme_xml

    p = build_workbook(_table(), tmp_path / "w.xlsx", config_from_dict({}))
    assert ooxml.check(p) == []
    with zipfile.ZipFile(p) as z:
        assert z.read("xl/theme/theme1.xml") == theme_xml()
        chart = z.read("xl/charts/chart1.xml").decode()
    assert "group A" in chart and "group B" in chart
