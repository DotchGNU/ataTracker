"""Figure style and the one save function every plot goes through."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("atatracker")


def use_style(font_family: str = "Arial", font_files: list[str] | None = None) -> None:
    """Select the Agg backend, register extra font files and keep SVG text as text.

    Parameters
    ----------
    font_family : str
        Font family for every text element (falls back to the matplotlib default if not installed).
    font_files : list of str, optional
        Font files (``.ttf``) to register first, e.g. a system Arial.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager as fm

    for f in font_files or []:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    try:
        fm.findfont(font_family, fallback_to_default=False)
        family = [font_family]
    except ValueError:
        LOG.warning("font %s not found; using the matplotlib default", font_family)
        family = ["DejaVu Sans"]
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": family,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def minute_axis(ax, xmax_min: float, labelbottom: bool = True) -> None:
    """x axis in minutes, integer ticks from 0, outward ticks."""
    ax.set_xlim(0, xmax_min)
    ax.set_xticks(range(0, int(xmax_min) + 1))
    ax.tick_params(labelsize=9, length=3, direction="out", labelbottom=labelbottom)


def xmax_minutes(durations_s: list[float]) -> float:
    """Common x range (min) for a set of recordings, rounded up to 0.1 min."""
    return float(np.ceil(max(durations_s) / 60 * 10) / 10)


def save_figure(fig, out_dir: str | Path, name: str, source: pd.DataFrame, dpi: int = 300) -> list[Path]:
    """Save ``<name>.png`` (transparent, ``dpi``), ``<name>.svg`` and ``source_<name>.csv`` with the plotted values.

    Returns
    -------
    list of Path
        The three files.
    """
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png, svg, csv = out / f"{name}.png", out / f"{name}.svg", out / f"source_{name}.csv"
    fig.savefig(png, dpi=dpi, transparent=True)
    fig.savefig(svg, transparent=True)
    plt.close(fig)
    source.to_csv(csv, index=False, encoding="utf-8")
    return [png, svg, csv]
