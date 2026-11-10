"""Per-animal twitching plot: raw amplitudes (top) and display traces (bottom), markers at head onset and end line."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..phase import PhaseResult, display_traces
from ..signal import Signals
from .style import minute_axis, save_figure


def twitching_source(name: str, tag: str, sig: Signals, res: PhaseResult, cfg: Config) -> pd.DataFrame:
    """Plotted values of one twitching plot (one row per frame).

    Columns: ``animal, checkpoint_tag, t_s, chest_signal, head_signal, chest_display, head_display,
    head_onset_s, last_twitching_s``.
    """
    t = np.arange(len(sig.chest)) / sig.fps
    ok = not (np.isnan(res.head_on_s) or np.isnan(res.end_s))
    c1, h1 = display_traces(sig.chest, sig.head, t, res.head_on_s, res.end_s, cfg.display) if ok else (sig.chest, sig.head)
    return pd.DataFrame(
        dict(
            animal=name,
            checkpoint_tag=tag,
            t_s=t,
            chest_signal=sig.chest,
            head_signal=sig.head,
            chest_display=c1,
            head_display=h1,
            head_onset_s=res.head_on_s,
            last_twitching_s=res.end_s,
        )
    )


def plot_twitching(src: pd.DataFrame, out_dir: str | Path, name: str, cfg: Config, markers: str = "triangles") -> list[Path]:
    """Draw one twitching plot from :func:`twitching_source` output.

    Parameters
    ----------
    src : DataFrame
        Source table of one recording.
    out_dir : str or Path
        Output folder.
    name : str
        File stem.
    cfg : Config
        Colours, x range and dpi from ``cfg.figures``.
    markers : {"triangles", "lines"}
        Head onset / end line as triangles above the top axis, or dashed lines in both panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    f = cfg.figures
    t = src.t_s.values
    tm = t / 60
    hon, end = float(src.head_onset_s.iloc[0]), float(src.last_twitching_s.iloc[0])
    ok = not (np.isnan(hon) or np.isnan(end))
    dur_s = t[-1] if len(t) else 0.0
    xmax = max(f.xmax_min, np.ceil(dur_s / 60 * 5) / 5)
    fig, ax = plt.subplots(2, 1, figsize=(6.0, 3.3), sharex=True)
    for a, c, h in (
        (ax[0], src.chest_signal.values, src.head_signal.values),
        (ax[1], src.chest_display.values, src.head_display.values),
    ):
        a.vlines(tm, 0, c, color=f.chest_color, lw=0.45)
        a.vlines(tm, 0, h, color=f.head_color, lw=0.45)
        a.set_ylim(0, 1)
        a.set_yticks([])
        if ok and markers == "lines":
            for x in (hon, end):
                a.axvline(x / 60, color="k", ls="--", lw=0.9)
        for sp in a.spines.values():
            sp.set_linewidth(0.5)
    top = 0.985
    if ok and markers == "triangles":
        for x, col in ((hon / 60, f.onset_marker_color), (end / 60, f.end_marker_color)):
            ax[0].plot(
                [x], [1.07], marker="v", ms=6, color=col, mec=col, transform=ax[0].get_xaxis_transform(), clip_on=False, ls="none"
            )
        fig.legend(
            handles=[
                Line2D([], [], marker="v", ls="none", color=f.onset_marker_color, ms=5, label="chest to head transition"),
                Line2D([], [], marker="v", ls="none", color=f.end_marker_color, ms=5, label="last twitching"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.01, 1.0),
            ncol=2,
            fontsize=6.5,
            frameon=False,
            handletextpad=0.1,
            columnspacing=1.0,
        )
        top = 0.86
    minute_axis(ax[0], xmax, labelbottom=False)
    minute_axis(ax[1], xmax)
    ax[1].set_xlabel("Time (min)", fontsize=9, labelpad=1)
    if not ok:
        ax[0].text(
            0.995,
            0.93,
            "no head phase found",
            transform=ax[0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=f.end_marker_color,
        )
    fig.subplots_adjust(hspace=0.08, left=0.02, right=0.99, top=top, bottom=0.17)
    return save_figure(fig, out_dir, name, src, dpi=f.dpi)
