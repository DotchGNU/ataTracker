"""Stacked twitching plot: one thin row per animal, grouped, chest and head traces filled from 0.

For a compact SVG every trace can be reduced to the maximum of each of ``stack_svg_bins`` x bins
(:func:`binmax`); for traces filled from 0 this keeps the drawn envelope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .style import save_figure


def thin(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop points inside runs of identical values, keeping the run ends (lossless for a step trace)."""
    k = np.ones(len(v), bool)
    k[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    return x[k], v[k]


def binmax(x: np.ndarray, v: np.ndarray, n_bins: int, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Maximum of ``v`` in each of ``n_bins`` equal x bins over ``[0, xmax]``, at the bin centres.

    ``n_bins <= 0`` returns the full-resolution trace (only :func:`thin` applied). Points with
    ``x > xmax`` are dropped; empty bins are skipped.
    """
    if n_bins <= 0:
        return thin(x, v)
    e = np.linspace(0, xmax, n_bins + 1)
    i = np.clip(np.digitize(x, e) - 1, 0, n_bins - 1)
    m = np.full(n_bins, np.nan)
    keep = x <= xmax
    np.fmax.at(m, i[keep], v[keep])
    c = (e[:-1] + e[1:]) / 2
    ok = ~np.isnan(m)
    return thin(c[ok], m[ok])


def plot_stack(table: pd.DataFrame, plot_dir: str | Path, out_dir: str | Path, name: str, cfg: Config) -> list[Path]:
    """Stacked twitching plot of every animal in ``table``, grouped by ``table.group``.

    Parameters
    ----------
    table : DataFrame
        ``animal, group`` (and optionally ``label``) rows in display order; animals without a group are skipped.
    plot_dir : str or Path
        Folder with the twitching source tables ``source_<animal>.csv`` (``ata phase`` output ``twitching/``).
    out_dir, name : str or Path, str
        Output folder and file stem.
    cfg : Config
        ``figures.stack_*`` geometry, colours, ``stack_svg_bins`` and ``stack_head_gate``.
    """
    import matplotlib.pyplot as plt

    from .bars import group_order

    f = cfg.figures
    groups = group_order(table, cfg)
    n_rows = [int((table.group == g).sum()) for g in groups]
    fig_w = f.stack_label_w_in + f.stack_plot_w_in + 0.25
    fig_h = sum(n * f.stack_row_h_in for n in n_rows) + f.stack_group_gap_in * len(groups) + 0.25
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0)

    def ax_in(x: float, ytop: float, w: float, h: float):
        return fig.add_axes([x / fig_w, 1 - (ytop + h) / fig_h, w / fig_w, h / fig_h])

    src, y = [], 0.12
    for g, n in zip(groups, n_rows, strict=True):
        sub = table[table.group == g].reset_index(drop=True)
        y += f.stack_group_gap_in * 0.45
        fig.text(0.02 / fig_w, 1 - (y - 0.07) / fig_h, f"{g} (n={n})", fontsize=9, fontweight="bold", va="bottom", ha="left")
        for i, r in sub.iterrows():
            d = pd.read_csv(Path(plot_dir) / f"source_{r.animal}.csv")
            tm = d.t_s.values / 60
            if f.stack_head_gate:
                hv = d.head_display.values
            else:
                hv = np.where(d.t_s.values > d.last_twitching_s.values, cfg.display.post_head_level, d.head_signal.values)
            xc, vc = binmax(tm, d.chest_display.values, f.stack_svg_bins, f.stack_xmax_min)
            xh, vh = binmax(tm, hv, f.stack_svg_bins, f.stack_xmax_min)
            ax = ax_in(f.stack_label_w_in, y + i * f.stack_row_h_in, f.stack_plot_w_in, f.stack_row_h_in)
            ax.fill_between(xc, 0, vc, step="mid", facecolor=f.chest_color, edgecolor=f.chest_color, lw=0.35)
            ax.fill_between(xh, 0, vh, step="mid", facecolor=f.head_color, edgecolor=f.head_color, lw=0.35)
            ax.set_xlim(0, f.stack_xmax_min)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(range(0, int(f.stack_xmax_min) + 1))
            ax.patch.set_alpha(0)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            last = i == n - 1
            ax.tick_params(
                axis="x", length=2.5 if last else 0, width=0.5, labelsize=8, labelbottom=last, direction="out", pad=1.5
            )
            if last:
                ax.set_xlabel("Time (min)", fontsize=8, labelpad=1)
            label = r.get("label", r.animal) if hasattr(r, "get") else r.animal
            fig.text(
                (f.stack_label_w_in - 0.06) / fig_w,
                1 - (y + (i + 0.5) * f.stack_row_h_in) / fig_h,
                str(label),
                fontsize=8,
                ha="right",
                va="center",
            )
            for trace, xs, vs in (("chest", xc, vc), ("head", xh, vh)):
                src.append(pd.DataFrame(dict(group=g, row=i + 1, animal=r.animal, trace=trace, t_min=xs, value=vs)))
        y += n * f.stack_row_h_in + f.stack_group_gap_in * 0.55
    return save_figure(fig, out_dir, name, pd.concat(src, ignore_index=True), dpi=f.dpi)
