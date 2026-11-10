"""Phase-duration bar plot (mean +- SEM per group, one dot per animal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .style import save_figure


def group_order(table: pd.DataFrame, cfg: Config) -> list[str]:
    """Groups in ``figures.group_order`` first (if present), then the rest in order of appearance."""
    present = [g for g in pd.unique(table.group) if g]
    return [g for g in cfg.figures.group_order if g in present] + [g for g in present if g not in cfg.figures.group_order]


def plot_phase_bars(table: pd.DataFrame, out_dir: str | Path, name: str, cfg: Config) -> list[Path]:
    """Chest-phase and head-phase durations per group.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`ataagent.pipeline.run_phase` (columns ``group, animal, chest_phase_min, head_phase_min``).
    out_dir, name : str or Path, str
        Output folder and file stem.
    cfg : Config
        Colours, jitter seed and dpi.

    Returns
    -------
    list of Path
        PNG, SVG and ``source_<name>.csv`` (one row per plotted dot, with the bar mean, SEM and n of its group).
    """
    import matplotlib.pyplot as plt

    groups = group_order(table, cfg)
    rng = np.random.default_rng(cfg.figures.jitter_seed)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    src = []
    for ax, col, title, colour in zip(
        axs,
        ("chest_phase_min", "head_phase_min"),
        ("Chest phase (min)", "Head phase (min)"),
        (cfg.figures.chest_color, cfg.figures.head_color),
        strict=True,
    ):
        for i, g in enumerate(groups):
            sub = table[table.group == g].dropna(subset=[col])
            v = sub[col].values
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            ax.bar(i, v.mean() if len(v) else np.nan, yerr=sem, color=colour, alpha=0.45, capsize=2)
            xj = rng.normal(i, 0.08, len(v))
            ax.plot(xj, v, "k.", ms=3)
            src.append(
                pd.DataFrame(
                    dict(
                        panel=col,
                        group=g,
                        n=len(v),
                        mean=v.mean() if len(v) else np.nan,
                        sem=sem,
                        animal=sub.animal.values,
                        x=xj,
                        y=v,
                    )
                )
            )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={int((table.group == g).sum())})" for g in groups], fontsize=6, rotation=25)
        ax.set_ylabel(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, out_dir, name, pd.concat(src, ignore_index=True), dpi=cfg.figures.dpi)
