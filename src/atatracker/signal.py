"""Per-frame motion signal, normalisation and peak calling.

A twitch is detected from a one-dimensional step signal of one tracked point. The signal is
masked where the tracker is unsure, scaled to 0-1 by a high percentile, and peaks above a
threshold are counted in 1-s bins (docs/method.md, section 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.signal as ss

from .config import MOTION_MODES, SignalConfig


def step(x: np.ndarray | None, y: np.ndarray, mode: str = "y") -> np.ndarray:
    """Frame-to-frame step signal of one tracked point.

    Parameters
    ----------
    x, y : ndarray
        Point coordinates per frame (px). ``x`` may be ``None`` for ``mode="y"``.
    mode : {"y", "x", "xy", "max", "pca"}
        ``y``: |dy|; ``x``: |dx|; ``xy``: sqrt(dx^2 + dy^2); ``max``: max(|dx|, |dy|);
        ``pca``: |projection of (dx, dy) on the main motion axis of this recording| (SVD of the steps).

    Returns
    -------
    ndarray
        Step per frame (px); the first frame is 0.
    """
    if mode not in MOTION_MODES:
        raise ValueError(f"signal mode must be one of {MOTION_MODES}, got {mode!r}")
    dy = np.diff(y, prepend=y[0]).astype(float)
    if mode == "y":
        return np.abs(dy)
    dx = np.diff(x, prepend=x[0]).astype(float)
    if mode == "x":
        return np.abs(dx)
    if mode == "xy":
        return np.hypot(dx, dy)
    if mode == "max":
        return np.maximum(np.abs(dx), np.abs(dy))
    ok = np.isfinite(dx) & np.isfinite(dy)
    m = np.c_[dx[ok], dy[ok]]
    ax = np.linalg.svd(m, full_matrices=False)[2][0] if len(m) > 2 else np.array([0.0, 1.0])
    return np.abs(dx * ax[0] + dy * ax[1])


def amplitude(
    x: np.ndarray, y: np.ndarray, likelihood: np.ndarray, mode: str, likelihood_min: float, norm_percentile: float
) -> np.ndarray:
    """Normalised motion amplitude (0-1) of one point.

    Frames whose likelihood (or the previous frame's likelihood) is below ``likelihood_min`` are
    masked; the step is scaled by ``(d - min) / (percentile - min)``, clipped at 1, masked frames
    set to 0.

    Parameters
    ----------
    x, y : ndarray
        Coordinates per frame (px).
    likelihood : ndarray
        DeepLabCut likelihood per frame (0-1).
    mode : str
        Step signal mode (see :func:`step`).
    likelihood_min : float
        Likelihood mask threshold (0 disables the mask).
    norm_percentile : float
        Percentile of the step distribution mapped to 1.

    Returns
    -------
    ndarray
        Amplitude per frame, dimensionless, in [0, 1].
    """
    d = step(x, y, mode)
    bad = likelihood < likelihood_min
    bad = bad | np.roll(bad, 1)
    d[bad] = np.nan
    lo, hi = np.nanmin(d), np.nanpercentile(d, norm_percentile)
    v = (d - lo) / (hi - lo if hi > lo else 1.0)
    return np.nan_to_num(np.minimum(v, 1.0), nan=0.0)


def peak_times(a: np.ndarray, threshold: float, distance_frames: int, fps: float) -> np.ndarray:
    """Peak times (s) of an amplitude trace (``scipy.signal.find_peaks`` with height and distance)."""
    p, _ = ss.find_peaks(a, height=threshold, distance=distance_frames)
    return p / fps


def bin_counts(times_s: np.ndarray, duration_s: float, bin_s: float = 1.0) -> np.ndarray:
    """Number of peaks per bin; bins ``[0, bin_s), [bin_s, 2 bin_s), ...`` up to ``ceil(duration / bin_s) * bin_s``."""
    return np.histogram(times_s, np.arange(0, np.ceil(duration_s / bin_s) * bin_s + bin_s, bin_s))[0].astype(float)


@dataclass
class Signals:
    """Motion signals and peak counts of one recording.

    Attributes
    ----------
    chest, head : ndarray
        Normalised amplitude per frame (0-1).
    chest_peaks_s, head_peaks_s : ndarray
        Peak times (s).
    duration_s : float
        Recording length (frames / fps, s).
    chest_counts, head_counts : ndarray
        Peaks per bin.
    fps : float
        Frame rate (frames/s).
    """

    chest: np.ndarray
    head: np.ndarray
    chest_peaks_s: np.ndarray
    head_peaks_s: np.ndarray
    duration_s: float
    chest_counts: np.ndarray
    head_counts: np.ndarray
    fps: float

    @property
    def counts(self) -> np.ndarray:
        """``(T, 2)`` array of chest and head counts per bin (HMM observations)."""
        return np.c_[self.chest_counts, self.head_counts]


def compute_signals(coords: pd.DataFrame, cfg: SignalConfig, bin_s: float = 1.0) -> Signals:
    """Amplitudes, peaks and per-bin counts of the chest and head points.

    Parameters
    ----------
    coords : DataFrame
        Columns ``cx, cy, cl, hx, hy, hl`` (see :func:`atatracker.common.load_coords`).
    cfg : SignalConfig
        Signal settings.
    bin_s : float
        Count bin width (s).
    """
    fps = cfg.frame_rate
    c = amplitude(coords.cx.values, coords.cy.values, coords.cl.values, cfg.chest_mode, cfg.likelihood_min, cfg.norm_percentile)
    h = amplitude(coords.hx.values, coords.hy.values, coords.hl.values, cfg.head_mode, cfg.likelihood_min, cfg.norm_percentile)
    cs = peak_times(c, cfg.peak_threshold, cfg.peak_distance_chest_frames, fps)
    hs = peak_times(h, cfg.peak_threshold, cfg.peak_distance_head_frames, fps)
    dur = len(coords) / fps
    return Signals(c, h, cs, hs, dur, bin_counts(cs, dur, bin_s), bin_counts(hs, dur, bin_s), fps)
