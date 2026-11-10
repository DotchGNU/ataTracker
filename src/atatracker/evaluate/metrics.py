"""Label-free tracking quality metrics and error against hand labels (docs/method.md, section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import QualityConfig


def body_length_z(coords: pd.DataFrame) -> np.ndarray:
    """Robust z score of the head-chest distance per frame: ``(d - median) / (1.4826 MAD)``."""
    hc = np.hypot(coords.hx - coords.cx, coords.hy - coords.cy).values
    med = np.median(hc)
    mad = 1.4826 * np.median(np.abs(hc - med))
    return (hc - med) / max(mad, 1e-6)


def quality_metrics(coords: pd.DataFrame, cfg: QualityConfig) -> dict[str, float]:
    """Per-recording tracking metrics that need no labels.

    Returns
    -------
    dict
        ``head_lik_lt05`` / ``chest_lik_lt05``: fraction of frames with likelihood < ``likelihood_low``;
        ``head_lik_lt09``: fraction with head likelihood < 0.9;
        ``bodylen_violation``: fraction of frames with |body-length z| > ``body_length_z``;
        ``head_jump_gt``: fraction of frames whose head step exceeds ``head_jump_px``;
        ``outlier_frac``: fraction of frames failing any of the three rules;
        ``head_step_p99`` / ``head_step_p999`` / ``chest_step_p99`` / ``chest_step_p999``: percentiles of the
        frame-to-frame step (px); ``hc_dist_median`` (px) and ``hc_dist_cv`` (MAD / median) of the head-chest distance.
    """
    hc = np.hypot(coords.hx - coords.cx, coords.hy - coords.cy).values
    med = np.median(hc)
    mad = 1.4826 * np.median(np.abs(hc - med))
    z = (hc - med) / max(mad, 1e-6)
    jump = np.hypot(coords.hx.diff(), coords.hy.diff()).fillna(0).values
    cstep = np.hypot(coords.cx.diff(), coords.cy.diff()).fillna(0).values
    low = cfg.likelihood_low
    return dict(
        head_lik_lt05=float((coords.hl < low).mean()),
        chest_lik_lt05=float((coords.cl < low).mean()),
        head_lik_lt09=float((coords.hl < 0.9).mean()),
        bodylen_violation=float((np.abs(z) > cfg.body_length_z).mean()),
        head_jump_gt=float((jump > cfg.head_jump_px).mean()),
        outlier_frac=float(((np.abs(z) > cfg.body_length_z) | (jump > cfg.head_jump_px) | (coords.hl.values < low)).mean()),
        head_step_p99=float(np.percentile(jump, 99)),
        head_step_p999=float(np.percentile(jump, 99.9)),
        chest_step_p99=float(np.percentile(cstep, 99)),
        chest_step_p999=float(np.percentile(cstep, 99.9)),
        hc_dist_median=float(med),
        hc_dist_cv=float(mad / max(med, 1e-6)),
    )


FLAG_METRICS = ("head_lik_lt05", "head_step_p999", "bodylen_violation")


def flag_against_reference(row: dict[str, float], reference_max: dict[str, float]) -> list[str]:
    """Names of the metrics in :data:`FLAG_METRICS` that exceed the maximum over the reference recordings."""
    return [f"{m} > reference max" for m in FLAG_METRICS if m in reference_max and row[m] > reference_max[m]]


def label_errors(coords: pd.DataFrame, frames: list[dict]) -> pd.DataFrame:
    """Distance (px) between tracked and hand-labelled points.

    Parameters
    ----------
    coords : DataFrame
        Tracked coordinates of one video (``cx, cy, cl, hx, hy, hl``).
    frames : list of dict
        Confirmed labels of that video (labeller export: ``frame``, ``chest`` [x, y], ``head`` [x, y]).

    Returns
    -------
    DataFrame
        ``frame, head_err, chest_err, head_along_body, head_lik``. ``head_along_body`` is the
        signed head error along the chest-to-head direction of the label (px; positive = beyond the label).
    """
    rows = []
    for f in frames:
        if f["frame"] >= len(coords):
            continue
        r = coords.iloc[f["frame"]]
        side = np.sign(f["head"][0] - f["chest"][0]) or 1
        rows.append(
            dict(
                frame=f["frame"],
                head_err=float(np.hypot(r.hx - f["head"][0], r.hy - f["head"][1])),
                chest_err=float(np.hypot(r.cx - f["chest"][0], r.cy - f["chest"][1])),
                head_along_body=float((r.hx - f["head"][0]) * side),
                head_lik=float(r.hl),
            )
        )
    return pd.DataFrame(rows, columns=["frame", "head_err", "chest_err", "head_along_body", "head_lik"])


def summarise_label_errors(err: pd.DataFrame, within_px: float = 10.0) -> dict[str, float]:
    """Median, 90th percentile and fraction within ``within_px`` of the head and chest errors (px)."""
    if err.empty:
        return dict(n=0)
    return dict(
        n=len(err),
        head_err_median=float(err.head_err.median()),
        head_err_p90=float(np.percentile(err.head_err, 90)),
        head_within=float((err.head_err <= within_px).mean()),
        chest_err_median=float(err.chest_err.median()),
        chest_within=float((err.chest_err <= within_px).mean()),
        head_bias_along_body=float(err.head_along_body.median()),
    )


def combine(tables: list[pd.DataFrame], how: str = "median") -> np.ndarray:
    """Per-frame ensemble of several models' coordinates.

    Parameters
    ----------
    tables : list of DataFrame
        Coordinates of the same video from several models (equal length).
    how : {"median", "wmean"}
        Median of the positions, or mean weighted by the head likelihood.

    Returns
    -------
    ndarray
        ``(frames, 4)`` array of ``cx, cy, hx, hy`` (px).
    """
    st = np.stack([d[["cx", "cy", "hx", "hy"]].values for d in tables])
    if how == "median":
        return np.median(st, axis=0)
    if how != "wmean":
        raise ValueError("how must be 'median' or 'wmean'")
    w = np.maximum(np.stack([d["hl"].values for d in tables]), 1e-3)[:, :, None]
    return (st * w).sum(0) / w.sum(0)
