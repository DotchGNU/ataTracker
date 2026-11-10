"""Suggest raw file, time and box for existing reference clips (multi-scale normalised cross-correlation).

Only suggestions: the user confirms every crop in the browser tool.
"""

from __future__ import annotations

import io
import logging
import subprocess

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

from ..config import CropConfig
from .media import grey_frames

LOG = logging.getLogger("ataagent")


def ncc(img: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation map of template ``t`` over ``img`` ("valid" positions)."""
    t = t - t.mean()
    tn = np.sqrt((t * t).sum()) + 1e-6
    k = t.shape
    num = fftconvolve(img, t[::-1, ::-1], mode="valid")
    ones = np.ones(k, np.float32)
    s1 = fftconvolve(img, ones, mode="valid")
    s2 = fftconvolve(img * img, ones, mode="valid")
    var = np.maximum(s2 - s1 * s1 / (k[0] * k[1]), 1e-6)
    return num / (np.sqrt(var) * tn)


def suggest(cfg: CropConfig, raw_paths: dict[str, str], ref_paths: dict[str, str]) -> dict[str, list[dict]]:
    """Best-matching raw position for each reference clip.

    Raw videos are sampled every ``suggest_step_s`` and downscaled to ``detect_w x detect_h``;
    reference frames at ``suggest_reference_times_s`` are matched at template widths
    ``suggest_scales_w`` (downscaled px). Returns, per reference clip, the ``suggest_top`` best
    ``raw, t (s), x, y, w (raw px), score`` candidates.
    """
    scale = 4  # raw px per downscaled px, for 1920 x 1080 raw videos
    raws = {n: grey_frames(cfg.ffmpeg, p, f"1/{cfg.suggest_step_s:g}", cfg.detect_w, cfg.detect_h) for n, p in raw_paths.items()}
    out: dict[str, list[dict]] = {}
    for a, rp in ref_paths.items():
        refs = []
        for t in cfg.suggest_reference_times_s:
            r = subprocess.run(
                [
                    cfg.ffmpeg,
                    "-v",
                    "error",
                    "-ss",
                    str(t),
                    "-i",
                    rp,
                    "-vf",
                    "format=gray",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                capture_output=True,
                check=False,
            ).stdout
            if r:
                refs.append(np.asarray(Image.open(io.BytesIO(r)), np.float32))
        best = []
        for rn, frames in raws.items():
            top: tuple[float, dict | None] = (-1.0, None)
            for ref in refs:
                for tw in cfg.suggest_scales_w:
                    th = int(round(tw * ref.shape[0] / ref.shape[1]))
                    tpl = np.asarray(Image.fromarray(ref).resize((tw, th), Image.BILINEAR), np.float32)
                    for fi, fr in enumerate(frames):
                        m = ncc(fr, tpl)
                        y, x = np.unravel_index(np.argmax(m), m.shape)
                        if m[y, x] > top[0]:
                            top = (
                                float(m[y, x]),
                                dict(
                                    raw=rn,
                                    t=fi * cfg.suggest_step_s,
                                    x=int(x * scale),
                                    y=int(y * scale),
                                    w=int(tw * scale),
                                    score=round(float(m[y, x]), 3),
                                ),
                            )
            if top[1]:
                best.append(top[1])
        out[a] = sorted(best, key=lambda d: -d["score"])[: cfg.suggest_top]
        LOG.info("suggestions for %s: %s", a, [(d["raw"], d["t"], d["score"]) for d in out[a]])
    return out
