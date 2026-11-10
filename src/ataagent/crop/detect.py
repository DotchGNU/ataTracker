"""Find the moving animals in a video segment (motion energy) and propose crop boxes."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def detect_boxes(
    frames: np.ndarray, n_animals: int, raw_w: int, raw_h: int, crop_w: float, out_w: int = 480, out_h: int = 272
) -> dict[str, object]:
    """Crop boxes around the moving region(s) of a segment.

    Motion energy = mean absolute frame difference (Gaussian sigma 2 px), thresholded at
    mean + 2 SD, closed and opened, components smaller than 15 % of the largest dropped. With two
    animals the moving pixels are split by a 1-D two-means on x (the animals lie side by side).

    Parameters
    ----------
    frames : ndarray
        ``(n, h, w)`` downscaled grey frames of the segment.
    n_animals : {1, 2}
        Animals in the segment.
    raw_w, raw_h : int
        Size of the raw video (px).
    crop_w : float
        Minimum crop width in raw px (keeps the pixel scale of earlier crops).
    out_w, out_h : int
        Output crop size (px); sets the box aspect ratio.

    Returns
    -------
    dict
        ``mice`` (list of ``x, y, w, bbox`` in raw px), ``frames``, ``motion_png`` (data URL), ``ds`` (downscaled size).
    """
    if len(frames) < 5:
        raise ValueError("segment too short or unreadable")
    ds_h, ds_w = frames.shape[1:]
    d = ndi.gaussian_filter(np.abs(np.diff(frames, axis=0)), (0, 2, 2))
    m = d.mean(0)
    b = ndi.binary_closing(m > m.mean() + 2.0 * m.std(), iterations=4)
    b = ndi.binary_opening(b, iterations=2)
    lab, k = ndi.label(b)
    if k:
        sizes = ndi.sum(b, lab, range(1, k + 1))
        b = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.15 * sizes.max()])
    ys, xs = np.nonzero(b)
    if len(xs) < 50:
        raise ValueError("no moving region found - check start/end")
    groups = [np.arange(len(xs))]
    if n_animals == 2:
        c = np.array([np.percentile(xs, 25), np.percentile(xs, 75)], float)
        for _ in range(30):
            a = np.abs(xs[:, None] - c[None]).argmin(1)
            c2 = np.array([xs[a == i].mean() if (a == i).any() else c[i] for i in range(2)])
            if np.allclose(c, c2):
                break
            c = c2
        groups = [np.nonzero(a == i)[0] for i in np.argsort(c)]
    sx, sy = raw_w / ds_w, raw_h / ds_h
    ar = out_w / out_h
    mice = []
    for g in groups:
        x0, x1 = np.percentile(xs[g], [1, 99]) * sx
        y0, y1 = np.percentile(ys[g], [1, 99]) * sy
        w = max(crop_w, (x1 - x0) * 1.1, (y1 - y0) * 1.1 * ar)
        w = min(w, raw_w, raw_h * ar)
        h = w / ar
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x = min(max(cx - w / 2, 0), raw_w - w)
        y = min(max(cy - h / 2, 0), raw_h - h)
        mice.append(dict(x=int(x), y=int(y), w=int(w), bbox=[int(x0), int(y0), int(x1), int(y1)]))
    img = (255 * np.clip(m / max(m.max(), 1e-6), 0, 1) ** 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return dict(
        mice=mice,
        frames=int(len(frames)),
        motion_png="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        ds=[int(ds_w), int(ds_h)],
    )
