"""ffmpeg / ffprobe calls (the tools are called, never bundled)."""

from __future__ import annotations

import json
import subprocess

import numpy as np


def probe(ffprobe: str, path: str) -> dict[str, float]:
    """Width, height (px), frame rate (frames/s) and duration (s) of the first video stream."""
    o = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    j = json.loads(o)
    s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return dict(w=s["width"], h=s["height"], fps=float(n) / float(d), dur=float(j["format"]["duration"]))


def grey_frames(
    ffmpeg: str, path: str, fps_expr: str, w: int, h: int, start: float | None = None, dur: float | None = None
) -> np.ndarray:
    """Decode a video to downscaled grey frames.

    Returns
    -------
    ndarray
        ``(n, h, w)`` float32 grey levels (0-255).
    """
    cmd = [ffmpeg, "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-vf", f"fps={fps_expr},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, check=False)
    a = np.frombuffer(r.stdout, np.uint8)
    n = len(a) // (w * h)
    return a[: n * w * h].reshape(n, h, w).astype(np.float32)
