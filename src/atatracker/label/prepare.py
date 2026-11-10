"""Pick frames to label (uniform + likely tracking failures), extract them and build the labeller page.

Pre-labels are the model predictions, so the user only corrects points that are wrong.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from ..common import load_coords
from ..config import Config
from ..crop.media import probe
from ..crop.tool import template

LOG = logging.getLogger("atatracker")


def outlier_scores(coords, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame failure score and flag.

    A frame is flagged when the head-chest distance robust z exceeds ``quality.body_length_z``,
    the head step exceeds ``quality.head_jump_px`` (px) or the head likelihood is below
    ``quality.likelihood_low``. Score = |z| / z_max + jump / jump_max + (likelihood below threshold).

    Returns
    -------
    score, flag, z, jump : ndarray
    """
    q = cfg.quality
    hc = np.hypot(coords.hx - coords.cx, coords.hy - coords.cy).values
    med = np.median(hc)
    mad = 1.4826 * np.median(np.abs(hc - med))
    z = (hc - med) / max(mad, 1e-6)
    jump = np.hypot(coords.hx.diff(), coords.hy.diff()).fillna(0).values
    low = coords.hl.values < q.likelihood_low
    score = np.abs(z) / q.body_length_z + jump / q.head_jump_px + low
    flag = (np.abs(z) > q.body_length_z) | (jump > q.head_jump_px) | low
    return score, flag, z, jump


def pick_frames(
    score: np.ndarray, flag: np.ndarray, n: int, fps: float, taken: list[int], n_frames: int, min_gap_s: float, edge_s: float
) -> list[int]:
    """Highest-scoring flagged frames, at least ``min_gap_s`` apart and ``edge_s`` away from both ends."""
    chosen: list[int] = []
    for i in np.argsort(-score):
        if not flag[i]:
            break
        if i < edge_s * fps or i > n_frames - edge_s * fps:
            continue
        if all(abs(i - j) / fps >= min_gap_s for j in chosen + taken):
            chosen.append(int(i))
        if len(chosen) >= n:
            break
    return chosen


def uniform_frames(n: int, fps: float, n_frames: int, taken: list[int], min_gap_s: float, edge_s: float) -> list[int]:
    """``n`` evenly spaced frames inside the edges, skipping those closer than ``min_gap_s`` to ``taken``."""
    out: list[int] = []
    for t in np.linspace(edge_s * fps, n_frames - edge_s * fps, n + 2)[1:-1]:
        i = int(t)
        if all(abs(i - j) / fps >= min_gap_s for j in taken + out):
            out.append(i)
    return out


def prepare(cfg: Config, video_dir: str, coords_dir: str, out_dir: str, html_only: bool = False) -> Path:
    """Extract frames and write ``prelabels.json``, ``data.js`` and ``labeler.html`` into ``out_dir``.

    Parameters
    ----------
    cfg : Config
        ``label`` and ``quality`` settings, ``crop.ffmpeg`` / ``crop.ffprobe``.
    video_dir : str
        Folder of the (cropped) ``*.mp4`` videos.
    coords_dir : str
        DeepLabCut coordinates of those videos (pre-labels).
    out_dir : str
        Labelling folder (``frames/<video>/img<frame>.png``).
    html_only : bool
        Rewrite the page from an existing ``prelabels.json`` without extracting frames.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lc = cfg.label
    if html_only:
        return write_page(cfg, out, json.loads((out / "prelabels.json").read_text(encoding="utf-8")))
    items = []
    for vp in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
        stem = os.path.basename(vp)[:-4]
        cs = glob.glob(os.path.join(coords_dir, f"{stem}{cfg.inputs.name_separator}*.csv"))
        if not cs:
            LOG.warning("no coordinates for %s", stem)
            continue
        d = load_coords(cs[0], cfg.signal.chest_bodypart, cfg.signal.head_bodypart)
        info = probe(cfg.crop.ffprobe, vp)
        fps, nfr = info["fps"], len(d)
        score, flag, z, jump = outlier_scores(d, cfg)
        q = cfg.quality

        def why(i: int, z=z, jump=jump, d=d, q=q) -> str:
            return ", ".join(
                t
                for t, c in (
                    ("head-chest distance outlier", abs(z[i]) > q.body_length_z),
                    ("head jumped", jump[i] > q.head_jump_px),
                    ("low head likelihood", d.hl[i] < q.likelihood_low),
                )
                if c
            )

        out_f = pick_frames(score, flag, lc.n_outlier, fps, [], nfr, lc.min_gap_s, lc.edge_s)
        uni = uniform_frames(lc.n_uniform, fps, nfr, out_f, lc.min_gap_s, lc.edge_s)
        frames = sorted([(i, "outlier") for i in out_f] + [(i, "uniform") for i in uni])
        fdir = out / "frames" / stem
        fdir.mkdir(parents=True, exist_ok=True)
        sel = "+".join(f"eq(n\\,{i})" for i, _ in frames)
        tmp = str(fdir / "_tmp_%03d.png")
        subprocess.run([cfg.crop.ffmpeg, "-v", "error", "-y", "-i", vp, "-vf", f"select='{sel}'", "-vsync", "0", tmp], check=True)
        for k, (i, b) in enumerate(frames, 1):
            os.replace(tmp % k, fdir / f"img{i:04d}.png")
            r = d.iloc[i]
            it = dict(
                video=stem,
                image=f"img{i:04d}.png",
                frame=i,
                w=info["w"],
                h=info["h"],
                chest=[round(r.cx, 2), round(r.cy, 2)],
                head=[round(r.hx, 2), round(r.hy, 2)],
                src="model",
                batch=b,
                mc=[round(r.cx, 2), round(r.cy, 2)],
                mh=[round(r.hx, 2), round(r.hy, 2)],
                head_lik=round(float(r.hl), 3),
                chest_lik=round(float(r.cl), 3),
                why=why(i) if b == "outlier" else "",
                confirmed=False,
            )
            it["k"] = f"{stem}/{it['image']}"
            it["prio"] = 1 if b == "outlier" else 3
            items.append(it)
        LOG.info(
            "%s: %d frames at %.2f fps, flagged %.1f %%, picked %d outlier + %d uniform",
            stem,
            nfr,
            fps,
            flag.mean() * 100,
            len(out_f),
            len(uni),
        )
    items.sort(key=lambda i: (i["prio"], i["video"], i["frame"]))
    (out / "prelabels.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    return write_page(cfg, out, items)


def write_page(cfg: Config, out: Path, items: list[dict]) -> Path:
    """Write ``data.js`` and ``labeler.html`` (export saved through the helper server when it runs)."""
    (out / "data.js").write_text("const FRAMES = " + json.dumps(items) + ";\n", encoding="utf-8")
    html = (
        template("labeler.html")
        .replace("__LS__", cfg.label.store_key)
        .replace("__SCORER__", cfg.label.scorer)
        .replace("__CONVENTION__", cfg.label.convention)
        .replace("__API__", f"http://localhost:{cfg.crop.server_port}")
        .replace("__SAVE_DIR__", out.resolve().as_posix())
    )
    p = out / "labeler.html"
    p.write_text(html, encoding="utf-8")
    LOG.info("labeler: %d frames (%d outlier frames)", len(items), sum(i["prio"] == 1 for i in items))
    return p
