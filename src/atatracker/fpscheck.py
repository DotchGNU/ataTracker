"""Frame-rate check of the videos before any analysis (``ata check-fps``).

Every time of the analysis is ``frame index / signal.fps``. A video whose real frame rate differs
from ``signal.fps`` (a re-encoded copy with fewer frames per second, a 59.94-fps recording) gives
wrong times without any error. This step reads, with ffprobe, for every original video and every
video DeepLabCut analysed:

* ``r_frame_rate`` (container base rate),
* ``avg_frame_rate`` (container average rate),
* the measured rate ``nb_frames / duration`` (frames counted by ffprobe when the container does
  not store ``nb_frames``),

and stops when any of them deviates from ``signal.fps`` by more than ``fps_check.tolerance_rel``.
The rate is never changed silently: the researcher decides what to do with the listed videos.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import prepare_step_dir, write_csv, write_manifest
from .config import Config, config_to_dict, parse_rate

LOG = logging.getLogger("atatracker")
CSV_NAME = "source_fps_check.csv"
ROLES = ("original", "dlc_input")


class FrameRateError(ValueError):
    """Raised when a video's frame rate deviates from the configured one (the message holds the table)."""


def _rate(text: str | None) -> float:
    """Frame rate (frames/s) from an ffprobe rate string such as ``"30000/1001"``; NaN for ``0/0`` or missing."""
    if text in (None, "", "N/A", "0/0"):
        return np.nan
    try:
        v = parse_rate(text)
    except (ValueError, ZeroDivisionError):
        return np.nan
    return v if v > 0 else np.nan


def _num(text: Any) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _ffprobe_json(ffprobe: str, path: str, extra: list[str], entries: str) -> dict[str, Any]:
    cmd = [ffprobe, "-v", "error", *extra, "-select_streams", "v:0", "-show_entries", entries, "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def probe_video(ffprobe: str, path: str, count_mode: str = "packets") -> dict[str, Any]:
    """Read the frame-rate fields of the first video stream.

    Parameters
    ----------
    ffprobe : str
        ffprobe executable.
    path : str
        Video file.
    count_mode : {"packets", "decode"}
        How frames are counted when the container has no ``nb_frames``: ``packets`` counts video
        packets (fast, one packet per frame for the usual codecs), ``decode`` decodes every frame.

    Returns
    -------
    dict
        ``r_frame_rate``, ``avg_frame_rate`` (strings as reported), ``nb_frames`` (frames),
        ``frames_source`` (``container`` / ``counted_packets`` / ``counted_frames``),
        ``duration_s`` (s) and ``duration_source`` (``stream`` / ``format``).
    """
    j = _ffprobe_json(ffprobe, path, [], "stream=r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration")
    streams = j.get("streams") or []
    if not streams:
        raise ValueError(f"{path}: no video stream")
    s = streams[0]
    dur, dsrc = _num(s.get("duration")), "stream"
    if np.isnan(dur):
        dur, dsrc = _num((j.get("format") or {}).get("duration")), "format"
    nb, fsrc = _num(s.get("nb_frames")), "container"
    if np.isnan(nb):
        if count_mode == "decode":
            c = _ffprobe_json(ffprobe, path, ["-count_frames"], "stream=nb_read_frames")
            nb, fsrc = _num((c.get("streams") or [{}])[0].get("nb_read_frames")), "counted_frames"
        else:
            c = _ffprobe_json(ffprobe, path, ["-count_packets"], "stream=nb_read_packets")
            nb, fsrc = _num((c.get("streams") or [{}])[0].get("nb_read_packets")), "counted_packets"
    return dict(
        r_frame_rate=s.get("r_frame_rate"),
        avg_frame_rate=s.get("avg_frame_rate"),
        nb_frames=nb,
        frames_source=fsrc,
        duration_s=dur,
        duration_source=dsrc if not np.isnan(dur) else "",
    )


def evaluate_video(info: dict[str, Any], expected_fps: float, tolerance_rel: float) -> dict[str, Any]:
    """Compare the three rates of one video with the expected rate.

    Parameters
    ----------
    info : dict
        Output of :func:`probe_video`.
    expected_fps : float
        Configured frame rate (frames/s).
    tolerance_rel : float
        Largest accepted relative deviation ``|rate / expected - 1|`` (dimensionless).

    Returns
    -------
    dict
        ``r_fps``, ``avg_fps``, ``measured_fps`` (frames/s), their relative deviations
        ``dev_r``, ``dev_avg``, ``dev_measured``, ``status`` (``OK`` / ``FAIL``) and ``reason``.
        A rate the container does not report (``0/0``) is listed as NaN and does not fail on its
        own; a video whose measured rate cannot be computed fails.
    """
    r, avg = _rate(info.get("r_frame_rate")), _rate(info.get("avg_frame_rate"))
    nb, dur = info.get("nb_frames", np.nan), info.get("duration_s", np.nan)
    meas = nb / dur if (np.isfinite(nb) and np.isfinite(dur) and dur > 0) else np.nan
    row: dict[str, Any] = dict(r_fps=r, avg_fps=avg, measured_fps=meas)
    reasons = []
    for key, v in (("r", r), ("avg", avg), ("measured", meas)):
        dev = abs(v / expected_fps - 1) if np.isfinite(v) else np.nan
        row[f"dev_{key}"] = dev
        if np.isfinite(dev) and dev > tolerance_rel:
            reasons.append(f"{key} {v:.4f} fps")
    if not np.isfinite(meas):
        reasons.append("measured rate not available (no frame count or duration)")
    row["status"] = "FAIL" if reasons else "OK"
    row["reason"] = "; ".join(reasons)
    return row


def list_videos(dirs: list[str], patterns: list[str]) -> list[str]:
    """Video files matching ``patterns`` in each folder (sorted, without duplicates)."""
    out: dict[str, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"fps_check: video folder not found: {d}")
        for pat in patterns:
            for f in glob.glob(os.path.join(d, pat)):
                out.setdefault(os.path.realpath(f), f)
    return sorted(out.values())


def check_videos(cfg: Config, original_dirs: list[str] | None = None, dlc_input_dirs: list[str] | None = None) -> pd.DataFrame:
    """Probe and evaluate every configured video.

    Parameters
    ----------
    cfg : Config
        ``fps_check`` settings, ``signal.fps`` (expected rate) and ``crop.ffprobe``.
    original_dirs, dlc_input_dirs : list of str, optional
        Folders to use instead of ``fps_check.original_dirs`` / ``fps_check.dlc_input_dirs``.

    Returns
    -------
    DataFrame
        One row per video (columns: see docs/outputs.md).
    """
    fc = cfg.fps_check
    if fc.count_mode not in ("packets", "decode"):
        raise ValueError("fps_check.count_mode must be packets or decode")
    dirs = {
        "original": original_dirs if original_dirs is not None else fc.original_dirs,
        "dlc_input": dlc_input_dirs if dlc_input_dirs is not None else fc.dlc_input_dirs,
    }
    if not any(dirs.values()):
        raise ValueError(
            "fps_check: no video folders configured (fps_check.original_dirs / dlc_input_dirs); "
            "list them, or set fps_check.require: false to skip the check"
        )
    expected = cfg.signal.frame_rate
    rows = []
    for role in ROLES:
        files = list_videos(dirs[role], fc.patterns) if dirs[role] else []
        if dirs[role] and not files:
            raise FileNotFoundError(f"fps_check: no videos matching {fc.patterns} in {dirs[role]}")
        for f in files:
            try:
                info = probe_video(cfg.crop.ffprobe, f, fc.count_mode)
            except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as e:
                info = dict(r_frame_rate=None, avg_frame_rate=None, nb_frames=np.nan, frames_source="", duration_s=np.nan)
                LOG.warning("ffprobe failed for %s: %s", f, e)
            row = dict(role=role, video=os.path.basename(f), path=f, **info)
            row.update(expected_fps=expected, tolerance_rel=fc.tolerance_rel)
            row.update(evaluate_video(info, expected, fc.tolerance_rel))
            rows.append(row)
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Short record of a check for ``run_manifest.json``."""
    bad = table[table.status != "OK"]
    return dict(
        status="pass" if bad.empty else "fail",
        expected_fps=cfg.signal.frame_rate,
        fps_setting=str(cfg.signal.fps),
        tolerance_rel=cfg.fps_check.tolerance_rel,
        videos=int(len(table)),
        videos_by_role={r: int((table.role == r).sum()) for r in ROLES},
        failed=int(len(bad)),
        failed_videos=[f"{r.role}: {r.video} ({r.reason})" for r in bad.itertuples()],
        table=CSV_NAME,
    )


def format_table(table: pd.DataFrame) -> str:
    """Per-video table for the console (rates in frames/s)."""
    cols = ["status", "role", "video", "r_frame_rate", "avg_frame_rate", "nb_frames", "duration_s", "measured_fps", "reason"]
    t = table[cols].copy()
    t["measured_fps"] = t.measured_fps.round(4)
    t["duration_s"] = t.duration_s.round(3)
    return t.to_string(index=False)


def _error(table: pd.DataFrame, cfg: Config, where: Path) -> FrameRateError:
    bad = table[table.status != "OK"]
    return FrameRateError(
        f"frame-rate check failed for {len(bad)} of {len(table)} video(s): expected signal.fps = {cfg.signal.fps} "
        f"({cfg.signal.frame_rate:.5f} frames/s, tolerance {cfg.fps_check.tolerance_rel:g} relative). "
        f"Nothing was analysed and the rate was not changed. Table: {where / CSV_NAME}\n{format_table(bad)}"
    )


def run_check_fps(
    cfg: Config,
    out_dir: str | Path,
    overwrite: bool = False,
    original_dirs: list[str] | None = None,
    dlc_input_dirs: list[str] | None = None,
) -> pd.DataFrame:
    """``ata check-fps``: write ``source_fps_check.csv`` and ``run_manifest.json``; raise :class:`FrameRateError` on failure."""
    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    table = check_videos(cfg, original_dirs, dlc_input_dirs)
    write_csv(table, out / CSV_NAME)
    rec = summary(table, cfg)
    write_manifest(
        out,
        "check-fps",
        config_to_dict(cfg),
        inputs=dict(videos=len(table)),
        outputs=dict(rows=len(table), failed=rec["failed"]),
        started=started,
        extra=dict(fps_check=rec),
    )
    LOG.info("frame-rate check (%s):\n%s", rec["status"], format_table(table))
    if rec["status"] != "pass":
        raise _error(table, cfg, out)
    return table


def require_fps_check(cfg: Config, out: Path, step: str) -> dict[str, Any]:
    """Frame-rate check run at the start of an analysis step.

    Returns the manifest record (``status`` ``skipped`` when ``fps_check.require`` is false). On a
    failure ``source_fps_check.csv`` and ``run_manifest_fps-check.json`` are written into ``out``
    and :class:`FrameRateError` is raised before anything is analysed.
    """
    if not cfg.fps_check.require:
        LOG.warning("frame-rate check skipped (fps_check.require: false); times assume %s frames/s", cfg.signal.fps)
        return dict(status="skipped", fps_setting=str(cfg.signal.fps))
    started = _dt.datetime.now()
    table = check_videos(cfg)
    write_csv(table, out / CSV_NAME)
    rec = summary(table, cfg)
    LOG.info("frame-rate check before %s: %s (%d videos)", step, rec["status"], rec["videos"])
    if rec["status"] != "pass":
        write_manifest(
            out,
            "check-fps",
            config_to_dict(cfg),
            inputs=dict(videos=len(table)),
            outputs=dict(rows=len(table), failed=rec["failed"]),
            started=started,
            extra=dict(fps_check=rec, stopped_step=step),
            filename="run_manifest_fps-check.json",
        )
        raise _error(table, cfg, out)
    return rec
