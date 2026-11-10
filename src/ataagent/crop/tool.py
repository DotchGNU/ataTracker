"""Build the browser crop tool and apply exported crops with ffmpeg."""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
from importlib import resources
from pathlib import Path

from ..config import CropConfig
from ..urls import file_url
from .media import probe

LOG = logging.getLogger("ataagent")


def template(name: str) -> str:
    """Text of a packaged HTML template."""
    return resources.files("ataagent").joinpath(f"templates/{name}").read_text(encoding="utf-8")


def raw_videos(cfg: CropConfig) -> dict[str, str]:
    """Raw video name -> path (``raw_patterns`` in ``raw_dir``, minus ``exclude``)."""
    files = sorted({p for pat in cfg.raw_patterns for p in glob.glob(os.path.join(cfg.raw_dir, pat))})
    return {os.path.basename(p): p for p in files if os.path.basename(p) not in set(cfg.exclude)}


def build_tool(cfg: CropConfig, with_suggestions: bool = True) -> Path:
    """Write ``crop_tool.html`` (+ reference thumbnails) into ``tool_dir``.

    Each raw video is probed with ffprobe. Reference clips in ``reference_dir`` (optional) get a
    thumbnail at ``reference_frame_s`` and, if ``with_suggestions``, crop suggestions
    (cached in ``suggest_cache.json``).
    """
    tool = Path(cfg.tool_dir)
    (tool / "ref").mkdir(parents=True, exist_ok=True)
    raws = raw_videos(cfg)
    data = dict(out_w=cfg.out_w, out_h=cfg.out_h, crop_w=cfg.crop_w, cohorts={})
    raw_list = [dict(name=n, url=file_url(p, cfg.url_path_map), **probe(cfg.ffprobe, p)) for n, p in raws.items()]
    animals, refs = [], {}
    if cfg.reference_dir:
        for p in sorted(glob.glob(os.path.join(cfg.reference_dir, "*.mp4"))):
            n = os.path.basename(p)[:-4]
            refs[n] = p
            subprocess.run(
                [
                    cfg.ffmpeg,
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(cfg.reference_frame_s),
                    "-i",
                    p,
                    "-vf",
                    f"scale={cfg.out_w}:{cfg.out_h}",
                    "-frames:v",
                    "1",
                    str(tool / "ref" / f"{n}.png"),
                ],
                check=False,
            )
            animals.append(dict(name=n, ref=f"ref/{n}.png"))
    if refs and with_suggestions:
        cache = tool / "suggest_cache.json"
        if cache.exists():
            sug = json.loads(cache.read_text(encoding="utf-8"))
        else:
            from .suggest import suggest

            sug = suggest(cfg, raws, refs)
            cache.write_text(json.dumps(sug), encoding="utf-8")
        for a in animals:
            a["suggest"] = sug.get(a["name"], [])
    data["cohorts"][cfg.batch] = dict(raws=raw_list, animals=animals)
    html = (
        template("crop_tool.html")
        .replace("/*DATA*/null", json.dumps(data))
        .replace("__API__", f"http://localhost:{cfg.server_port}")
    )
    out = tool / "crop_tool.html"
    out.write_text(html, encoding="utf-8")
    LOG.info("crop tool: %d raw videos, %d reference clips", len(raw_list), len(animals))
    return out


def ffmpeg_command(cfg: CropConfig, entry: dict, raw_path: str, out_path: str) -> list[str]:
    """ffmpeg command for one exported crop.

    Trim, crop, Lanczos scale to ``out_w x out_h``, libx264 ``crf``, no audio, frame rate kept.
    """
    vf = f"crop={entry['w']}:{entry['h']}:{entry['x']}:{entry['y']},scale={cfg.out_w}:{cfg.out_h}:flags=lanczos"
    return [
        cfg.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-ss",
        str(entry["start"]),
        "-to",
        str(entry["end"]),
        "-i",
        raw_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]


def apply_crops(cfg: CropConfig, crops_json: str | Path, dry_run: bool = False, overwrite: bool = False) -> list[list[str]]:
    """Cut every crop listed in the tool's export (``crops.json``) into ``out_dir/<name>.mp4``."""
    entries = json.loads(Path(crops_json).read_text(encoding="utf-8"))
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cmds = []
    for e in entries:
        out = os.path.join(cfg.out_dir, f"{e['name']}.mp4")
        if os.path.exists(out) and not overwrite:
            LOG.warning("%s exists; skipped (use --overwrite)", out)
            continue
        cmd = ffmpeg_command(cfg, e, os.path.join(cfg.raw_dir, e["raw"]), out)
        if overwrite:
            cmd[cmd.index("-n")] = "-y"
        cmds.append(cmd)
        LOG.info("%s", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)
    return cmds
