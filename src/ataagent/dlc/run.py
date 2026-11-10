"""Command lines for the DeepLabCut runner and checkpoint (snapshot) selection without touching projects."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import yaml

from ..config import Config, SnapshotFamily

LOG = logging.getLogger("ataagent")
RUNNER = str(Path(__file__).with_name("runner.py"))


def runner_cmd(cfg: Config, *args: str) -> list[str]:
    """``[<dlc.python>, runner.py, *args]``."""
    return [cfg.dlc.python, RUNNER, *[str(a) for a in args]]


def run(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command (or only log it)."""
    LOG.info("%s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=True).returncode


def analyze_cmd(cfg: Config, video_dir: str, dest_dir: str, project_config: str | None = None) -> list[str]:
    """Runner command for ``analyze_videos``."""
    d = cfg.dlc
    return runner_cmd(
        cfg,
        "analyze",
        "--config",
        project_config or d.project_config,
        "--video-dir",
        video_dir,
        "--dest-dir",
        dest_dir,
        "--shuffle",
        d.shuffle,
        "--gpu",
        -1 if d.gputouse is None else d.gputouse,
        "--ext",
        ",".join(d.video_extensions),
    )


def train_cmd(cfg: Config, project_config: str) -> list[str]:
    """Runner command for fine-tuning ``project_config`` from ``dlc.finetune.init_weights``."""
    f = cfg.dlc.finetune
    return runner_cmd(
        cfg,
        "train",
        "--config",
        project_config,
        "--init-weights",
        f.init_weights,
        "--iteration",
        f.new_iteration,
        "--lr-schedule",
        json.dumps(f.lr_schedule),
        "--max-iters",
        f.max_iters,
        "--save-every",
        f.save_every,
        "--display-every",
        f.display_every,
        "--keep",
        f.keep_snapshots,
        "--shuffle",
        cfg.dlc.shuffle,
        "--net-type",
        f.net_type,
        "--augmenter",
        f.augmenter,
    )


def snapshots(project_config: str, iteration: int) -> list[int]:
    """Training steps of the saved snapshots of a project iteration, in the order DeepLabCut indexes them."""
    proj = os.path.dirname(os.path.abspath(project_config))
    trains = sorted(glob.glob(os.path.join(proj, "dlc-models", f"iteration-{iteration}", "*", "train")))
    if not trains:
        raise FileNotFoundError(f"no dlc-models/iteration-{iteration}/*/train in the project")
    return sorted(
        int(m.group(1))
        for p in glob.glob(os.path.join(trains[0], "snapshot-*.index"))
        if (m := re.search(r"snapshot-(\d+)\.index$", p))
    )


def tag_of(family: str, step: int) -> str:
    """Checkpoint tag ``<family>-<step/1000>k``."""
    return f"{family}-{step // 1000}k"


def snapshot_config(fam: SnapshotFamily, step: int, work_dir: str | Path) -> Path:
    """Temporary project config that selects one snapshot; the project itself is never modified.

    The copy sets ``iteration`` and ``snapshotindex`` and keeps ``project_path`` pointing to the
    original project; ``dlc-models`` and ``training-datasets`` are symbolic links to the originals
    because DeepLabCut resolves them relative to the config file.
    """
    steps = snapshots(fam.project_config, fam.iteration)
    if step not in steps:
        raise ValueError(f"{fam.family}: no snapshot {step} (have {steps})")
    proj = os.path.dirname(os.path.abspath(fam.project_config))
    with open(fam.project_config, encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    c.update(iteration=fam.iteration, snapshotindex=steps.index(step), project_path=proj)
    td = Path(work_dir) / tag_of(fam.family, step)
    td.mkdir(parents=True, exist_ok=True)
    for sub in ("dlc-models", "training-datasets"):
        link = td / sub
        if not link.is_symlink() and not link.exists():
            os.symlink(os.path.join(proj, sub), link)
    p = td / "config.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c, fh)
    return p


def screen_inference(cfg: Config, video_dir: str, out_root: str | Path, dry_run: bool = False) -> list[str]:
    """Run every snapshot of ``dlc.screen`` on the videos of ``video_dir``.

    Output per snapshot: ``<out_root>/dlc-coords_<tag>/``. A snapshot whose folder already holds
    one csv per video is skipped, so the step can be resumed.
    """
    out_root = Path(out_root)
    n_vid = len([p for e in cfg.dlc.video_extensions for p in glob.glob(os.path.join(video_dir, "*" + e))])
    tags = []
    for fam in cfg.dlc.screen:
        for step in fam.steps:
            tag = tag_of(fam.family, step)
            out = out_root / f"dlc-coords_{tag}"
            tags.append(tag)
            if len(glob.glob(str(out / "*.csv"))) >= n_vid > 0:
                LOG.info("skip %s (done)", tag)
                continue
            cp = snapshot_config(fam, step, out_root / "_tmp_configs")
            run(analyze_cmd(cfg, video_dir, str(out), str(cp)), dry_run)
    return tags
