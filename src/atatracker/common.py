"""Shared helpers: logging, output folders, run manifests and DeepLabCut coordinate tables."""

from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import platform
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__

LOG = logging.getLogger("atatracker")
COLUMNS = ["cx", "cy", "cl", "hx", "hy", "hl"]


def setup_logging(verbose: bool = False) -> None:
    """Configure the ``atatracker`` logger (one line per message, to stderr)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


class OutputExistsError(FileExistsError):
    """Raised when a step would overwrite existing results without ``--overwrite``."""


def prepare_step_dir(path: str | Path, overwrite: bool = False) -> Path:
    """Create a step output folder.

    Parameters
    ----------
    path : str or Path
        Output folder of the step.
    overwrite : bool
        Allow writing into a folder that already holds a ``run_manifest.json``.

    Returns
    -------
    Path
        The folder (created if needed).
    """
    p = Path(path)
    if (p / "run_manifest.json").exists() and not overwrite:
        raise OutputExistsError(f"{p} already holds results; pass --overwrite to replace them")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _versions(packages: Iterable[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version(), "atatracker": __version__}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_manifest(
    out_dir: str | Path,
    step: str,
    params: dict[str, Any],
    inputs: dict[str, int],
    outputs: dict[str, int],
    started: _dt.datetime,
    extra: dict[str, Any] | None = None,
    packages: Iterable[str] = ("numpy", "scipy", "pandas", "hmmlearn", "matplotlib", "openpyxl"),
    filename: str = "run_manifest.json",
) -> Path:
    """Write ``run_manifest.json`` (parameters, versions, input/output counts, time).

    Parameters
    ----------
    out_dir : str or Path
        Step output folder.
    step : str
        Step name (CLI sub-command).
    params : dict
        Parameters used by the step (usually the full configuration).
    inputs, outputs : dict
        Counts of input and output items, by kind.
    started : datetime
        Start time of the step (local time).
    extra : dict, optional
        Additional step-specific records.
    packages : iterable of str
        Distribution names whose versions are recorded.
    filename : str
        Manifest file name (steps that add pages to another step's folder use ``run_manifest_<step>.json``).
    """
    finished = _dt.datetime.now()
    doc = dict(
        step=step,
        started=started.isoformat(timespec="seconds"),
        finished=finished.isoformat(timespec="seconds"),
        duration_s=round((finished - started).total_seconds(), 3),
        command=sys.argv,
        versions=_versions(packages),
        params=params,
        inputs=inputs,
        outputs=outputs,
    )
    if extra:
        doc.update(extra)
    p = Path(out_dir) / filename
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return p


def animal_name(path: str | Path, separator: str = "DLC_") -> str:
    """Animal (video) name from a DeepLabCut output file name: the part before ``separator``."""
    return os.path.basename(str(path)).split(separator)[0]


def list_coord_files(directory: str | Path, pattern: str = "*DLC_*.csv", separator: str = "DLC_") -> dict[str, str]:
    """Map animal name -> coordinate file for every DeepLabCut csv in ``directory``.

    Raises
    ------
    ValueError
        If two files give the same animal name.
    """
    out: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(str(directory), pattern))):
        name = animal_name(f, separator)
        if name in out:
            raise ValueError(f"two coordinate files for {name}: {out[name]} and {f}")
        out[name] = f
    return out


def load_coords(path: str | Path, chest: str = "CHEST", head: str = "HEAD") -> pd.DataFrame:
    """Read a DeepLabCut coordinate csv (three header rows: scorer, bodyparts, coords).

    Parameters
    ----------
    path : str or Path
        DeepLabCut ``*.csv`` output of ``analyze_videos``.
    chest, head : str
        Body-part names of the chest and head points.

    Returns
    -------
    DataFrame
        Columns ``cx, cy, cl, hx, hy, hl``: chest x, y (px), likelihood, head x, y (px), likelihood; one row per frame.
    """
    d = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    parts = d.columns.get_level_values(1)
    cols = []
    for bp in (chest, head):
        if bp not in set(parts):
            raise KeyError(f"{path}: body part {bp!r} not found (have {sorted(set(parts))})")
        for c in ("x", "y", "likelihood"):
            cols.append(d.loc[:, (slice(None), bp, c)].iloc[:, 0].to_numpy(dtype=float))
    return pd.DataFrame(np.column_stack(cols), columns=COLUMNS)


def write_csv(df: pd.DataFrame, path: str | Path, **kw: Any) -> Path:
    """Write a csv in UTF-8 without index (the format of every table of the package)."""
    df.to_csv(path, index=kw.pop("index", False), encoding="utf-8", **kw)
    return Path(path)
