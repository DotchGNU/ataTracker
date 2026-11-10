"""Synthetic DeepLabCut coordinate tables with known phase boundaries (no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FPS = 30.0


def synthetic_coords(
    chest_on_s: float,
    head_on_s: float,
    last_s: float,
    duration_s: float = 300.0,
    seed: int = 0,
    chest_period_s: float = 1.0,
    head_period_s: float = 2.5,
) -> pd.DataFrame:
    """Chest and head tracks: chest jerks (3 px in y) every ``chest_period_s`` in [chest_on, head_on),
    head jerks (4 px in x and y) every ``head_period_s`` in [head_on, last]; Gaussian noise 0.05 px."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FPS)
    cx = 200 + rng.normal(0, 0.05, n)
    cy = 150 + rng.normal(0, 0.05, n)
    hx = 300 + rng.normal(0, 0.05, n)
    hy = 140 + rng.normal(0, 0.05, n)
    for t in np.arange(chest_on_s, head_on_s, chest_period_s):
        i = int(t * FPS)
        cy[i] += 3.0
    for t in np.arange(head_on_s, last_s + 1e-9, head_period_s):
        i = int(t * FPS)
        hx[i] += 4.0
        hy[i] += 4.0
    lik = np.full(n, 0.99)
    return pd.DataFrame(dict(cx=cx, cy=cy, cl=lik, hx=hx, hy=hy, hl=lik))


def write_dlc_csv(df: pd.DataFrame, path: Path, scorer: str = "DLC_resnet50_test") -> Path:
    """Write ``df`` (cx..hl) in the DeepLabCut csv layout (scorer / bodyparts / coords header rows)."""
    cols = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in ("CHEST", "HEAD") for c in ("x", "y", "likelihood")], names=["scorer", "bodyparts", "coords"]
    )
    out = pd.DataFrame(df[["cx", "cy", "cl", "hx", "hy", "hl"]].values, columns=cols)
    out.to_csv(path)
    return path


BOUNDS = [(10, 80, 200), (12, 95, 210), (8, 70, 180), (15, 100, 220), (10, 85, 190), (9, 75, 205), (11, 90, 215)]


@pytest.fixture
def coord_dirs(tmp_path: Path) -> dict[str, Path]:
    """``reference/`` with six recordings and ``targets/`` with one recording of known boundaries."""
    ref, tgt = tmp_path / "reference", tmp_path / "targets"
    ref.mkdir()
    tgt.mkdir()
    for k, (c0, h0, e0) in enumerate(BOUNDS[:-1]):
        write_dlc_csv(synthetic_coords(c0, h0, e0, seed=k), ref / f"ref{k}DLC_resnet50_test.csv")
    c0, h0, e0 = BOUNDS[-1]
    write_dlc_csv(synthetic_coords(c0, h0, e0, seed=99), tgt / "animalADLC_resnet50_test.csv")
    return dict(reference=ref, targets=tgt, root=tmp_path)


@pytest.fixture
def config_dict(coord_dirs: dict[str, Path]) -> dict:
    """Configuration mapping pointing at the synthetic folders (30 frames/s)."""
    return dict(
        output_dir=str(coord_dirs["root"] / "out"),
        signal=dict(fps=FPS),
        inputs=dict(
            reference_dir=str(coord_dirs["reference"]), coord_sets=[dict(tag="model-1k", dir=str(coord_dirs["targets"]))]
        ),
    )
