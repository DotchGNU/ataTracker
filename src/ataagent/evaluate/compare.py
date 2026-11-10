"""Compare checkpoints: label-free metrics per video and error against hand labels."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd

from ..common import list_coord_files, load_coords, prepare_step_dir, write_csv, write_manifest
from ..config import Config, config_to_dict
from .metrics import label_errors, quality_metrics, summarise_label_errors

LOG = logging.getLogger("ataagent")


def compare_models(cfg: Config, out_dir: str | Path, labels_json: str | None = None, overwrite: bool = False) -> pd.DataFrame:
    """Metrics of every coordinate set in ``inputs.coord_sets``.

    Writes ``source_per-video-metrics.csv`` (one row per set and video), and with a labeller
    export ``source_label-error.csv`` (one row per labelled frame) and
    ``source_label-error-summary.csv`` (one row per set).
    """
    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    frames = []
    if labels_json:
        frames = [f for f in json.loads(Path(labels_json).read_text(encoding="utf-8"))["frames"] if f["confirmed"]]
    rows, errs = [], []
    for s in cfg.inputs.coord_sets:
        for name, f in list_coord_files(s.dir, cfg.inputs.coord_glob, cfg.inputs.name_separator).items():
            d = load_coords(f, cfg.signal.chest_bodypart, cfg.signal.head_bodypart)
            rows.append(dict(tag=s.tag, animal=name, frames=len(d), **quality_metrics(d, cfg.quality)))
            vf = [x for x in frames if x["video"] == name]
            if vf:
                e = label_errors(d, vf)
                e.insert(0, "animal", name)
                e.insert(0, "tag", s.tag)
                errs.append(e)
    table = pd.DataFrame(rows)
    write_csv(table, out / "source_per-video-metrics.csv")
    n_err = 0
    if errs:
        e = pd.concat(errs, ignore_index=True)
        write_csv(e, out / "source_label-error.csv")
        summ = pd.DataFrame([dict(tag=t, **summarise_label_errors(g)) for t, g in e.groupby("tag", sort=False)])
        write_csv(summ, out / "source_label-error-summary.csv")
        n_err = len(e)
    write_manifest(
        out,
        "evaluate-models",
        config_to_dict(cfg),
        inputs=dict(coord_sets=len(cfg.inputs.coord_sets), labelled_frames=len(frames)),
        outputs=dict(rows=len(table), label_errors=n_err),
        started=started,
    )
    return table
