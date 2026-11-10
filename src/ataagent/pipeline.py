"""Phase analysis and checkpoint screening over sets of DeepLabCut coordinate files."""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import list_coord_files, load_coords, prepare_step_dir, write_csv, write_manifest
from .config import Config, config_to_dict
from .evaluate.metrics import FLAG_METRICS, flag_against_reference, quality_metrics
from .phase import PhaseModel, PhaseResult, fit_model, segment
from .signal import Signals, compute_signals

LOG = logging.getLogger("ataagent")


@dataclass
class Recording:
    """Coordinates, signals and quality metrics of one coordinate file."""

    name: str
    tag: str
    path: str
    coords: pd.DataFrame
    sig: Signals
    quality: dict[str, float]


def load_recording(name: str, tag: str, path: str, cfg: Config) -> Recording:
    """Read one coordinate file and compute its signals and quality metrics."""
    coords = load_coords(path, cfg.signal.chest_bodypart, cfg.signal.head_bodypart)
    return Recording(
        name, tag, path, coords, compute_signals(coords, cfg.signal, cfg.phase.bin_s), quality_metrics(coords, cfg.quality)
    )


def load_reference(cfg: Config) -> dict[str, Recording]:
    """All recordings of ``inputs.reference_dir`` (the HMM training set)."""
    files = list_coord_files(
        cfg.inputs.reference_dir, cfg.inputs.reference_glob or cfg.inputs.coord_glob, cfg.inputs.name_separator
    )
    if not files:
        raise FileNotFoundError("no coordinate files matching the reference pattern in inputs.reference_dir")
    return {n: load_recording(n, "reference", f, cfg) for n, f in files.items()}


def fit_reference(ref: dict[str, Recording], cfg: Config, exclude: str | None = None) -> PhaseModel:
    """Fit the phase model on the reference recordings, optionally leaving one out."""
    return fit_model({n: r.sig for n, r in ref.items() if n != exclude}, cfg.phase)


def checkpoint_choice(cfg: Config) -> dict[str, str]:
    """Animal -> checkpoint tag from ``inputs.checkpoint_choice`` (csv with ``animal, checkpoint_tag``); empty if unset."""
    if not cfg.inputs.checkpoint_choice:
        return {}
    t = pd.read_csv(cfg.inputs.checkpoint_choice)
    return dict(zip(t.animal.astype(str), t.checkpoint_tag.astype(str), strict=True))


def _lookup(path: str | None, key: str, value: str) -> dict[str, object]:
    if not path:
        return {}
    t = pd.read_csv(path)
    return dict(zip(t[key].astype(str), t[value], strict=True))


def phase_row(rec: Recording, res: PhaseResult) -> dict[str, object]:
    """Per-animal result row (times in min unless the column name says s)."""
    hon, end = res.head_on_s, res.end_s
    has = not np.isnan(hon)
    sig = rec.sig
    return dict(
        animal=rec.name,
        checkpoint_tag=rec.tag,
        coord_file=Path(rec.path).name,
        frames=len(sig.chest),
        fps=sig.fps,
        dur_min=sig.duration_s / 60,
        hmm_chest_on_min=res.chest_on_s / 60,
        head_on_min=hon / 60,
        end_min=end / 60,
        hmm_post_on_min=res.post_on_s / 60,
        chest_phase_min=hon / 60,
        head_phase_min=(end - hon) / 60,
        head_peaks_in_head_phase=int(((sig.head_peaks_s >= hon) & (sig.head_peaks_s <= end)).sum()) if has else 0,
        chest_peaks_in_chest_phase=int((sig.chest_peaks_s < hon).sum()) if has else 0,
        **rec.quality,
    )


def run_phase(cfg: Config, out_dir: str | Path, overwrite: bool = False, plots: bool = True) -> pd.DataFrame:
    """Phase analysis of every target recording with one checkpoint per animal.

    Targets are the animals of the first coordinate set; an animal listed in
    ``inputs.checkpoint_choice`` takes its coordinates from the set with that tag. The HMM is
    trained on all reference recordings; a target that is itself a reference recording is
    segmented with a model trained on the other reference recordings (``phase.leave_one_out``).

    Writes ``phase_model.json``, ``source_phase-per-animal.csv``, twitching plots
    (``twitching/``), the phase-duration bar plot and ``run_manifest.json``.
    """
    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    ref = load_reference(cfg)
    model = fit_reference(ref, cfg)
    model.save(out / "phase_model.json")
    sets = {s.tag: list_coord_files(s.dir, cfg.inputs.coord_glob, cfg.inputs.name_separator) for s in cfg.inputs.coord_sets}
    default_tag = cfg.inputs.coord_sets[0].tag
    choice = checkpoint_choice(cfg)
    unknown = sorted(set(choice.values()) - set(sets))
    if unknown:
        raise KeyError(f"checkpoint_choice names tags without a coordinate set: {unknown}")
    groups = _lookup(cfg.inputs.groups_csv, "animal", "group")
    manual = _lookup(cfg.inputs.manual_csv, "animal", "manual_last_s")
    ref_max = pd.DataFrame([r.quality for r in ref.values()]).max().to_dict()
    rows, recs = [], {}
    for name in sorted(sets[default_tag]):
        tag = choice.get(name, default_tag)
        if name not in sets[tag]:
            raise FileNotFoundError(f"{name}: no coordinates in set {tag}")
        rec = load_recording(name, tag, sets[tag][name], cfg)
        loo = cfg.phase.leave_one_out and name in ref
        m = fit_reference(ref, cfg, exclude=name) if loo else model
        res = segment(rec.sig, m, cfg.phase.bin_s, cfg.end_line)
        row = phase_row(rec, res)
        row.update(group=groups.get(name, ""), selection="chosen" if name in choice else "default", hmm_leave_one_out=loo)
        mv = manual.get(name, np.nan)
        row["manual_last_s"] = float(mv) if mv == mv else np.nan
        row["end_minus_manual_s"] = res.end_s - row["manual_last_s"]
        flags = flag_against_reference(row, ref_max)
        if np.isnan(res.head_on_s):
            flags.append("no head phase")
        if abs(row["end_minus_manual_s"]) > cfg.quality.end_vs_manual_flag_s:
            flags.append(f"|end - manual| > {cfg.quality.end_vs_manual_flag_s:g} s")
        row["metric_flags"] = "; ".join(flags)
        rows.append(row)
        recs[name] = (rec, res)
        LOG.info("%s [%s] head onset %.3f min, end %.3f min", name, tag, row["head_on_min"], row["end_min"])
    table = pd.DataFrame(rows)
    write_csv(table, out / "source_phase-per-animal.csv")
    write_csv(
        pd.DataFrame({"metric": list(FLAG_METRICS), "reference_max": [ref_max[m] for m in FLAG_METRICS]}),
        out / "source_reference-quality-max.csv",
    )
    n_plots = 0
    if plots:
        from .figures.bars import plot_phase_bars
        from .figures.style import use_style
        from .figures.twitching import plot_twitching, twitching_source

        use_style(cfg.figures.font_family, cfg.figures.font_files)
        for name, (rec, res) in recs.items():
            plot_twitching(twitching_source(name, rec.tag, rec.sig, res, cfg), out / "twitching", name, cfg)
            n_plots += 1
        if table.group.astype(bool).any():
            plot_phase_bars(table, out, "bar_phase-durations", cfg)
            n_plots += 1
    write_manifest(
        out,
        "phase",
        config_to_dict(cfg),
        inputs=dict(reference=len(ref), targets=len(rows)),
        outputs=dict(rows=len(rows), plots=n_plots, no_head_phase=int(table.head_on_min.isna().sum())),
        started=started,
    )
    return table


def split_tag(tag: str) -> tuple[str, int | None]:
    """``"<family>-<N>k"`` -> (family, N * 1000); other tags -> (tag, None)."""
    m = re.fullmatch(r"(.+)-(\d+)k", tag)
    return (m.group(1), int(m.group(2)) * 1000) if m else (tag, None)


def run_screen(cfg: Config, out_dir: str | Path, overwrite: bool = False, plots: bool = True) -> pd.DataFrame:
    """Checkpoint screen: every screened animal x every coordinate set.

    Animals are ``inputs.screen_animals`` (all animals of the first set if empty). Writes
    ``source_checkpoint-screen.csv`` (one row per animal and tag), twitching plots
    ``png/<animal>/<tag>.png`` and ``run_manifest.json``; ``ata review picker`` builds the selection page from it.
    """
    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    ref = load_reference(cfg)
    model = fit_reference(ref, cfg)
    model.save(out / "phase_model.json")
    sets = {s.tag: list_coord_files(s.dir, cfg.inputs.coord_glob, cfg.inputs.name_separator) for s in cfg.inputs.coord_sets}
    animals = cfg.inputs.screen_animals or sorted(sets[cfg.inputs.coord_sets[0].tag])
    if plots:
        from .figures.style import use_style
        from .figures.twitching import plot_twitching, twitching_source

        use_style(cfg.figures.font_family, cfg.figures.font_files)
    rows = []
    for name in animals:
        for k, (tag, files) in enumerate(sets.items()):
            if name not in files:
                LOG.warning("%s: no coordinates in set %s", name, tag)
                continue
            rec = load_recording(name, tag, files[name], cfg)
            res = segment(rec.sig, model, cfg.phase.bin_s, cfg.end_line)
            fam, it = split_tag(tag)
            row = phase_row(rec, res)
            row.update(tag=tag, family=fam, iteration=it, set_order=k, is_default=(k == 0))
            rows.append(row)
            if plots:
                plot_twitching(twitching_source(name, tag, rec.sig, res, cfg), out / "png" / name, tag, cfg, markers="lines")
        LOG.info("screened %s over %d checkpoint(s)", name, len(sets))
    table = pd.DataFrame(rows)
    write_csv(table, out / "source_checkpoint-screen.csv")
    write_manifest(
        out,
        "screen",
        config_to_dict(cfg),
        inputs=dict(reference=len(ref), animals=len(animals), coord_sets=len(sets)),
        outputs=dict(rows=len(table)),
        started=started,
    )
    return table
