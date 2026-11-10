"""Optional settings search without ground truth (docs/method.md, section 6).

Varied: chest and head signal modes, peak threshold, normalisation percentile, likelihood mask
and, in a second stage, end-line threshold and window. Objectives per combination:

A  spread of (end line - manual last twitching) over recordings with a manual value (median
   absolute deviation, so a constant definition offset does not count);
B  share of animals whose head onset agrees within ``onset_agreement_min`` (min) between the two
   coordinate sets named in ``grid.compare_sets``;
C  share of target animals without a usable head phase (none, or fewer than ``min_head_peaks`` head peaks).

Rank = mean of the three ranks (lower is better).
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import list_coord_files, load_coords, prepare_step_dir, write_csv, write_manifest
from ..config import Config, config_to_dict
from ..phase import end_line, fit_model, state_onsets
from ..signal import compute_signals

LOG = logging.getLogger("ataagent")


def spread(diff: list[float]) -> float:
    """Median absolute deviation from the median (s); NaN-safe."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    return float(np.median(np.abs(d - np.median(d)))) if len(d) else np.nan


def run_grid(cfg: Config, out_dir: str | Path, overwrite: bool = False) -> pd.DataFrame:
    """Run the settings search; writes ``source_grid-raw.csv`` and ``source_grid-ranked.csv``."""
    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    g = cfg.grid
    manual = {}
    if cfg.inputs.manual_csv:
        m = pd.read_csv(cfg.inputs.manual_csv)
        manual = dict(zip(m.animal.astype(str), m.manual_last_s.astype(float), strict=True))
    tags = g.compare_sets or [s.tag for s in cfg.inputs.coord_sets[:2]]
    if len(tags) != 2:
        raise ValueError("grid.compare_sets must name two coordinate sets")
    dirs = {s.tag: s.dir for s in cfg.inputs.coord_sets}
    ref_files = list_coord_files(
        cfg.inputs.reference_dir, cfg.inputs.reference_glob or cfg.inputs.coord_glob, cfg.inputs.name_separator
    )
    set_files = {t: list_coord_files(dirs[t], cfg.inputs.coord_glob, cfg.inputs.name_separator) for t in tags}
    cache: dict[str, pd.DataFrame] = {}

    def coords(f: str) -> pd.DataFrame:
        if f not in cache:
            cache[f] = load_coords(f, cfg.signal.chest_bodypart, cfg.signal.head_bodypart)
        return cache[f]

    combos = list(itertools.product(g.modes, g.modes, g.peak_thresholds, g.norm_percentiles, g.likelihood_mins))
    rows = []
    for k, (cm, hm, thr, pct, lik) in enumerate(combos, 1):
        c = copy.deepcopy(cfg)
        c.signal.chest_mode, c.signal.head_mode, c.signal.peak_threshold, c.signal.norm_percentile, c.signal.likelihood_min = (
            cm,
            hm,
            thr,
            pct,
            lik,
        )
        ref = {n: compute_signals(coords(f), c.signal, c.phase.bin_s) for n, f in ref_files.items()}
        try:
            model = fit_model(ref, c.phase)
        except Exception as e:  # a combination can make the HMM degenerate
            LOG.warning("HMM failed for %s/%s thr %s pct %s lik %s: %s", cm, hm, thr, pct, lik, e)
            continue

        def bounds(s, model=model, c=c):
            e = state_onsets(model.predict(s.counts), c.phase.bin_s)
            return e[2], e[3]

        bref = {n: bounds(s) for n, s in ref.items()}
        sig = {t: {n: compute_signals(coords(f), c.signal, c.phase.bin_s) for n, f in fs.items()} for t, fs in set_files.items()}
        bset = {t: {n: bounds(s) for n, s in d.items()} for t, d in sig.items()}
        common = sorted(set(bset[tags[0]]) & set(bset[tags[1]]))
        agree = np.array([abs(bset[tags[0]][n][0] - bset[tags[1]][n][0]) / 60 for n in common])
        nohead = np.mean(
            [
                np.isnan(bset[tags[0]][n][0])
                or ((s.head_peaks_s >= bset[tags[0]][n][0]) & (s.head_peaks_s <= s.duration_s)).sum() < g.min_head_peaks
                for n, s in sig[tags[0]].items()
            ]
        )
        for et, ew in itertools.product(g.end_thresholds, g.end_windows_s):

            def end_of(s, hon, post, et=et, ew=ew, c=c):
                if np.isnan(hon):
                    return np.nan
                stop = post if not np.isnan(post) else s.duration_s
                e = end_line(s.head, hon, et, ew, stop + c.end_line.search_after_post_s, s.fps)
                return e if not np.isnan(e) else stop

            d_ref = [end_of(s, *bref[n]) - manual[n] for n, s in ref.items() if n in manual]
            d_set = [end_of(s, *bset[tags[0]][n]) - manual[n] for n, s in sig[tags[0]].items() if n in manual and n not in ref]
            rows.append(
                dict(
                    chest=cm,
                    head=hm,
                    thr=thr,
                    pct=pct,
                    lik_min=lik,
                    end_thr=et,
                    end_win=ew,
                    end_spread_reference_s=spread(d_ref),
                    end_bias_reference_s=float(np.nanmedian(d_ref)) if d_ref else np.nan,
                    end_spread_targets_s=spread(d_set),
                    end_bias_targets_s=float(np.nanmedian(d_set)) if d_set else np.nan,
                    end_spread_pooled_s=spread(d_ref + d_set),
                    onset_agree=float(np.nanmean(agree <= g.onset_agreement_min)) if len(agree) else np.nan,
                    onset_absdiff_median=float(np.nanmedian(agree)) if len(agree) else np.nan,
                    no_head_phase=float(nohead),
                )
            )
        if k % 25 == 0 or k == 1:
            LOG.info("[%d/%d] %s/%s thr %s", k, len(combos), cm, hm, thr)
    raw = pd.DataFrame(rows)
    write_csv(raw, out / "source_grid-raw.csv")
    ranked = raw.copy()
    ranked["rank"] = (ranked.end_spread_pooled_s.rank() + (-ranked.onset_agree).rank() + ranked.no_head_phase.rank()) / 3
    ranked = ranked.sort_values("rank")
    write_csv(ranked, out / "source_grid-ranked.csv")
    write_manifest(
        out,
        "evaluate-grid",
        config_to_dict(cfg),
        inputs=dict(reference=len(ref_files), **{f"set_{t}": len(v) for t, v in set_files.items()}),
        outputs=dict(combinations=len(combos), rows=len(raw)),
        started=started,
    )
    return ranked
