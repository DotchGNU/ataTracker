# Changelog

## 0.2.0 — 2026-11-23

Results of `ata phase` change for the same coordinates (offset rule); configuration files of
version 0.1 need the changes listed under *Configuration*.

* **Frame-rate check before the analysis.** New step `ata check-fps`: ffprobe reads
  `r_frame_rate`, `avg_frame_rate` and the measured rate `nb_frames / duration` (frames counted
  when the container has no `nb_frames`) of every original video (`fps_check.original_dirs`) and
  every DeepLabCut input video (`fps_check.dlc_input_dirs`) and stops with a per-video table when
  any rate deviates from `signal.fps` by more than `fps_check.tolerance_rel` (0.005). It writes
  `source_fps_check.csv` and records the result in `run_manifest.json`. `ata phase` and
  `ata screen` run it first (`fps_check.require`, default true) and stop before any analysis. The
  frame rate is never changed by the package; the default stays 30000/1001 frames/s.
* **Stored HMM by default.** `ata phase` and `ata screen` load a stored model
  (`phase.model_file`; `builtin` = the reference model shipped in `atatracker/data`, trained on 17
  recordings of the original study, training names replaced by anonymous IDs, parameter hash
  `9a9100d891fa273c`) instead of fitting one on every run. Loading stops on a missing file, a
  parameter-hash mismatch or a configuration that differs from the one the model was trained
  under (signal modes, peak threshold, percentile, likelihood mask, peak distances, bin width,
  frame rate). Training recordings of the model are segmented with stored leave-one-out models
  (`phase.loo_dir`); a missing one stops the run. `phase.refit: true` / `--refit` fits on
  `inputs.reference_dir` as in version 0.1. New `ata hmm fit` (export a full model and its
  leave-one-out models) and `ata hmm check`; model file format in docs/hmm-model.md.
  `phase_model.json` uses that format; the per-animal table has `hmm_params_hash`; the run
  manifest records the model under `hmm`.
* **Offset refinement (weighted chest-head rule) is the default.** The step after the HMM that
  places the end of twitching is called offset refinement (`atatracker.phase.offset_refinement`).
  The envelope is the centred rolling maximum of `0.25 * chest + 0.75 * head` over 12 s; the
  offset is the last time it reaches 0.15 between the head onset and the HMM post onset + 40 s
  (post onset, or recording end, if none). Offsets are later than with the head-only 5-s rule of
  version 0.1, which remains available (`chest_weight: 0`, `head_weight: 1`, `window_s: 5`;
  function `atatracker.phase.end_line`). The output column stays `end_min`.
* **Configuration.** New sections `fps_check` and keys `phase.model_file`, `phase.loo_dir`,
  `phase.refit`; the section `end_line` is renamed `offset_refinement` with new keys
  `chest_weight`, `head_weight` and `window_s` 5 -> 12. A file with the old section name is
  accepted only when it states both weights (a version-0.1 section without them is refused, so
  the old window is never combined with the new weights). `inputs.reference_dir` defaults to
  null: it is needed to fit, and otherwise only sets the quality-flag thresholds.
* Regression on archived coordinates of the original study: head onset, post onset and offset
  identical (difference 0) for 35 animals of three cohorts, against the original scripts and
  master workbooks (docs/validation.md). Tests for the frame-rate check (stand-in ffprobe), the
  stored model and its checks, leave-one-out models, export, and the offset rule.

## 0.1.1 — 2026-11-11

* Project renamed from ataAgent to ataTracker (distribution and import package `atatracker`; the `ata` command is unchanged). No change to the algorithms or outputs.

## 0.1.0 — 2026-11-11

First version, ported from the analysis scripts of the original study with the algorithms unchanged.

* `ata crop` (tool / serve / apply): browser crop tool with motion-based box detection and
  optional crop suggestions (normalised cross-correlation against reference clips); ffmpeg export
  to a fixed size (default 480 x 272, libx264 CRF 18, Lanczos, no audio, frame rate kept).
* `ata label`: frame selection (uniform + likely tracking failures), pre-labels from model
  predictions, browser labeller with JSON export through the local helper.
* `ata dlc` (analyze / screen / pseudo-labels / build-project / train): DeepLabCut 2.2.3 steps run
  by a standalone Python 3.7 runner in the DeepLabCut environment; snapshots are selected through
  temporary configs, projects are never modified.
* `ata phase`: motion signals (chest |dy|, head max(|dx|, |dy|) by default), 1-s peak counts,
  4-state left-to-right Poisson HMM (hmmlearn `PoissonHMM`), offset (called end line in this
  version) from the 5-s rolling max of the head amplitude; one checkpoint per animal; twitching plots; `phase_model.json`.
* `ata screen` + `ata review picker` / `choices`: per-animal checkpoint screening and selection page.
* `ata review review` / `viewer`: per-set twitching review page and synchronised video viewer.
* `ata evaluate models` / `grid`: label-free tracking metrics, label error, settings search.
* `ata figures` (stack / bars / xlsx): stacked twitching plot (optional per-bin max
  downsampling for compact SVG), phase-duration bars, native-chart Excel workbook (Office
  2013-2022 theme, schema-order check).
* Every step writes `run_manifest.json`; results are never overwritten without `--overwrite`.
