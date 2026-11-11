# Changelog

## 0.2.0 — 2026-11-12

* **End line: weighted chest-head rule is now the default.** The envelope is the centred rolling
  maximum of `chest_weight * chest + head_weight * head` (0.25 / 0.75) over a 12-s window; the end
  line is the last time it reaches 0.15 between the head onset and the HMM post onset + 40 s
  (new function `atatracker.phase.weighted_end_line`; new keys `end_line.chest_weight` and
  `end_line.head_weight`, `end_line.window_s` default 5 -> 12). **Results of `ata phase` change
  for the same coordinates**: end lines are later than with version 0.1. The previous rule is
  reproduced exactly with `end_line: {chest_weight: 0, head_weight: 1, window_s: 5}`.
  `atatracker.phase.end_line` (head amplitude only) is kept.
* Configuration validation of the end-line weights (non-negative, not both 0) and window (> 0).
* Documentation of the rule and its selection (docs/method.md section 4, docs/validation.md);
  regression of both rules against the original analysis tables (17 reference recordings with
  leave-one-out HMMs, 12 non-reference recordings): all differences 0.
* Tests for the weighted end line, the default values, the previous rule by configuration and the validation of the keys.
* The HMM (`atatracker.phase.fit_model`, `PhaseModel`, `state_onsets`) is unchanged.

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
  4-state left-to-right Poisson HMM (hmmlearn `PoissonHMM`), end line from the 5-s rolling max
  of the head amplitude; one checkpoint per animal; twitching plots; `phase_model.json`.
* `ata screen` + `ata review picker` / `choices`: per-animal checkpoint screening and selection page.
* `ata review review` / `viewer`: per-set twitching review page and synchronised video viewer.
* `ata evaluate models` / `grid`: label-free tracking metrics, label error, settings search.
* `ata figures` (stack / bars / xlsx): stacked twitching plot (optional per-bin max
  downsampling for compact SVG), phase-duration bars, native-chart Excel workbook (Office
  2013-2022 theme, schema-order check).
* Every step writes `run_manifest.json`; results are never overwritten without `--overwrite`.
