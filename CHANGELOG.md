# Changelog

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
