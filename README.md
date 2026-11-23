# ataTracker

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)

ataTracker — asphyxia twitching analysis from DeepLabCut pose tracks, from raw video to phase
durations, with the researcher choosing crops, labels and checkpoints in local review pages.

An agent layer built on this package (MCP tools with decision gates and decision records) is planned as a separate project, ataAgent.

**Contents** — [What it does](#what-it-does) · [What the results are and are not](#what-the-results-are-and-are-not) ·
[Installation](#installation) · [Quick start](#quick-start) · [Commands](#commands) ·
[Outputs](#outputs) · [Reproducibility](#reproducibility) · [Validation](#validation) ·
[Dependencies and licenses](#dependencies-and-licenses) · [Citation](#citation) · [License](#license)

## What it does

A mouse recorded during asphyxia first twitches with the chest, then with the head, then stops.
`atatracker` turns two tracked points per frame (`CHEST`, `HEAD`, from DeepLabCut) into per-second
twitch counts and segments every recording into four phases — *pre → chest → head → post* — with
a stored left-to-right Poisson hidden Markov model. Offset refinement then places the end of
twitching on the motion signal. It reports the chest-to-head transition (head onset), the last
twitching (offset) and the chest- and head-phase durations per animal. Before any analysis it
checks that the videos have the frame rate the analysis assumes.

```
raw videos ─► crop ─► label ─► fine-tune ─► infer ─► check-fps ─► phase ─► screen / review checkpoints ─► phase (final) ─► figures
```

| Stage | What is decided (by the researcher) | Command |
|---|---|---|
| crop | segment and box of each animal in the raw video | `ata crop tool / serve / apply` |
| label | corrected `CHEST` / `HEAD` positions on selected frames | `ata label` |
| fine-tune | base model, label set, held-out videos | `ata dlc build-project / train` |
| infer | checkpoints to run | `ata dlc analyze / screen` |
| frame-rate check | what to do with videos whose rate differs from `signal.fps` (the package stops, never changes the rate) | `ata check-fps` (also run by `ata phase`) |
| phase | analysis settings (defaults = the validated settings); stored HMM or own model | `ata phase`, `ata hmm` |
| checkpoint screen | one checkpoint per flagged animal, after checking plots and videos | `ata screen`, `ata review picker / choices` |
| review | problem animals, notes | `ata review review / viewer` |
| figures | group order, row labels | `ata figures stack / bars / xlsx` |

Method, every parameter and its default: [docs/method.md](docs/method.md); the full workflow
with commands: [docs/workflow.md](docs/workflow.md).

## What the results are and are not

* **Are**: per-animal head onset and offset (minutes), phase durations, peak counts and
  label-free tracking metrics, computed by a fixed, documented algorithm and a stored model
  (identified by its parameter hash) from the coordinates of a stated checkpoint.
* **Are not**: ground truth. Phase boundaries have no independent reference; the offset is a
  threshold rule on a weighted chest-head motion amplitude, and amplitudes depend on tracking quality and on
  the animal's size in the crop. Checkpoint choices are researcher decisions recorded in the
  checkpoint-choice csv and must be reported with the results.

## Installation

Two environments: DeepLabCut needs Python 3.7 and TensorFlow; the analysis needs Python >= 3.10.

```bash
git clone https://github.com/DotchGNU/ataTracker.git
cd ataTracker
conda env create -f environment-analysis.yml      # ata-analysis: the `ata` command
conda env create -f environment-dlc.yml           # ata-dlc: DeepLabCut 2.2.3 (only for the DeepLabCut steps)
```

Without conda: `pip install -e ".[figures,test]"` in a Python >= 3.10 environment. ffmpeg and
ffprobe (tested: 6.1.1 with libx264; ffprobe 8.0.1 for the frame-rate check) must be on the
`PATH` for the crop, label and frame-rate check steps, or set `crop.ffmpeg` / `crop.ffprobe`. Point `dlc.python` to the Python of `ata-dlc`; for GPU use, its
`lib` folder must be on `LD_LIBRARY_PATH` (see environment-dlc.yml).

## Quick start

```bash
ata config my-config.yaml       # copy of configs/default.yaml; replace every /path/to/...
ata check-fps -c my-config.yaml --out /path/to/output/fps-check   # optional: ata phase runs it first
ata phase  -c my-config.yaml --out /path/to/output/phase
ata review review -c my-config.yaml --dir /path/to/output/phase --set-name batch1
ata figures all -c my-config.yaml --phase-dir /path/to/output/phase
```

The minimum configuration for `ata phase`:

```yaml
fps_check:
  original_dirs: [/path/to/original_videos]        # checked before the analysis
  dlc_input_dirs: [/path/to/cropped_videos]        # the videos DeepLabCut analysed
inputs:
  coord_sets:
  - {tag: default-380k, dir: /path/to/coords_default-380k}
  groups_csv: /path/to/groups.csv                  # optional: animal,group
```

The HMM is the stored reference model shipped with the package (`phase.model_file: builtin`;
[docs/hmm-model.md](docs/hmm-model.md)). It is used only if the configuration equals the one it was
trained under; otherwise `ata phase` stops. To use your own model: `ata hmm fit` on
`inputs.reference_dir` and point `phase.model_file` to it; `--refit` fits on every run instead.

## Commands

| Command | What it does |
|---|---|
| `ata config <file>` | write the default configuration |
| `ata crop tool` / `serve` / `apply` | build the browser crop tool; run the localhost helper (motion detection, saving); cut the exported crops with ffmpeg |
| `ata label` | pick uniform + likely-failure frames, pre-label them from a model, build the browser labeller |
| `ata dlc analyze` / `screen` / `pseudo-labels` / `build-project` / `train` | DeepLabCut steps, run by `src/atatracker/dlc/runner.py` in the DeepLabCut environment |
| `ata check-fps` | frame-rate check of original and DeepLabCut input videos (ffprobe); stops on any deviation |
| `ata phase` | frame-rate check, signals, peaks, HMM phases (stored model), offset refinement, flags, twitching plots |
| `ata screen` | the same for every screened animal x checkpoint |
| `ata hmm fit` / `check` | fit and export a model (full + leave-one-out model files); check the configured model |
| `ata review picker` / `choices` / `review` / `viewer` | checkpoint picker, picker export -> choice csv, per-set review page, synchronised video viewer |
| `ata evaluate models` / `grid` | tracking metrics and label error per checkpoint; settings search |
| `ata figures stack` / `bars` / `xlsx` / `all` | stacked twitching plot, phase-duration bars, native-chart Excel workbook |

`ata <command> -h` lists the options. Every command takes `-c <config>`; results are never
overwritten without `--overwrite`. Configuration files of version 0.1 need one change: the
section `end_line` is now `offset_refinement` and must state `chest_weight` and `head_weight`
(see [CHANGELOG.md](CHANGELOG.md)).

## Outputs

Per-animal table `source_phase-per-animal.csv`, the frame-rate table `source_fps_check.csv`,
twitching plots, the applied model `phase_model.json`, review pages and figures. Every plot is a 300-dpi transparent PNG, an SVG
and a `source_<name>.csv` with the plotted values; the Excel workbook uses formulas over its
`data_raw` sheet and the Office 2013-2022 theme. All files and columns:
[docs/outputs.md](docs/outputs.md). Checkpoint naming for published weights:
[docs/model-weights.md](docs/model-weights.md).

## Reproducibility

* One YAML configuration holds every parameter; `run_manifest.json` in each output folder records
  the configuration, package versions, input and output counts and times.
* The HMM is a stored model checked by its parameter hash and by the configuration it was trained
  under; the hash is recorded per animal and in `run_manifest.json`.
* The frame rate of every video is checked against `signal.fps` before the analysis; the result
  is recorded in `source_fps_check.csv` and `run_manifest.json`.
* Fixed seeds (`phase.random_seed`, `figures.jitter_seed`); a refitted HMM starts from
  deterministic initial parameters.
* DeepLabCut projects are never modified: snapshots are selected through temporary config copies,
  fine-tune projects are new folders.
* External tools are called, not bundled: DeepLabCut 2.2.3 and ffmpeg 6.1.1 (versions pinned in
  the environment files).

## Validation

On archived coordinates of the original study the phase step reproduces the per-animal onsets
and offsets of the original analysis exactly. Version 0.2 (stored model, offset refinement):
difference 0 for 35 animals of three cohorts (17 with leave-one-out models, 18 with the full model and per-animal
checkpoint choices). Version 0.1: difference 0 over 38 animals. Details:
[docs/validation.md](docs/validation.md).
`pytest` runs on synthetic data only.

## Dependencies and licenses

Licenses as stated in each package's metadata or license file (checked for the versions below).

| Dependency | Version tested | License | Role |
|---|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | required |
| scipy | 1.17.1 | BSD-3-Clause | required |
| pandas | 3.1.1 | BSD-3-Clause | required |
| hmmlearn | 0.3.3 | BSD (3-clause) | required |
| scikit-learn (hmmlearn dependency) | 1.9.1 | BSD-3-Clause | required |
| matplotlib | 3.11.2 | Matplotlib license (PSF-based) | required |
| PyYAML | 6.0.3 | MIT | required |
| Pillow | 13.0.0 | MIT-CMU | required |
| openpyxl | 3.1.5 | MIT | extra `figures` |
| DeepLabCut | 2.2.3 | LGPL-3.0 | separate environment |
| TensorFlow | 2.11.0 | Apache-2.0 | separate environment (DeepLabCut dependency) |
| ffmpeg | 6.1.1 (conda-forge `gpl` build with libx264) | GPL-2.0-or-later | external program |

All of these licenses are compatible with distributing ataTracker under GPL-3.0-only. DeepLabCut,
TensorFlow and ffmpeg are not included in this repository; they are installed separately and
called at run time (DeepLabCut through `runner.py` in its own environment, ffmpeg as a
subprocess). The ffmpeg build with libx264 is licensed GPL-2.0-or-later, which permits use
under GPL-3.0.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite
DeepLabCut: Mathis, A. et al. *Nat. Neurosci.* 21, 1281–1289 (2018),
[doi:10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y).

## License

ataTracker is free software, distributed under the [GNU General Public License v3.0 only](LICENSE)
(SPDX: `GPL-3.0-only`).

DeepLabCut (LGPL-3.0) and ffmpeg (GPL-2.0-or-later build) are not included in this repository;
they are installed separately and called at run time.

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · Changes: [CHANGELOG.md](CHANGELOG.md) ·
Issues: <https://github.com/DotchGNU/ataTracker/issues>
