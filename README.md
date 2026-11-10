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
a left-to-right Poisson hidden Markov model. It reports the chest-to-head transition (head
onset), the last twitching (end line) and the chest- and head-phase durations per animal.

```
raw videos ─► crop ─► label ─► fine-tune ─► infer ─► phase ─► screen / review checkpoints ─► phase (final) ─► figures
```

| Stage | What is decided (by the researcher) | Command |
|---|---|---|
| crop | segment and box of each animal in the raw video | `ata crop tool / serve / apply` |
| label | corrected `CHEST` / `HEAD` positions on selected frames | `ata label` |
| fine-tune | base model, label set, held-out videos | `ata dlc build-project / train` |
| infer | checkpoints to run | `ata dlc analyze / screen` |
| phase | analysis settings (defaults = the validated settings) | `ata phase` |
| checkpoint screen | one checkpoint per flagged animal, after checking plots and videos | `ata screen`, `ata review picker / choices` |
| review | problem animals, notes | `ata review review / viewer` |
| figures | group order, row labels | `ata figures stack / bars / xlsx` |

Method, every parameter and its default: [docs/method.md](docs/method.md); the full workflow
with commands: [docs/workflow.md](docs/workflow.md).

## What the results are and are not

* **Are**: per-animal head onset and end line (minutes), phase durations, peak counts and
  label-free tracking metrics, computed by a fixed, documented algorithm from the coordinates of
  a stated checkpoint.
* **Are not**: ground truth. Phase boundaries have no independent reference; the end line is a
  threshold rule on the head motion amplitude, and amplitudes depend on tracking quality and on
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

Without conda: `pip install -e ".[figures,test]"` in a Python >= 3.10 environment. ffmpeg
(tested: 6.1.1 with libx264) must be on the `PATH` for the crop and label steps, or set
`crop.ffmpeg` / `crop.ffprobe`. Point `dlc.python` to the Python of `ata-dlc`; for GPU use, its
`lib` folder must be on `LD_LIBRARY_PATH` (see environment-dlc.yml).

## Quick start

```bash
ata config my-config.yaml       # copy of configs/default.yaml; replace every /path/to/...
ata phase  -c my-config.yaml --out /path/to/output/phase
ata review review -c my-config.yaml --dir /path/to/output/phase --set-name batch1
ata figures all -c my-config.yaml --phase-dir /path/to/output/phase
```

The minimum configuration for `ata phase`:

```yaml
inputs:
  reference_dir: /path/to/reference_coords        # recordings that train the HMM
  coord_sets:
  - {tag: default-380k, dir: /path/to/coords_default-380k}
  groups_csv: /path/to/groups.csv                  # optional: animal,group
```

## Commands

| Command | What it does |
|---|---|
| `ata config <file>` | write the default configuration |
| `ata crop tool` / `serve` / `apply` | build the browser crop tool; run the localhost helper (motion detection, saving); cut the exported crops with ffmpeg |
| `ata label` | pick uniform + likely-failure frames, pre-label them from a model, build the browser labeller |
| `ata dlc analyze` / `screen` / `pseudo-labels` / `build-project` / `train` | DeepLabCut steps, run by `src/atatracker/dlc/runner.py` in the DeepLabCut environment |
| `ata phase` | signals, peaks, HMM phases, end line, flags, twitching plots |
| `ata screen` | the same for every screened animal x checkpoint |
| `ata review picker` / `choices` / `review` / `viewer` | checkpoint picker, picker export -> choice csv, per-set review page, synchronised video viewer |
| `ata evaluate models` / `grid` | tracking metrics and label error per checkpoint; settings search |
| `ata figures stack` / `bars` / `xlsx` / `all` | stacked twitching plot, phase-duration bars, native-chart Excel workbook |

`ata <command> -h` lists the options. Every command takes `-c <config>`; results are never
overwritten without `--overwrite`.

## Outputs

Per-animal table `source_phase-per-animal.csv`, twitching plots, the fitted model
`phase_model.json`, review pages and figures. Every plot is a 300-dpi transparent PNG, an SVG
and a `source_<name>.csv` with the plotted values; the Excel workbook uses formulas over its
`data_raw` sheet and the Office 2013-2022 theme. All files and columns:
[docs/outputs.md](docs/outputs.md). Checkpoint naming for published weights:
[docs/model-weights.md](docs/model-weights.md).

## Reproducibility

* One YAML configuration holds every parameter; `run_manifest.json` in each output folder records
  the configuration, package versions, input and output counts and times.
* Fixed seeds (`phase.random_seed`, `figures.jitter_seed`); the HMM starts from deterministic
  initial parameters.
* DeepLabCut projects are never modified: snapshots are selected through temporary config copies,
  fine-tune projects are new folders.
* External tools are called, not bundled: DeepLabCut 2.2.3 and ffmpeg 6.1.1 (versions pinned in
  the environment files).

## Validation

On archived coordinates of the original study the phase step reproduces the original head onsets
and end lines exactly (difference 0 for 3 animals, and 0 over all 38 animals of that analysis,
including leave-one-out models and per-animal checkpoints): [docs/validation.md](docs/validation.md).
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
