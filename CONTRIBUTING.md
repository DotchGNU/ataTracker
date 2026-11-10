# Contributing

Repository: <https://github.com/myLabTechnician/ataAgent>. Bug reports and pull requests go to
the issue tracker of that repository.

## Coding standards

* Python >= 3.10 for the package, except `src/ataagent/dlc/runner.py`, which runs inside the
  DeepLabCut environment: it must stay Python 3.7 compatible and import only the standard library,
  numpy, pandas, PyYAML and (inside functions) deeplabcut.
* DeepLabCut and TensorFlow are never imported by the rest of the package; the analysis installs
  without them.
* Type hints on public functions; NumPy-style docstrings that state units (s, min, px, frames).
* No parameter is hard-coded in logic: add it to the dataclass in `config.py` **and** to
  `configs/default.yaml` with a comment, and document it in `docs/method.md` or `docs/workflow.md`.
  `tests/test_config.py` fails if the YAML and the dataclasses differ.
* Logging through the `ataagent` logger; no `print` in library code.
* A step never overwrites existing results unless `--overwrite` is given (`common.prepare_step_dir`),
  and every step writes `run_manifest.json` through `common.write_manifest`.
* Every plot goes through `figures.style.save_figure` (300-dpi transparent PNG + SVG +
  `source_<name>.csv` with the plotted values).
* Excel output keeps the Office 2013-2022 theme and passes `figures.ooxml.check`.
* Random seeds are fixed in the configuration (`phase.random_seed`, `figures.jitter_seed`).
* All repository text is in English.

## Changing an algorithm

The phase results are validated against the original analysis (docs/validation.md). A change
that alters `head_on_min` or `end_min` for unchanged settings must be a new, documented option
with the old behaviour as default, recorded in `CHANGELOG.md`.

## Tests and lint

```bash
pip install -e ".[figures,test]"
pytest
ruff check .
ruff format --check .
```

Tests use synthetic coordinate tables only (fixed seeds) and run without DeepLabCut, a GPU or
ffmpeg. Never add real videos, coordinates, labels, weights or results to the repository.

## Before publishing: leak check

Run a text search over every file git would track (tracked files plus
`git ls-files --others --exclude-standard`) and over the commit messages before each push. It
must find:

* no absolute machine paths (home or mount folders, drive letters), user or person names,
  institution, project, cohort or dataset names, animal identifiers, real data or results;
* no Hangul or other non-English repository text;
* no tool-vendor names or attribution lines in commit messages (e.g. `Co-Authored-By`).

## Commits

Small, logical commits with plain descriptive messages.
