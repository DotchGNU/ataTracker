"""Stored phase models: the fixed reference HMM and its leave-one-out models (docs/hmm-model.md).

A model file is JSON with the fitted parameters of the four-state Poisson HMM (start
probabilities, transition matrix, Poisson rates per state and count series), a hash of those
parameters, the training recordings, the signal and peak-calling configuration the model was
trained under, the fit settings and library versions::

    kind, left_out, n_components, n_features, states, signals,
    params {startprob, transmat, lambdas}, params_hash, training_animals, n_training,
    config {SIG_CHEST, SIG_HEAD, THR, NORM_HI_PCT, LIK_MIN, DIST_CHEST, DIST_HEAD, BIN_S, FPS},
    fit_settings {...}, converged, n_iter_run, hmmlearn_version, numpy_version

By default ``ata phase`` and ``ata screen`` load the model named by ``phase.model_file``
(``builtin`` = the reference model shipped with the package) instead of fitting one. Loading
stops when the file is missing, its parameter hash does not match, or the configuration it was
trained under differs from the current one; ``phase.refit: true`` (``--refit``) fits instead.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, config_to_dict
from .phase import STATE_NAMES, K, PhaseModel, fit_model
from .signal import Signals

LOG = logging.getLogger("atatracker")
BUILTIN = "builtin"
BUILTIN_FILE = "hmm_reference-17_full.json"
SIGNALS = ["chest_count_per_s", "head_count_per_s"]
CONFIG_TOL = 1e-9
FULL_NAME = "hmm_full.json"
LOO_NAME = "hmm_loo_{name}.json"


class StoredModelMismatchError(ValueError):
    """Raised when a stored model was trained under another signal / peak-calling configuration."""


def store_config(cfg: Config) -> dict[str, Any]:
    """Configuration keys a stored model is checked against (key names of the model file).

    ``SIG_CHEST`` / ``SIG_HEAD`` = ``signal.chest_mode`` / ``head_mode``; ``THR`` =
    ``signal.peak_threshold``; ``NORM_HI_PCT`` = ``signal.norm_percentile``; ``LIK_MIN`` =
    ``signal.likelihood_min``; ``DIST_CHEST`` / ``DIST_HEAD`` = peak distances (frames);
    ``BIN_S`` = ``phase.bin_s`` (s); ``FPS`` = ``signal.fps`` (frames/s).
    """
    s = cfg.signal
    return dict(
        SIG_CHEST=s.chest_mode,
        SIG_HEAD=s.head_mode,
        THR=s.peak_threshold,
        NORM_HI_PCT=s.norm_percentile,
        LIK_MIN=s.likelihood_min,
        DIST_CHEST=s.peak_distance_chest_frames,
        DIST_HEAD=s.peak_distance_head_frames,
        BIN_S=cfg.phase.bin_s,
        FPS=s.frame_rate,
    )


def fit_settings(cfg: Config) -> dict[str, Any]:
    """Fit settings recorded in a model file (``phase.*``)."""
    p = cfg.phase
    return dict(
        iter=p.hmm_n_iter,
        tol=p.hmm_tol,
        seed=p.random_seed,
        params="stl",
        init_params="",
        start_prob=list(p.start_prob),
        init_stay=p.stay_prob,
        init_advance=p.move_prob,
        lambda_floor=p.lambda_floor,
        init_paths="atatracker.phase.init_path (rule-based)",
    )


def params_hash(params: dict[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of the canonical JSON of ``params`` (sorted keys)."""
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def model_record(model: PhaseModel, cfg: Config, left_out: str | None = None) -> dict[str, Any]:
    """JSON record of a fitted model (the model-file format above)."""
    import hmmlearn

    params = model.params()
    return dict(
        kind="loo" if left_out else "full",
        left_out=left_out,
        n_components=K,
        n_features=2,
        states=list(STATE_NAMES),
        signals=list(SIGNALS),
        params=params,
        params_hash=params_hash(params),
        training_animals=list(model.train_names),
        n_training=len(model.train_names),
        config=store_config(cfg),
        fit_settings=fit_settings(cfg),
        converged=model.converged,
        n_iter_run=model.n_iter,
        hmmlearn_version=hmmlearn.__version__,
        numpy_version=np.__version__,
    )


def save_model(model: PhaseModel, path: str | Path, cfg: Config, left_out: str | None = None) -> dict[str, Any]:
    """Write ``model`` as a model file and return the record."""
    rec = model_record(model, cfg, left_out)
    Path(path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def read_record(path: str | Path) -> dict[str, Any]:
    """Read a model file and verify its structure and parameter hash.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the hash or the shapes do not match.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"stored HMM not found: {p} (set phase.model_file, or phase.refit: true / --refit to fit one)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    for key in ("params", "params_hash", "config", "training_animals"):
        if key not in rec:
            raise ValueError(f"{p}: not a stored HMM file (no {key!r})")
    if params_hash(rec["params"]) != rec["params_hash"]:
        raise ValueError(f"{p}: parameter hash mismatch (file changed after it was written)")
    par = rec["params"]
    if (
        np.shape(par["startprob"]) != (K,)
        or np.shape(par["transmat"]) != (K, K)
        or np.shape(par["lambdas"]) != (K, 2)
        or list(rec.get("states", STATE_NAMES)) != list(STATE_NAMES)
    ):
        raise ValueError(f"{p}: expected a {K}-state model with states {STATE_NAMES} and two count series")
    return rec


def model_from_record(rec: dict[str, Any]) -> PhaseModel:
    """:class:`~atatracker.phase.PhaseModel` with the parameters of a model record."""
    p = rec["params"]
    return PhaseModel(
        np.array(p["startprob"], float),
        np.array(p["transmat"], float),
        np.array(p["lambdas"], float),
        int(rec.get("n_training", len(rec["training_animals"]))),
        list(rec["training_animals"]),
        converged=rec.get("converged"),
        n_iter=rec.get("n_iter_run"),
        left_out=rec.get("left_out"),
    )


def check_config(rec: dict[str, Any], cfg: Config, where: str = "") -> None:
    """Raise :class:`StoredModelMismatchError` if ``rec`` was trained under another configuration.

    Numbers are compared with an absolute tolerance of 1e-9, text exactly; a key missing from the
    file counts as a mismatch.
    """
    cur = store_config(cfg)
    bad = []
    for k, b in cur.items():
        if k not in rec["config"]:
            bad.append(f"{k}: missing in the model file, current {b}")
            continue
        a = rec["config"][k]
        num = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
        if not (abs(a - b) <= CONFIG_TOL if num else a == b):
            bad.append(f"{k}: stored {a} vs current {b}")
    if bad:
        raise StoredModelMismatchError(
            f"stored HMM {where} was trained under another configuration: "
            + "; ".join(bad)
            + ". Use the configuration of the model, another model file, or phase.refit: true (--refit)."
        )


def resolve_model_file(model_file: str) -> Path:
    """Path of ``phase.model_file`` (``builtin`` = the reference model shipped in ``atatracker/data``)."""
    if model_file == BUILTIN:
        return Path(str(resources.files("atatracker") / "data" / BUILTIN_FILE))
    return Path(model_file)


@dataclass
class StoredModels:
    """The full stored model and the leave-one-out models found next to it."""

    full: PhaseModel
    full_record: dict[str, Any]
    path: Path
    loo_dir: Path
    loo_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def training_names(self) -> list[str]:
        """Recordings the full model was trained on."""
        return list(self.full.train_names)

    def loo(self, name: str, cfg: Config) -> PhaseModel:
        """Leave-one-out model of training recording ``name`` (checked like the full model).

        Raises
        ------
        FileNotFoundError
            If no stored leave-one-out model for ``name`` exists.
        ValueError
            If its training set is not the full training set without ``name``.
        """
        if name not in self.loo_paths:
            raise FileNotFoundError(
                f"{name} is a training recording of the stored HMM but no leave-one-out model for it was found "
                f"in {self.loo_dir}; add it (ata hmm fit), or set phase.leave_one_out: false"
            )
        rec = read_record(self.loo_paths[name])
        check_config(rec, cfg, f"({self.loo_paths[name].name})")
        if set(rec["training_animals"]) != set(self.training_names) - {name}:
            raise ValueError(f"{self.loo_paths[name]}: training set is not the full training set without {name}")
        return model_from_record(rec)

    def manifest(self) -> dict[str, Any]:
        """Short record for ``run_manifest.json``."""
        return dict(
            source="stored",
            model_file=str(self.path),
            loo_dir=str(self.loo_dir),
            params_hash=self.full_record["params_hash"],
            n_training=len(self.training_names),
            loo_models=len(self.loo_paths),
            config_check="pass",
        )


def load_store(cfg: Config) -> StoredModels:
    """Load and check ``phase.model_file`` and index the leave-one-out models of ``phase.loo_dir``.

    Leave-one-out models are the ``*.json`` model files with ``kind`` ``loo`` in ``phase.loo_dir``
    (default: the folder of the model file); each is read and checked only when it is used.
    """
    path = resolve_model_file(cfg.phase.model_file)
    rec = read_record(path)
    if rec.get("kind", "full") != "full":
        raise ValueError(f"{path}: phase.model_file must be a full model (kind 'full'), got {rec.get('kind')!r}")
    check_config(rec, cfg, f"({path.name})")
    full = model_from_record(rec)
    loo_dir = Path(cfg.phase.loo_dir) if cfg.phase.loo_dir else path.parent
    loo: dict[str, Path] = {}
    if loo_dir.is_dir():
        for f in sorted(loo_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(d, dict) and d.get("kind") == "loo" and d.get("left_out"):
                if d["left_out"] in loo:
                    raise ValueError(f"two leave-one-out models for {d['left_out']}: {loo[d['left_out']]} and {f}")
                loo[d["left_out"]] = f
    LOG.info(
        "stored HMM %s (hash %s, %d training recordings, %d leave-one-out models)",
        path.name,
        rec["params_hash"],
        len(full.train_names),
        len(loo),
    )
    return StoredModels(full, rec, path, loo_dir, loo)


def fit_store(train: dict[str, Signals], cfg: Config, out_dir: Path, leave_one_out: bool = True) -> pd.DataFrame:
    """Fit the full model (and one leave-one-out model per recording) and write them as model files.

    Writes ``hmm_full.json``, ``hmm_loo_<name>.json``, ``index_models.csv`` (file, kind,
    left_out, params_hash, converged, n_iter_run) and ``source_hmm-params.csv`` (one row per model
    and state: start probability, transitions, Poisson rates in peaks per bin).
    """
    names = list(train)
    rows, params = [], []

    def one(model: PhaseModel, file: str, left_out: str | None) -> None:
        rec = save_model(model, out_dir / file, cfg, left_out)
        rows.append(
            dict(
                file=file,
                kind=rec["kind"],
                left_out=left_out or "",
                params_hash=rec["params_hash"],
                converged=rec["converged"],
                n_iter_run=rec["n_iter_run"],
            )
        )
        p = rec["params"]
        for k, st in enumerate(STATE_NAMES):
            params.append(
                dict(
                    model=left_out or "full",
                    state=st,
                    startprob=p["startprob"][k],
                    **{f"trans_to_{t}": p["transmat"][k][j] for j, t in enumerate(STATE_NAMES)},
                    lambda_chest_per_bin=p["lambdas"][k][0],
                    lambda_head_per_bin=p["lambdas"][k][1],
                )
            )

    one(fit_model(train, cfg.phase), FULL_NAME, None)
    if leave_one_out:
        for n in names:
            one(fit_model({m: s for m, s in train.items() if m != n}, cfg.phase), LOO_NAME.format(name=n), n)
            LOG.info("leave-one-out model without %s", n)
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "index_models.csv", index=False, encoding="utf-8")
    pd.DataFrame(params).to_csv(out_dir / "source_hmm-params.csv", index=False, encoding="utf-8")
    return index


def run_hmm_fit(cfg: Config, out_dir: str | Path, overwrite: bool = False, leave_one_out: bool = True) -> pd.DataFrame:
    """``ata hmm fit``: fit on ``inputs.reference_dir`` and write a model folder (see :func:`fit_store`)."""
    from .common import prepare_step_dir, write_manifest
    from .pipeline import load_reference

    started = _dt.datetime.now()
    out = prepare_step_dir(out_dir, overwrite)
    ref = load_reference(cfg)
    index = fit_store({n: r.sig for n, r in ref.items()}, cfg, out, leave_one_out)
    write_manifest(
        out,
        "hmm-fit",
        config_to_dict(cfg),
        inputs=dict(reference=len(ref)),
        outputs=dict(models=len(index)),
        started=started,
        extra=dict(full_params_hash=str(index.params_hash.iloc[0]), training_animals=sorted(ref)),
    )
    return index


def describe(cfg: Config) -> pd.DataFrame:
    """``ata hmm check``: load and check the configured model; one row per state."""
    st = load_store(cfg)
    p = st.full
    rows = [
        dict(
            state=s,
            startprob=p.startprob[k],
            **{f"trans_to_{t}": p.transmat[k][j] for j, t in enumerate(STATE_NAMES)},
            lambda_chest_per_bin=p.lambdas[k][0],
            lambda_head_per_bin=p.lambdas[k][1],
        )
        for k, s in enumerate(STATE_NAMES)
    ]
    return pd.DataFrame(rows)
