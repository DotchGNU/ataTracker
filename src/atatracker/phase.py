"""Phase segmentation: pre -> chest -> head -> post (docs/method.md, sections 3-5).

The per-second chest and head peak counts of a recording are segmented by a four-state,
left-to-right Poisson hidden Markov model (``hmmlearn.hmm.PoissonHMM``). The model is
trained on a reference set of recordings and applied to every target recording. The head
onset is the first second of the head state; the end line is the last time the rolling
maximum of the head amplitude reaches a threshold.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DisplayConfig, EndLineConfig, PhaseConfig
from .signal import Signals

LOG = logging.getLogger("atatracker")
K = 4  # states: 0 pre, 1 chest, 2 head, 3 post
STATE_NAMES = ("pre", "chest", "head", "post")


def rule_onset(chest_peaks_s: np.ndarray, head_peaks_s: np.ndarray, bin_s: float = 10.0, max_s: float = 300.0) -> float:
    """Head onset by the bin rule: first ``bin_s`` bin where min-max scaled head count exceeds chest count.

    Used only to initialise the HMM state paths.

    Returns
    -------
    float
        Start of that bin (s), or NaN if the head count never exceeds the chest count.
    """
    bins = np.arange(0, max_s, bin_s)
    ch = np.append(np.histogram(chest_peaks_s, bins)[0], 0)
    hh = np.append(np.histogram(head_peaks_s, bins)[0], 0)

    def mm(a: np.ndarray) -> np.ndarray:
        return np.zeros(len(a)) if a.max() == a.min() else (a - a.min()) / (a.max() - a.min())

    cn, hn = mm(ch.astype(float)), mm(hh.astype(float))
    for i, b in enumerate(bins):
        if hn[i] > cn[i]:
            return float(b)
    return np.nan


def init_path(sig: Signals, cfg: PhaseConfig) -> np.ndarray:
    """Initial state path of one reference recording (sets the starting Poisson rates only).

    chest state from the first window of ``chest_onset_window_s`` with at least
    ``chest_onset_min_peaks`` chest peaks, head state from the rule onset (or half of the last
    peak time), post state after the last peak.
    """
    cc = sig.chest_counts
    n_bins = len(cc)
    cs, hs = sig.chest_peaks_s, sig.head_peaks_s
    lastp = max(cs.max() if len(cs) else 0, hs.max() if len(hs) else 0)
    win = int(round(cfg.chest_onset_window_s / cfg.bin_s))
    cum = np.convolve(cc, np.ones(win), "same")
    t0 = int(np.argmax(cum >= cfg.chest_onset_min_peaks)) if (cum >= cfg.chest_onset_min_peaks).any() else 0
    head_on = rule_onset(cs, hs, cfg.rule_bin_s, cfg.rule_max_s)
    th = int(head_on / cfg.bin_s) if not np.isnan(head_on) else int(lastp * 0.5 / cfg.bin_s)
    tl = int(lastp / cfg.bin_s)
    seg = np.zeros(n_bins, int)
    seg[t0:th] = 1
    seg[th : tl + 1] = 2
    seg[tl + 1 :] = 3
    return seg


@dataclass
class PhaseModel:
    """Fitted Poisson HMM parameters (JSON serialisable)."""

    startprob: np.ndarray
    transmat: np.ndarray
    lambdas: np.ndarray
    n_train: int
    train_names: list[str]

    def to_hmm(self) -> Any:
        """Return an ``hmmlearn`` ``PoissonHMM`` with these parameters."""
        from hmmlearn.hmm import PoissonHMM

        m = PoissonHMM(n_components=K, init_params="", params="")
        m.n_features = 2
        m.startprob_ = np.asarray(self.startprob, float)
        m.transmat_ = np.asarray(self.transmat, float)
        m.lambdas_ = np.asarray(self.lambdas, float)
        return m

    def predict(self, counts: np.ndarray) -> np.ndarray:
        """Viterbi state path of a ``(T, 2)`` count array."""
        return self.to_hmm().predict(counts)

    def save(self, path: str | Path) -> None:
        """Write the parameters as JSON (floats are written with full precision)."""
        doc = dict(
            states=list(STATE_NAMES),
            startprob=self.startprob.tolist(),
            transmat=self.transmat.tolist(),
            lambdas=self.lambdas.tolist(),
            n_train=self.n_train,
            train_names=self.train_names,
        )
        Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PhaseModel:
        """Read parameters written by :meth:`save`."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["startprob"]), np.array(d["transmat"]), np.array(d["lambdas"]), d["n_train"], d["train_names"])


def fit_model(train: dict[str, Signals], cfg: PhaseConfig) -> PhaseModel:
    """Fit the left-to-right Poisson HMM on reference recordings.

    Starting rates are the mean counts of each state of the initial paths (:func:`init_path`),
    floored at ``lambda_floor``; start probabilities ``start_prob``; transitions stay with
    ``stay_prob`` or move one state forward with ``move_prob``; the post state is absorbing. EM updates start
    probabilities, transitions and rates (``params="stl"``); zero transitions stay zero.

    Parameters
    ----------
    train : dict
        Animal name -> :class:`Signals` of the reference recordings.
    cfg : PhaseConfig
        HMM settings.
    """
    from hmmlearn.hmm import PoissonHMM

    names = list(train)
    xs = [train[n].counts for n in names]
    inits = [init_path(train[n], cfg) for n in names]
    lam = np.zeros((K, 2))
    for k in range(K):
        if any((p == k).any() for p in inits):
            m = np.concatenate([x[p == k] for x, p in zip(xs, inits, strict=True)])
        else:
            m = np.array([[cfg.empty_state_rate, cfg.empty_state_rate]])
        lam[k] = np.maximum(m.mean(0), cfg.lambda_floor)
    hmm = PoissonHMM(
        n_components=K, n_iter=cfg.hmm_n_iter, tol=cfg.hmm_tol, params="stl", init_params="", random_state=cfg.random_seed
    )
    hmm.startprob_ = np.array(cfg.start_prob, float)
    tm = np.zeros((K, K))
    for k in range(K - 1):
        tm[k, k], tm[k, k + 1] = cfg.stay_prob, cfg.move_prob
    tm[K - 1, K - 1] = 1.0
    hmm.transmat_ = tm
    hmm.lambdas_ = lam
    hmm.n_features = 2
    with contextlib.redirect_stdout(io.StringIO()):
        hmm.fit(np.vstack(xs), lengths=[len(x) for x in xs])
    LOG.debug("HMM fitted on %d recordings, converged=%s", len(xs), hmm.monitor_.converged)
    return PhaseModel(hmm.startprob_.copy(), hmm.transmat_.copy(), hmm.lambdas_.copy(), len(names), names)


def state_onsets(path: np.ndarray, bin_s: float = 1.0) -> list[float]:
    """Time (s) at which each of the four states is first entered (NaN if never)."""
    return [float(np.argmax(path == k)) * bin_s if (path == k).any() else np.nan for k in range(K)]


def end_line(head: np.ndarray, t0_s: float, threshold: float, window_s: float, limit_s: float, fps: float) -> float:
    """Last time (s) in ``[t0_s, limit_s]`` where the centred rolling max of ``head`` over ``window_s`` is >= ``threshold``.

    Returns NaN if there is no such frame.
    """
    w = int(window_s * fps)
    env = pd.Series(head).rolling(w, center=True, min_periods=1).max().values
    idx = np.where(env >= threshold)[0]
    idx = idx[(idx / fps >= t0_s) & (idx / fps <= limit_s)]
    return float(idx.max() / fps) if len(idx) else np.nan


@dataclass
class PhaseResult:
    """Phase boundaries of one recording (all in s)."""

    chest_on_s: float
    head_on_s: float
    post_on_s: float
    end_s: float


def segment(sig: Signals, model: PhaseModel, bin_s: float, end_cfg: EndLineConfig) -> PhaseResult:
    """Head onset and end line of one recording.

    The end line is searched from the head onset up to the HMM post onset (or the recording end)
    plus ``search_after_post_s``; if no frame reaches the threshold the post onset (or the
    recording end) is used. Without a head state both are NaN.
    """
    e = state_onsets(model.predict(sig.counts), bin_s)
    hon, post = e[2], e[3]
    if np.isnan(hon):
        return PhaseResult(e[1], np.nan, post, np.nan)
    stop = post if not np.isnan(post) else sig.duration_s
    end = end_line(sig.head, hon, end_cfg.threshold, end_cfg.window_s, stop + end_cfg.search_after_post_s, sig.fps)
    return PhaseResult(e[1], hon, post, end if not np.isnan(end) else stop)


def display_traces(
    chest: np.ndarray, head: np.ndarray, t_s: np.ndarray, head_on_s: float, end_s: float, cfg: DisplayConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Display-only traces: soft gate on the head trace before the head onset, flat floor after the end line.

    ``head * (floor + (1 - floor) / (1 + exp(-(t - head_on) / tau)))``; after ``end_s`` the chest
    and head traces are set to ``post_chest_level`` and ``post_head_level`` so the recording
    length stays visible. Never used for counting.
    """
    h1 = head * (cfg.gate_floor + (1 - cfg.gate_floor) / (1 + np.exp(-(t_s - head_on_s) / cfg.gate_tau_s)))
    c1 = chest.copy()
    post = t_s > end_s
    c1[post] = cfg.post_chest_level
    h1[post] = cfg.post_head_level
    return c1, h1
