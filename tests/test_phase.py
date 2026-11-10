import numpy as np

from ataagent.common import load_coords
from ataagent.config import config_from_dict
from ataagent.phase import PhaseModel, end_line, fit_model, segment, state_onsets
from ataagent.pipeline import load_recording, load_reference
from conftest import BOUNDS, FPS


def test_hmm_recovers_known_boundaries(config_dict, coord_dirs):
    cfg = config_from_dict(config_dict)
    ref = load_reference(cfg)
    model = fit_model({n: r.sig for n, r in ref.items()}, cfg.phase)
    rec = load_recording("animalA", "model-1k", str(next(coord_dirs["targets"].glob("*.csv"))), cfg)
    res = segment(rec.sig, model, cfg.phase.bin_s, cfg.end_line)
    c0, h0, e0 = BOUNDS[-1]
    assert abs(res.chest_on_s - c0) <= 2
    assert abs(res.head_on_s - h0) <= 3
    # end line = last head jerk + half of the centred rolling-max window
    assert abs(res.end_s - (e0 + cfg.end_line.window_s / 2)) <= 0.5


def test_model_json_round_trip(config_dict, tmp_path):
    cfg = config_from_dict(config_dict)
    ref = load_reference(cfg)
    m = fit_model({n: r.sig for n, r in ref.items()}, cfg.phase)
    m.save(tmp_path / "m.json")
    m2 = PhaseModel.load(tmp_path / "m.json")
    x = next(iter(ref.values())).sig.counts
    assert np.array_equal(m.predict(x), m2.predict(x))


def test_fit_is_deterministic(config_dict):
    cfg = config_from_dict(config_dict)
    sig = {n: r.sig for n, r in load_reference(cfg).items()}
    a, b = fit_model(sig, cfg.phase), fit_model(sig, cfg.phase)
    assert np.array_equal(a.lambdas, b.lambdas) and np.array_equal(a.transmat, b.transmat)


def test_state_onsets():
    path = np.array([0, 0, 1, 1, 1, 2, 2, 3])
    assert state_onsets(path) == [0.0, 2.0, 5.0, 7.0]
    on = state_onsets(np.array([0, 0, 1, 1]))
    assert np.isnan(on[2]) and np.isnan(on[3])


def test_end_line_rule():
    fps = 10.0
    h = np.zeros(1000)
    h[[200, 400, 600]] = [0.5, 0.2, 0.14]
    # 0.14 at 60 s is below the threshold: the last reaching frame is 40 s + half the 5-s window
    e = end_line(h, t0_s=10.0, threshold=0.15, window_s=5.0, limit_s=100.0, fps=fps)
    assert abs(e - (40.0 + 2.5)) < 1e-9
    assert np.isnan(end_line(h, t0_s=50.0, threshold=0.15, window_s=5.0, limit_s=100.0, fps=fps))


def test_load_coords_selects_bodyparts(coord_dirs):
    f = next(coord_dirs["targets"].glob("*.csv"))
    d = load_coords(f)
    assert list(d.columns) == ["cx", "cy", "cl", "hx", "hy", "hl"]
    assert len(d) == int(300 * FPS)
