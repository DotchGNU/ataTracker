import numpy as np
import pytest

from atatracker.common import load_coords
from atatracker.config import config_from_dict
from atatracker.phase import PhaseModel, end_line, fit_model, segment, state_onsets, weighted_end_line
from atatracker.pipeline import load_recording, load_reference
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


def test_end_line_default_is_weighted_rule(config_dict):
    cfg = config_from_dict(config_dict)
    e = cfg.end_line
    assert (e.chest_weight, e.head_weight, e.threshold, e.window_s, e.search_after_post_s) == (0.25, 0.75, 0.15, 12.0, 40.0)


def test_weighted_end_line_counts_chest_and_head():
    fps = 10.0
    chest, head = np.zeros(2000), np.zeros(2000)
    head[[200, 400]] = [0.5, 0.5]  # last head jerk at 40 s
    chest[1000] = 1.0  # chest jerk at 100 s: 0.25 * 1.0 = 0.25 >= 0.15
    # weighted rule: the late chest jerk is the last qualifying time (100 s + half of the 12-s window)
    e = weighted_end_line(chest, head, 10.0, 0.15, 12.0, 190.0, fps, 0.25, 0.75)
    assert abs(e - (100.0 + 6.0)) <= 0.1
    # head only: the chest jerk is ignored (40 s + half of the window)
    e0 = weighted_end_line(chest, head, 10.0, 0.15, 12.0, 190.0, fps, 0.0, 1.0)
    assert abs(e0 - (40.0 + 6.0)) <= 0.1
    # a chest-only jerk below weight * amplitude < threshold does not count
    assert np.isnan(weighted_end_line(chest * 0.5, np.zeros(2000), 10.0, 0.15, 12.0, 190.0, fps, 0.25, 0.75))


def test_weighted_end_line_search_interval():
    fps = 10.0
    head = np.zeros(1000)
    head[800] = 1.0  # 80 s
    assert np.isnan(weighted_end_line(np.zeros(1000), head, 10.0, 0.15, 2.0, 50.0, fps))  # after the limit
    assert not np.isnan(weighted_end_line(np.zeros(1000), head, 10.0, 0.15, 2.0, 90.0, fps))


def test_head_only_wrapper_equals_zero_chest_weight():
    rng = np.random.default_rng(1)
    head = np.minimum(rng.exponential(0.05, 3000), 1.0)
    chest = rng.random(3000)
    a = end_line(head, 5.0, 0.15, 5.0, 250.0, 10.0)
    b = weighted_end_line(chest, head, 5.0, 0.15, 5.0, 250.0, 10.0, 0.0, 1.0)
    assert a == b


def test_previous_rule_by_configuration(config_dict, coord_dirs):
    cfg = config_from_dict(dict(config_dict, end_line=dict(chest_weight=0.0, head_weight=1.0, window_s=5.0)))
    ref = load_reference(cfg)
    model = fit_model({n: r.sig for n, r in ref.items()}, cfg.phase)
    rec = load_recording("animalA", "model-1k", str(next(coord_dirs["targets"].glob("*.csv"))), cfg)
    res = segment(rec.sig, model, cfg.phase.bin_s, cfg.end_line)
    assert abs(res.end_s - (BOUNDS[-1][2] + 2.5)) <= 0.5


@pytest.mark.parametrize(
    "bad",
    [dict(chest_weight=-0.1), dict(chest_weight=0.0, head_weight=0.0), dict(window_s=0.0)],
)
def test_end_line_config_validation(config_dict, bad):
    with pytest.raises(ValueError):
        config_from_dict(dict(config_dict, end_line=bad))
