import numpy as np
import pytest

from atatracker.signal import amplitude, bin_counts, peak_times, step


def test_step_modes():
    x = np.array([0.0, 3.0, 3.0])
    y = np.array([0.0, 4.0, 2.0])
    assert step(x, y, "y").tolist() == [0, 4, 2]
    assert step(x, y, "x").tolist() == [0, 3, 0]
    assert step(x, y, "xy").tolist() == [0, 5, 2]
    assert step(x, y, "max").tolist() == [0, 4, 2]
    with pytest.raises(ValueError):
        step(x, y, "q")


def test_pca_mode_follows_main_axis():
    rng = np.random.default_rng(0)
    t = rng.normal(0, 1, 500).cumsum()
    x, y = t * 0.6, t * 0.8
    s = step(x, y, "pca")
    assert np.allclose(s, step(x, y, "xy"), atol=1e-9)


def test_amplitude_normalisation_and_mask():
    y = np.zeros(1000)
    y[100] = 10.0
    y[500] = 5.0
    lik = np.ones(1000)
    a = amplitude(None, y, lik, "y", 0.9, 99.9)
    assert a.min() == 0.0 and a.max() == 1.0
    assert np.all((a >= 0) & (a <= 1))
    lik[500] = 0.1
    b = amplitude(None, y, lik, "y", 0.9, 99.9)
    assert b[500] == 0.0 and b[501] == 0.0


def test_peak_counting():
    a = np.zeros(300)
    a[[30, 32, 90, 150, 151]] = [1.0, 0.9, 0.8, 0.6, 0.4]
    t = peak_times(a, threshold=0.5, distance_frames=4, fps=30.0)
    assert t.tolist() == [1.0, 3.0, 5.0]
    c = bin_counts(t, duration_s=10.0, bin_s=1.0)
    assert len(c) == 10 and c.sum() == 3 and c[1] == 1 and c[3] == 1 and c[5] == 1
