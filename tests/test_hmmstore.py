"""Stored HMM: packaged reference model, configuration check, export and leave-one-out models."""

import json

import numpy as np
import pandas as pd
import pytest

from atatracker.cli import main
from atatracker.config import config_from_dict, save_config
from atatracker.hmmstore import (
    StoredModelMismatchError,
    describe,
    fit_store,
    load_store,
    params_hash,
    read_record,
    resolve_model_file,
)
from atatracker.phase import PhaseModel
from atatracker.pipeline import load_reference, run_phase


def _stored(config_dict, coord_dirs, tmp_path, loo=True):
    """Fit and export a model folder from the synthetic reference recordings; return a stored-model config."""
    cfg = config_from_dict(config_dict)
    folder = tmp_path / "model"
    folder.mkdir()
    fit_store({n: r.sig for n, r in load_reference(cfg).items()}, cfg, folder, leave_one_out=loo)
    d = json.loads(json.dumps(config_dict))
    d["phase"] = dict(refit=False, model_file=str(folder / "hmm_full.json"))
    return d, folder


def test_builtin_model_matches_default_configuration():
    cfg = config_from_dict({})
    st = load_store(cfg)
    rec = st.full_record
    assert resolve_model_file("builtin").name == "hmm_reference-17_full.json"
    assert rec["params_hash"] == params_hash(rec["params"]) and rec["kind"] == "full"
    assert st.full.train_names == [f"ref{i:02d}" for i in range(1, 18)] and st.loo_paths == {}
    assert rec["config"] == dict(
        SIG_CHEST="y",
        SIG_HEAD="max",
        THR=0.5,
        NORM_HI_PCT=99.9,
        LIK_MIN=0.9,
        DIST_CHEST=4,
        DIST_HEAD=24,
        BIN_S=1.0,
        FPS=30000 / 1001,
    )
    assert np.allclose(st.full.transmat.sum(1), 1) and st.full.transmat[3, 3] == 1.0
    assert len(describe(cfg)) == 4


@pytest.mark.parametrize(
    "change",
    [
        dict(signal=dict(peak_threshold=0.4)),
        dict(signal=dict(norm_percentile=99.5)),
        dict(signal=dict(likelihood_min=0.5)),
        dict(signal=dict(peak_distance_head_frames=30)),
        dict(signal=dict(fps=30)),
        dict(signal=dict(head_mode="xy")),
        dict(phase=dict(bin_s=2.0)),
    ],
)
def test_configuration_mismatch_stops(change):
    with pytest.raises(StoredModelMismatchError):
        load_store(config_from_dict(change))


def test_missing_and_tampered_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_store(config_from_dict(dict(phase=dict(model_file=str(tmp_path / "none.json")))))
    rec = json.loads(resolve_model_file("builtin").read_text())
    rec["params"]["lambdas"][1][0] += 1e-6
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="hash"):
        read_record(bad)


def test_stored_equals_refit(config_dict, coord_dirs, tmp_path):
    refit = run_phase(config_from_dict(config_dict), tmp_path / "refit", plots=False)
    stored_cfg, folder = _stored(config_dict, coord_dirs, tmp_path)
    assert len(list(folder.glob("hmm_loo_*.json"))) == 6
    stored = run_phase(config_from_dict(stored_cfg), tmp_path / "stored", plots=False)
    cols = ["animal", "hmm_chest_on_min", "head_on_min", "end_min", "hmm_post_on_min", "hmm_leave_one_out", "hmm_params_hash"]
    pd.testing.assert_frame_equal(refit[cols], stored[cols])
    m = json.loads((tmp_path / "stored" / "run_manifest.json").read_text())
    assert m["hmm"]["source"] == "stored" and m["hmm"]["config_check"] == "pass"
    assert json.loads((tmp_path / "stored" / "phase_model.json").read_text())["params_hash"] == m["hmm"]["params_hash"]


def test_reference_targets_use_stored_loo_models(config_dict, coord_dirs, tmp_path):
    config_dict["inputs"]["coord_sets"] = [dict(tag="ref", dir=str(coord_dirs["reference"]))]
    refit = run_phase(config_from_dict(config_dict), tmp_path / "refit", plots=False)
    stored_cfg, folder = _stored(config_dict, coord_dirs, tmp_path)
    stored = run_phase(config_from_dict(stored_cfg), tmp_path / "stored", plots=False)
    assert stored.hmm_leave_one_out.all() and len(stored) == 6
    for c in ("head_on_min", "end_min", "hmm_params_hash"):
        assert refit[c].tolist() == stored[c].tolist()
    # a missing leave-one-out model stops; without leave-one-out the full model is used
    next(folder.glob("hmm_loo_ref0*.json")).unlink()
    with pytest.raises(FileNotFoundError, match="leave-one-out"):
        run_phase(config_from_dict(stored_cfg), tmp_path / "stored2", plots=False)
    stored_cfg["phase"]["leave_one_out"] = False
    t = run_phase(config_from_dict(stored_cfg), tmp_path / "stored3", plots=False)
    assert not t.hmm_leave_one_out.any() and t.hmm_params_hash.nunique() == 1


def test_stored_model_needs_no_reference_dir(config_dict, coord_dirs, tmp_path):
    stored_cfg, _ = _stored(config_dict, coord_dirs, tmp_path, loo=False)
    stored_cfg["inputs"]["reference_dir"] = None
    t = run_phase(config_from_dict(stored_cfg), tmp_path / "p", plots=False)
    assert len(t) == 1 and not (tmp_path / "p" / "source_reference-quality-max.csv").exists()
    with pytest.raises(ValueError, match="reference_dir"):
        run_phase(config_from_dict(dict(stored_cfg, phase=dict(refit=True))), tmp_path / "q", plots=False)


def test_old_phase_model_json_is_readable(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        json.dumps(
            dict(
                startprob=[1, 0, 0, 0],
                transmat=np.eye(4).tolist(),
                lambdas=np.ones((4, 2)).tolist(),
                n_train=1,
                train_names=["a"],
            )
        )
    )
    assert PhaseModel.load(p).train_names == ["a"]


def test_cli_hmm_fit_check_and_refit_flag(config_dict, tmp_path):
    cfg_file = tmp_path / "c.yaml"
    save_config(config_from_dict(dict(config_dict, phase=dict(refit=False))), cfg_file)
    # default (stored builtin model) fails here: the synthetic data run at 30 frames/s
    assert main(["phase", "-c", str(cfg_file), "--out", str(tmp_path / "a"), "--no-plots"]) == 2
    assert main(["phase", "-c", str(cfg_file), "--out", str(tmp_path / "b"), "--no-plots", "--refit"]) == 0
    assert main(["hmm", "fit", "-c", str(cfg_file), "--out", str(tmp_path / "m")]) == 0
    idx = pd.read_csv(tmp_path / "m" / "index_models.csv")
    assert idx.kind.tolist() == ["full"] + ["loo"] * 6
    rec = read_record(tmp_path / "m" / "hmm_full.json")
    assert rec["config"]["FPS"] == 30.0 and rec["fit_settings"]["iter"] == 50 and rec["converged"] in (True, False)
    d = dict(config_dict, phase=dict(model_file=str(tmp_path / "m" / "hmm_full.json")))
    save_config(config_from_dict(d), cfg_file)
    assert main(["hmm", "check", "-c", str(cfg_file)]) == 0
    assert main(["phase", "-c", str(cfg_file), "--out", str(tmp_path / "c"), "--no-plots"]) == 0
    b = pd.read_csv(tmp_path / "b" / "source_phase-per-animal.csv")
    c = pd.read_csv(tmp_path / "c" / "source_phase-per-animal.csv")
    assert b.end_min.tolist() == c.end_min.tolist() and b.head_on_min.tolist() == c.head_on_min.tolist()
