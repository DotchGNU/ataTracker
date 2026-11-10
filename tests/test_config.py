from pathlib import Path

import pytest
import yaml

from atatracker.config import config_from_dict, config_to_dict, load_config, parse_rate, save_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_dataclasses():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as fh:
        assert yaml.safe_load(fh) == config_to_dict(load_config(None))


def test_round_trip(tmp_path):
    cfg = config_from_dict(
        {"signal": {"head_mode": "xy"}, "inputs": {"coord_sets": [{"tag": "a", "dir": "/x"}, {"tag": "b", "dir": "/y"}]}}
    )
    save_config(cfg, tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert config_to_dict(back) == config_to_dict(cfg)
    assert back.inputs.coord_sets[1].tag == "b"


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        config_from_dict({"signal": {"no_such_key": 1}})


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"signal": {"chest_mode": "z"}})


def test_frame_rate_ratio():
    assert parse_rate("30000/1001") == 30000 / 1001
    assert parse_rate(29.97) == 29.97
    assert config_from_dict({}).signal.frame_rate == 30000 / 1001
