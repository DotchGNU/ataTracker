"""Frame-rate check with a stand-in ffprobe (no real videos)."""

import json
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest

from atatracker.cli import main
from atatracker.config import config_from_dict, save_config
from atatracker.fpscheck import FrameRateError, evaluate_video, probe_video, run_check_fps
from atatracker.pipeline import run_phase

FAKE_FFPROBE = """#!{python}
# Stand-in for ffprobe: prints the JSON stored next to the video as <video>.probe.json.
import json, sys
path = sys.argv[-1]
spec = json.load(open(path + ".probe.json"))
if "-count_packets" in sys.argv:
    print(json.dumps({{"streams": [{{"nb_read_packets": str(spec["counted"])}}]}}))
elif "-count_frames" in sys.argv:
    print(json.dumps({{"streams": [{{"nb_read_frames": str(spec["counted"])}}]}}))
else:
    print(json.dumps({{"streams": [spec["stream"]], "format": spec.get("format", {{}})}}))
"""

NTSC = "30000/1001"

# measured cases: original 29.97 fps; re-encoded copy whose container says 30000/1001 but holds ~24 fps
# (variable frame interval); a 59.94-fps recording; a 2997/100 copy (29.97 exactly, within tolerance);
# a file without nb_frames (frames counted)
VIDEOS = {
    "original": {
        "rec1.mp4": dict(
            stream=dict(r_frame_rate="30000/1001", avg_frame_rate="30000/1001", nb_frames="4680", duration="156.156")
        ),
        "rec2.MP4": dict(stream=dict(r_frame_rate="60000/1001", avg_frame_rate="60000/1001", nb_frames="120", duration="2.002")),
    },
    "dlc_input": {
        "a.mp4": dict(stream=dict(r_frame_rate="2997/100", avg_frame_rate="2997/100", nb_frames="10767", duration="359.259259")),
        "b.mp4": dict(
            stream=dict(r_frame_rate="30000/1001", avg_frame_rate="238320000/9928919", nb_frames="7944", duration="330.963967")
        ),
        "c.avi": dict(stream=dict(r_frame_rate="30000/1001", avg_frame_rate="0/0"), format=dict(duration="100.1"), counted=3000),
    },
}


def _setup(tmp_path: Path, videos=VIDEOS) -> dict:
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_text(FAKE_FFPROBE.format(python=sys.executable))
    ffprobe.chmod(ffprobe.stat().st_mode | stat.S_IEXEC)
    dirs = {}
    for role, files in videos.items():
        d = tmp_path / role
        d.mkdir()
        for name, spec in files.items():
            (d / name).write_bytes(b"")
            (d / (name + ".probe.json")).write_text(json.dumps(spec))
        dirs[role] = str(d)
    return dict(ffprobe=str(ffprobe), dirs=dirs)


def _cfg(env: dict, roles=("original", "dlc_input"), **fc):
    return config_from_dict(
        dict(
            crop=dict(ffprobe=env["ffprobe"]),
            fps_check=dict(
                original_dirs=[env["dirs"]["original"]] if "original" in roles else [],
                dlc_input_dirs=[env["dirs"]["dlc_input"]] if "dlc_input" in roles else [],
                **fc,
            ),
        )
    )


def test_evaluate_rates():
    fps = 30000 / 1001
    ok = evaluate_video(
        dict(r_frame_rate="30000/1001", avg_frame_rate="2997/100", nb_frames=300, duration_s=300 / fps), fps, 0.005
    )
    assert ok["status"] == "OK" and ok["dev_avg"] < 1e-5
    vfr = evaluate_video(
        dict(r_frame_rate="30000/1001", avg_frame_rate="30000/1001", nb_frames=7146, duration_s=297.7), fps, 0.005
    )
    assert vfr["status"] == "FAIL" and vfr["reason"].startswith("measured 24.00")
    missing = evaluate_video(
        dict(r_frame_rate="30000/1001", avg_frame_rate="0/0", nb_frames=float("nan"), duration_s=10.0), fps, 0.005
    )
    assert missing["status"] == "FAIL" and "not available" in missing["reason"]


def test_probe_counts_frames_when_missing(tmp_path):
    env = _setup(tmp_path)
    c = Path(env["dirs"]["dlc_input"]) / "c.avi"
    p = probe_video(env["ffprobe"], str(c))
    assert p["nb_frames"] == 3000 and p["frames_source"] == "counted_packets" and p["duration_source"] == "format"
    assert probe_video(env["ffprobe"], str(c), "decode")["frames_source"] == "counted_frames"


def test_check_stops_with_table(tmp_path):
    env = _setup(tmp_path)
    cfg = _cfg(env)
    out = tmp_path / "check"
    with pytest.raises(FrameRateError) as e:
        run_check_fps(cfg, out)
    t = pd.read_csv(out / "source_fps_check.csv")
    status = dict(zip(t.video, t.status, strict=True))
    assert status == {"rec1.mp4": "OK", "rec2.MP4": "FAIL", "a.mp4": "OK", "b.mp4": "FAIL", "c.avi": "OK"}
    assert set(t.role) == {"original", "dlc_input"}
    assert "rec2.MP4" in str(e.value) and "b.mp4" in str(e.value) and "rec1.mp4" not in str(e.value)
    m = json.loads((out / "run_manifest.json").read_text())
    assert m["fps_check"]["status"] == "fail" and m["fps_check"]["failed"] == 2 and m["fps_check"]["videos"] == 5


def test_check_passes_and_tolerance(tmp_path):
    env = _setup(tmp_path, {"original": VIDEOS["original"], "dlc_input": {"a.mp4": VIDEOS["dlc_input"]["a.mp4"]}})
    t = run_check_fps(_cfg(env, roles=("dlc_input",)), tmp_path / "ok")
    assert len(t) == 1 and (t.status == "OK").all()
    m = json.loads((tmp_path / "ok" / "run_manifest.json").read_text())
    assert m["fps_check"]["status"] == "pass"
    # a 59.94-fps configuration accepts the 59.94 recording and rejects the 29.97 one
    cfg = _cfg(env, roles=("original",))
    cfg.signal.fps = "60000/1001"
    with pytest.raises(FrameRateError):
        run_check_fps(cfg, tmp_path / "x")
    assert pd.read_csv(tmp_path / "x" / "source_fps_check.csv").set_index("video").status.to_dict() == {
        "rec1.mp4": "FAIL",
        "rec2.MP4": "OK",
    }


def test_check_needs_folders(tmp_path):
    with pytest.raises(ValueError):
        run_check_fps(config_from_dict({}), tmp_path / "none")


def _video_root(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    root.mkdir()
    return root


def test_phase_stops_before_analysis(config_dict, tmp_path):
    env = _setup(_video_root(tmp_path))
    config_dict["fps_check"] = dict(require=True, dlc_input_dirs=[env["dirs"]["dlc_input"]])
    config_dict["crop"] = dict(ffprobe=env["ffprobe"])
    out = tmp_path / "phase"
    with pytest.raises(FrameRateError):
        run_phase(config_from_dict(config_dict), out, plots=False)
    assert (out / "source_fps_check.csv").exists() and (out / "run_manifest_fps-check.json").exists()
    assert not (out / "run_manifest.json").exists() and not (out / "source_phase-per-animal.csv").exists()


def test_phase_records_passed_check(config_dict, tmp_path):
    env = _setup(
        tmp_path / "v" if (tmp_path / "v").mkdir() is None else tmp_path, {"dlc_input": {"a.mp4": VIDEOS["dlc_input"]["a.mp4"]}}
    )
    config_dict["fps_check"] = dict(require=True, dlc_input_dirs=[env["dirs"]["dlc_input"]])
    config_dict["crop"] = dict(ffprobe=env["ffprobe"])
    config_dict["signal"] = dict(fps=NTSC)
    out = tmp_path / "phase"
    run_phase(config_from_dict(config_dict), out, plots=False)
    m = json.loads((out / "run_manifest.json").read_text())
    assert m["fps_check"]["status"] == "pass" and m["fps_check"]["videos"] == 1


def test_cli_exit_code(tmp_path):
    env = _setup(tmp_path)
    cfg_file = tmp_path / "c.yaml"
    save_config(_cfg(env), cfg_file)
    assert main(["check-fps", "-c", str(cfg_file), "--out", str(tmp_path / "cli")]) == 2
    assert main(["check-fps", "-c", str(cfg_file), "--out", str(tmp_path / "cli2"), "--original", env["dirs"]["original"]]) == 2
