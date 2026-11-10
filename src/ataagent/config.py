"""Typed configuration.

Every tunable value of the package lives in one of the dataclasses below and in
``configs/default.yaml``. ``tests/test_config.py`` fails if the two disagree.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

MOTION_MODES = ("y", "x", "xy", "max", "pca")


def parse_rate(value: float | int | str) -> float:
    """Parse a frame rate given as a number or as a ratio string such as ``"30000/1001"``.

    Parameters
    ----------
    value : float, int or str
        Frame rate in frames per second.

    Returns
    -------
    float
        Frame rate (frames/s). A ratio is evaluated as ``numerator / denominator``.
    """
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(Fraction(str(value))) if isinstance(value, str) else float(value)


@dataclass
class SignalConfig:
    """Per-frame motion signal and peak calling (see docs/method.md, section 2)."""

    fps: str | float = "30000/1001"
    chest_bodypart: str = "CHEST"
    head_bodypart: str = "HEAD"
    chest_mode: str = "y"
    head_mode: str = "max"
    likelihood_min: float = 0.9
    norm_percentile: float = 99.9
    peak_threshold: float = 0.5
    peak_distance_chest_frames: int = 4
    peak_distance_head_frames: int = 24

    @property
    def frame_rate(self) -> float:
        """Frame rate in frames/s."""
        return parse_rate(self.fps)


@dataclass
class PhaseConfig:
    """Four-state left-to-right Poisson HMM (pre -> chest -> head -> post)."""

    bin_s: float = 1.0
    hmm_n_iter: int = 50
    hmm_tol: float = 1e-4
    random_seed: int = 0
    start_prob: list[float] = field(default_factory=lambda: [0.6, 0.4, 0.0, 0.0])
    stay_prob: float = 0.99
    move_prob: float = 0.01
    lambda_floor: float = 0.01
    empty_state_rate: float = 0.1
    chest_onset_min_peaks: int = 5
    chest_onset_window_s: float = 5.0
    rule_bin_s: float = 10.0
    rule_max_s: float = 300.0
    leave_one_out: bool = True


@dataclass
class EndLineConfig:
    """End line: last time the rolling max of the head amplitude reaches the threshold."""

    threshold: float = 0.15
    window_s: float = 5.0
    search_after_post_s: float = 40.0


@dataclass
class DisplayConfig:
    """Display-only correction of the twitching traces (never used for counting)."""

    gate_tau_s: float = 3.0
    gate_floor: float = 0.0
    post_chest_level: float = 0.03
    post_head_level: float = 0.015


@dataclass
class QualityConfig:
    """Label-free tracking quality metrics."""

    body_length_z: float = 4.0
    head_jump_px: float = 15.0
    likelihood_low: float = 0.5
    end_vs_manual_flag_s: float = 60.0


@dataclass
class CoordSet:
    """One set of DeepLabCut coordinate files produced by one checkpoint."""

    tag: str = "default"
    dir: str = "/path/to/coords"


@dataclass
class InputConfig:
    """Where the coordinate tables are."""

    reference_dir: str = "/path/to/reference_coords"
    coord_sets: list[CoordSet] = field(default_factory=lambda: [CoordSet()])
    coord_glob: str = "*DLC_*.csv"
    reference_glob: str | None = None
    name_separator: str = "DLC_"
    checkpoint_choice: str | None = None
    groups_csv: str | None = None
    manual_csv: str | None = None
    video_dir: str | None = None
    video_suffix: str = ".mp4"
    screen_animals: list[str] = field(default_factory=list)


@dataclass
class FigureConfig:
    """Figure style. Every plot is written as PNG + SVG + source CSV."""

    dpi: int = 300
    font_family: str = "Arial"
    font_files: list[str] = field(default_factory=list)
    xmax_min: float = 6.2
    chest_color: str = "#002060"
    head_color: str = "#C00000"
    onset_marker_color: str = "#1f2a6b"
    end_marker_color: str = "#C00000"
    group_order: list[str] = field(default_factory=list)
    jitter_seed: int = 0
    stack_xmax_min: float = 6.0
    stack_plot_w_in: float = 4.8
    stack_row_h_in: float = 0.24
    stack_label_w_in: float = 0.75
    stack_group_gap_in: float = 0.55
    stack_svg_bins: int = 0
    stack_head_gate: bool = True
    xlsx_bar_colors: list[str] = field(default_factory=lambda: ["7D9AB4", "E57D7D"])
    xlsx_plot_w_in: float = 4.8
    xlsx_jitter_span: float = 0.34


@dataclass
class CropConfig:
    """Raw video -> fixed-size crop (ffmpeg)."""

    batch: str = "batch"
    raw_dir: str = "/path/to/raw_videos"
    reference_dir: str | None = None
    out_dir: str = "/path/to/cropped_videos"
    tool_dir: str = "/path/to/crop_tool"
    raw_patterns: list[str] = field(default_factory=lambda: ["*.MP4", "*.mp4"])
    exclude: list[str] = field(default_factory=list)
    out_w: int = 480
    out_h: int = 272
    crop_w: int = 925
    crf: int = 18
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    server_port: int = 8765
    url_path_map: dict[str, str] = field(default_factory=dict)
    detect_fps: float = 3.0
    detect_w: int = 480
    detect_h: int = 270
    reference_frame_s: float = 20.0
    suggest_step_s: float = 20.0
    suggest_reference_times_s: list[float] = field(default_factory=lambda: [10.0, 30.0, 60.0])
    suggest_scales_w: list[int] = field(default_factory=lambda: [90, 120, 150, 180, 210, 240, 270])
    suggest_top: int = 3


@dataclass
class LabelConfig:
    """Frame selection and pre-labelling for the browser labeller."""

    n_uniform: int = 15
    n_outlier: int = 15
    min_gap_s: float = 3.0
    edge_s: float = 5.0
    scorer: str = "scorer"
    convention: str = "CHEST = centre of the back, HEAD = nose tip. Decide once and keep it for every video."
    store_key: str = "ata-labels"


@dataclass
class FinetuneConfig:
    """Continue training from an existing snapshot."""

    init_weights: str = "/path/to/dlc-models/iteration-0/train/snapshot-350000"
    new_iteration: int = 1
    lr_schedule: list[list[float]] = field(default_factory=lambda: [[0.001, 10000], [0.0005, 20000], [0.0001, 30000]])
    max_iters: int = 30000
    save_every: int = 5000
    display_every: int = 500
    keep_snapshots: int = 7
    net_type: str = "resnet_50"
    augmenter: str = "default"
    train_fraction: float | None = None
    pseudo_likelihood_min: float = 0.9


@dataclass
class SnapshotFamily:
    """Snapshots of one DeepLabCut project that are screened."""

    family: str = "model"
    project_config: str = "/path/to/project/config.yaml"
    iteration: int = 0
    steps: list[int] = field(default_factory=list)


@dataclass
class DLCConfig:
    """DeepLabCut inference and training (runs in the DLC environment)."""

    project_config: str = "/path/to/project/config.yaml"
    shuffle: int = 1
    gputouse: int | None = None
    save_as_csv: bool = True
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".MP4", ".avi", ".AVI"])
    python: str = "python"
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    screen: list[SnapshotFamily] = field(default_factory=list)


@dataclass
class GridConfig:
    """Optional settings search (docs/method.md, section 6)."""

    modes: list[str] = field(default_factory=lambda: list(MOTION_MODES))
    peak_thresholds: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5])
    norm_percentiles: list[float] = field(default_factory=lambda: [99.0, 99.5, 99.9])
    likelihood_mins: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9])
    end_thresholds: list[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.30])
    end_windows_s: list[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])
    min_head_peaks: int = 10
    onset_agreement_min: float = 0.3
    compare_sets: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Top-level configuration."""

    output_dir: str = "/path/to/output"
    signal: SignalConfig = field(default_factory=SignalConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    end_line: EndLineConfig = field(default_factory=EndLineConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    dlc: DLCConfig = field(default_factory=DLCConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` on values the algorithms cannot use."""
        for name in ("chest_mode", "head_mode"):
            if getattr(self.signal, name) not in MOTION_MODES:
                raise ValueError(f"signal.{name} must be one of {MOTION_MODES}")
        if len(self.phase.start_prob) != 4:
            raise ValueError("phase.start_prob must have 4 entries (pre, chest, head, post)")
        if not self.inputs.coord_sets:
            raise ValueError("inputs.coord_sets must list at least one coordinate set")


def _build(cls: type, data: Any) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _build(hint, value)
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            kwargs[key] = [_build(args[0], v) for v in (value or [])]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any] | None) -> Config:
    """Build a :class:`Config` from a (possibly partial) mapping; missing keys keep their defaults."""
    cfg = _build(Config, data or {})
    cfg.validate()
    return cfg


def load_config(path: str | Path | None) -> Config:
    """Load a YAML configuration file; ``None`` returns the defaults."""
    if path is None:
        return config_from_dict({})
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Return the configuration as plain Python types (for manifests and round trips)."""
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the configuration as YAML."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
