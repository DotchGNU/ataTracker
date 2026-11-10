"""Command-line interface: ``ata <step> ...`` (run ``ata <step> -h`` for the options of a step)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .common import OutputExistsError, prepare_step_dir, setup_logging, write_manifest
from .config import config_to_dict, load_config, save_config

LOG = logging.getLogger("ataagent")


def _cfg(a: argparse.Namespace):
    return load_config(a.config)


def cmd_config(a: argparse.Namespace) -> int:
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise OutputExistsError(f"{out} exists; pass --overwrite")
    save_config(load_config(None), out)
    LOG.info("default configuration written to %s", out)
    return 0


def cmd_crop(a: argparse.Namespace) -> int:
    from .crop.tool import apply_crops, build_tool

    cfg = _cfg(a)
    started = _dt.datetime.now()
    if a.action == "tool":
        p = build_tool(cfg.crop, with_suggestions=not a.no_suggest)
        write_manifest(p.parent, "crop-tool", config_to_dict(cfg), inputs={}, outputs=dict(pages=1), started=started)
    elif a.action == "serve":
        from .crop.server import serve

        serve(cfg.crop, a.save_root or cfg.crop.tool_dir)
    else:
        cmds = apply_crops(cfg.crop, a.crops_json, dry_run=a.dry_run, overwrite=a.overwrite)
        if not a.dry_run:
            write_manifest(
                cfg.crop.out_dir,
                "crop-apply",
                config_to_dict(cfg),
                inputs=dict(crops=len(cmds)),
                outputs=dict(videos=len(cmds)),
                started=started,
                extra=dict(ffmpeg_commands=cmds),
                filename="run_manifest_crop-apply.json",
            )
    return 0


def cmd_label(a: argparse.Namespace) -> int:
    from .label.prepare import prepare

    cfg = _cfg(a)
    started = _dt.datetime.now()
    if not a.html_only:
        prepare_step_dir(a.out, a.overwrite)
    p = prepare(cfg, a.videos, a.coords, a.out, html_only=a.html_only)
    n = len(json.loads((p.parent / "prelabels.json").read_text(encoding="utf-8")))
    write_manifest(
        p.parent,
        "label",
        config_to_dict(cfg),
        inputs={},
        outputs=dict(frames=n),
        started=started,
        filename="run_manifest_label-page.json" if a.html_only else "run_manifest.json",
    )
    return 0


def cmd_dlc(a: argparse.Namespace) -> int:
    from .dlc import run as R

    cfg = _cfg(a)
    if a.action == "analyze":
        R.run(R.analyze_cmd(cfg, a.videos, a.dest), a.dry_run)
    elif a.action == "screen":
        R.screen_inference(cfg, a.videos, a.dest, a.dry_run)
    elif a.action == "train":
        R.run(R.train_cmd(cfg, a.project), a.dry_run)
    elif a.action == "pseudo-labels":
        R.run(
            R.runner_cmd(
                cfg,
                "pseudo-labels",
                "--config",
                a.model_config,
                "--labeled-data",
                a.labeled_data,
                "--scratch",
                a.scratch,
                "--out",
                a.out,
            ),
            a.dry_run,
        )
    else:
        args = [
            "build-project",
            "--base-config",
            a.base_config,
            "--labels",
            a.labels,
            "--dest",
            a.dest,
            "--video-dir",
            a.video_dir,
            "--scorer",
            cfg.label.scorer,
            "--base-head",
            a.base_head,
            "--pseudo-lik",
            cfg.dlc.finetune.pseudo_likelihood_min,
            "--width",
            cfg.crop.out_w,
            "--height",
            cfg.crop.out_h,
        ]
        if a.pseudo_csv:
            args += ["--pseudo-csv", a.pseudo_csv]
        if a.holdout:
            args += ["--holdout", a.holdout]
        if a.only_videos:
            args += ["--only-videos", a.only_videos]
        if a.new_only:
            args += ["--new-only"]
        if cfg.dlc.finetune.train_fraction:
            args += ["--train-fraction", cfg.dlc.finetune.train_fraction]
        R.run(R.runner_cmd(cfg, *args), a.dry_run)
    return 0


def cmd_phase(a: argparse.Namespace) -> int:
    from .pipeline import run_phase

    t = run_phase(_cfg(a), a.out or Path(_cfg(a).output_dir) / "phase", overwrite=a.overwrite, plots=not a.no_plots)
    LOG.info("%d animals, %d without a head phase", len(t), int(t.head_on_min.isna().sum()))
    return 0


def cmd_screen(a: argparse.Namespace) -> int:
    from .pipeline import run_screen

    t = run_screen(_cfg(a), a.out or Path(_cfg(a).output_dir) / "checkpoint-screen", overwrite=a.overwrite, plots=not a.no_plots)
    LOG.info("%d animal x checkpoint rows", len(t))
    return 0


def cmd_review(a: argparse.Namespace) -> int:
    from .review import pages

    started = _dt.datetime.now()
    cfg = _cfg(a)
    if a.action == "picker":
        p = pages.build_picker(a.dir)
    elif a.action == "choices":
        out = Path(a.out)
        if out.exists() and not a.overwrite:
            raise OutputExistsError(f"{out} exists; pass --overwrite")
        t = pages.choices_to_csv(a.json, out, rule=a.rule)
        LOG.info("%d animals with a chosen checkpoint -> %s", len(t), out.name)
        return 0
    elif a.action == "review":
        p = pages.build_review(a.dir, a.set_name)
    else:
        p = pages.build_viewer(cfg, a.dir, a.videos or cfg.inputs.video_dir, frame_w=cfg.crop.out_w)
    write_manifest(
        p.parent,
        f"review-{a.action}",
        config_to_dict(cfg),
        inputs={},
        outputs=dict(pages=1),
        started=started,
        filename=f"run_manifest_review-{a.action}.json",
    )
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    cfg = _cfg(a)
    if a.action == "models":
        from .evaluate.compare import compare_models

        compare_models(cfg, a.out or Path(cfg.output_dir) / "model-compare", labels_json=a.labels, overwrite=a.overwrite)
    else:
        from .evaluate.grid import run_grid

        r = run_grid(cfg, a.out or Path(cfg.output_dir) / "settings-grid", overwrite=a.overwrite)
        LOG.info("best combination: %s", r.iloc[0].to_dict() if len(r) else None)
    return 0


def cmd_figures(a: argparse.Namespace) -> int:
    import pandas as pd

    from .figures.style import use_style

    cfg = _cfg(a)
    started = _dt.datetime.now()
    phase_dir = Path(a.phase_dir)
    t = pd.read_csv(phase_dir / "source_phase-per-animal.csv")
    if a.labels:
        lab = pd.read_csv(a.labels)
        t = t.drop(columns=[c for c in ("label",) if c in t]).merge(lab[["animal", "label"]], on="animal", how="left")
    t = t[t.group.notna() & (t.group.astype(str) != "")]
    out = prepare_step_dir(a.out or phase_dir / "figures", a.overwrite)
    use_style(cfg.figures.font_family, cfg.figures.font_files)
    n = 0
    if a.kind in ("stack", "all"):
        from .figures.stack import plot_stack

        plot_stack(t, phase_dir / "twitching", out, "twitching-stack", cfg)
        n += 1
    if a.kind in ("bars", "all"):
        from .figures.bars import plot_phase_bars

        plot_phase_bars(t, out, "bar_phase-durations", cfg)
        n += 1
    if a.kind in ("xlsx", "all"):
        from .figures.xlsx import build_workbook

        build_workbook(t, out / "phase-durations.xlsx", cfg)
        n += 1
    write_manifest(out, "figures", config_to_dict(cfg), inputs=dict(animals=len(t)), outputs=dict(figures=n), started=started)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ata", description="Asphyxia twitching analysis from DeepLabCut pose tracks.")
    ap.add_argument("--version", action="version", version=f"ataagent {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="step", required=True)

    def add(name: str, func, help_: str, config: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        if config:
            p.add_argument("-c", "--config", required=True, help="YAML configuration (see configs/default.yaml)")
        p.add_argument("--overwrite", action="store_true", help="replace existing results")
        p.set_defaults(func=func)
        return p

    p = add("config", cmd_config, "write the default configuration", config=False)
    p.add_argument("out")

    p = add("crop", cmd_crop, "raw videos -> fixed-size crops (browser tool, helper server, ffmpeg)")
    p.add_argument("action", choices=["tool", "serve", "apply"])
    p.add_argument("--crops-json", help="crop list exported by the tool (apply)")
    p.add_argument("--save-root", help="folder below which the helper may save JSON (serve; default crop.tool_dir)")
    p.add_argument("--no-suggest", action="store_true", help="skip crop suggestions (tool)")
    p.add_argument("--dry-run", action="store_true")

    p = add("label", cmd_label, "pick frames, pre-label them from a model and build the browser labeller")
    p.add_argument("--videos", required=True)
    p.add_argument("--coords", required=True, help="DeepLabCut coordinates of those videos (pre-labels)")
    p.add_argument("--out", required=True)
    p.add_argument("--html-only", action="store_true", help="rewrite the page from an existing prelabels.json")

    p = add("dlc", cmd_dlc, "DeepLabCut inference, checkpoint screen inference, fine-tune project, training (DLC environment)")
    p.add_argument("action", choices=["analyze", "screen", "pseudo-labels", "build-project", "train"])
    p.add_argument("--videos", help="video folder (analyze, screen)")
    p.add_argument("--dest", help="output folder (analyze, screen) or new project folder (build-project)")
    p.add_argument("--project", help="project config.yaml to train (train)")
    p.add_argument("--model-config", help="model whose predictions become pseudo labels (pseudo-labels)")
    p.add_argument("--labeled-data", help="labeled-data folder of the base project (pseudo-labels)")
    p.add_argument("--scratch", help="scratch folder for frame copies (pseudo-labels)")
    p.add_argument("--out", help="output csv (pseudo-labels)")
    p.add_argument("--base-config", help="base project config.yaml (build-project)")
    p.add_argument("--labels", help="labeller export json (build-project)")
    p.add_argument("--video-dir", help="folder of the labelled videos (build-project)")
    p.add_argument("--base-head", choices=["keep", "nan", "pseudo"], default="nan")
    p.add_argument("--pseudo-csv")
    p.add_argument("--holdout", default="", help="comma-separated videos left out of training")
    p.add_argument("--only-videos", default="", help="comma-separated videos to use (per-animal fine-tune)")
    p.add_argument("--new-only", action="store_true", help="train on the new labels only")
    p.add_argument("--dry-run", action="store_true")

    p = add("phase", cmd_phase, "phase segmentation of every target animal (one checkpoint per animal)")
    p.add_argument("--out")
    p.add_argument("--no-plots", action="store_true")

    p = add("screen", cmd_screen, "checkpoint screen: onset / end / metrics / plot for every animal x checkpoint")
    p.add_argument("--out")
    p.add_argument("--no-plots", action="store_true")

    p = add("review", cmd_review, "review pages (checkpoint picker, twitching review, video viewer)")
    p.add_argument("action", choices=["picker", "choices", "review", "viewer"])
    p.add_argument("--dir", help="ata screen output (picker) or ata phase output (review, viewer)")
    p.add_argument("--json", help="picker export (choices)")
    p.add_argument("--out", help="checkpoint choice csv to write (choices)")
    p.add_argument("--rule", choices=["last", "first"], default="last", help="which tick wins when several are ticked (choices)")
    p.add_argument("--set-name", default="set", help="name shown on the review page (review)")
    p.add_argument("--videos", help="video folder (viewer; default inputs.video_dir)")

    p = add("evaluate", cmd_evaluate, "compare checkpoints (models) or search analysis settings (grid)")
    p.add_argument("action", choices=["models", "grid"])
    p.add_argument("--labels", help="labeller export json (models)")
    p.add_argument("--out")

    p = add("figures", cmd_figures, "stacked twitching plot, phase-duration bars and native-chart workbook from ata phase output")
    p.add_argument("kind", choices=["stack", "bars", "xlsx", "all"])
    p.add_argument("--phase-dir", required=True)
    p.add_argument("--labels", help="csv animal,label with the row labels to show")
    p.add_argument("--out")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``ata`` command."""
    a = parser().parse_args(argv)
    setup_logging(a.verbose)
    try:
        return a.func(a)
    except (OutputExistsError, FileNotFoundError, KeyError, ValueError) as e:
        LOG.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
