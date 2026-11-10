"""DeepLabCut steps, executed inside the DeepLabCut environment (Python 3.7, DeepLabCut 2.2.3).

This file must stay importable without the rest of the package and compatible with Python 3.7:
it imports only the standard library, numpy, pandas, PyYAML and (lazily) deeplabcut. The
analysis package calls it as ``<dlc.python> runner.py <command> ...`` (``ata dlc ...``); it can
also be run directly.

Commands
--------
analyze        ``deeplabcut.analyze_videos`` on every video of a folder.
pseudo-labels  run a model on the labelled frames of a project (``analyze_time_lapse_frames``).
build-project  new fine-tune project = labelled frames of a base project + labeller export.
train          ``create_training_dataset`` + ``train_network`` starting from given weights.
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

LOG = logging.getLogger("atatracker.dlc")
PARTS = ["CHEST", "HEAD"]


def videos_in(folder, extensions):
    """Sorted video files of ``folder`` with one of ``extensions``."""
    return sorted(p for e in extensions for p in glob.glob(os.path.join(folder, "*" + e)))


def cmd_analyze(a):
    vids = videos_in(a.video_dir, a.ext.split(","))
    if not vids:
        raise SystemExit("no video in " + a.video_dir)
    LOG.info("config %s; %d videos -> %s", a.config, len(vids), a.dest_dir)
    if a.dry_run:
        for v in vids:
            LOG.info("  %s", os.path.basename(v))
        return
    if not os.path.isdir(a.dest_dir):
        os.makedirs(a.dest_dir)
    import deeplabcut

    LOG.info("deeplabcut %s", deeplabcut.__version__)
    deeplabcut.analyze_videos(
        a.config, vids, shuffle=a.shuffle, save_as_csv=True, destfolder=a.dest_dir, gputouse=None if a.gpu < 0 else a.gpu
    )


def cmd_pseudo(a):
    """Predictions of ``--config`` on the png frames of every ``labeled-data/<video>`` folder.

    The frames are copied to ``--scratch`` first, so the project is untouched.
    """
    import deeplabcut

    rows = []
    for v in sorted(os.listdir(a.labeled_data)):
        src = os.path.join(a.labeled_data, v)
        if not glob.glob(os.path.join(src, "CollectedData_*.h5")):
            continue
        d = os.path.join(a.scratch, v)
        if not os.path.isdir(d):
            os.makedirs(d)
        for f in glob.glob(os.path.join(src, "*.png")):
            shutil.copy2(f, d)
        deeplabcut.analyze_time_lapse_frames(a.config, d, frametype=".png", save_as_csv=True)
        c = glob.glob(os.path.join(d, "*DLC_*.csv"))[0]
        x = pd.read_csv(c, header=2)
        x.columns = ["image", "cx", "cy", "cl", "hx", "hy", "hl"]
        x.insert(0, "video", v)
        rows.append(x)
        LOG.info("%s: %d frames", v, len(x))
    pd.concat(rows).to_csv(a.out, index=False)


def labels_frame(frames, scorer):
    """DeepLabCut ``CollectedData`` table of labeller frames (CHEST, HEAD x/y)."""
    cols = pd.MultiIndex.from_product([[scorer], PARTS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    rows = [list(f["chest"]) + list(f["head"]) for f in frames]
    idx = ["labeled-data\\{}\\{}".format(f["video"], f["image"]) for f in frames]
    return pd.DataFrame(rows, index=idx, columns=cols)


def cmd_build(a):
    """New project in ``--dest``: base project's labelled data (optionally with HEAD masked or replaced) + new labels."""
    if os.path.exists(a.dest):
        raise SystemExit(a.dest + " exists - refusing to overwrite")
    base = os.path.dirname(os.path.abspath(a.base_config))
    with open(a.labels) as fh:
        frames = [f for f in json.load(fh)["frames"] if f["confirmed"]]
    holdout = {v for v in a.holdout.split(",") if v}
    only = {v for v in a.only_videos.split(",") if v}
    frames = [f for f in frames if f["video"] not in holdout and (not only or f["video"] in only)]
    vids = sorted({f["video"] for f in frames})
    old = [] if a.new_only else sorted(os.listdir(os.path.join(base, "labeled-data")))
    clash = set(old) & set(vids)
    if clash:
        raise SystemExit("videos both in the base project and in the new labels: " + ", ".join(sorted(clash)))
    LOG.info("%d base videos, %d new frames in %d videos, base HEAD = %s", len(old), len(frames), len(vids), a.base_head)
    if a.dry_run:
        return
    os.makedirs(a.dest)
    pseudo = pd.read_csv(a.pseudo_csv) if a.base_head == "pseudo" else None
    if old:
        shutil.copytree(os.path.join(base, "labeled-data"), os.path.join(a.dest, "labeled-data"))
    for v in old:
        for ext in ("csv", "h5"):
            p = os.path.join(a.dest, "labeled-data", v, "CollectedData_{}.{}".format(a.scorer, ext))
            if not os.path.exists(p):
                continue
            df = pd.read_hdf(p) if ext == "h5" else pd.read_csv(p, header=[0, 1, 2], index_col=0)
            if a.base_head in ("nan", "pseudo"):
                df.loc[:, (slice(None), "HEAD", slice(None))] = np.nan
            if a.base_head == "pseudo":
                ps = pseudo[pseudo.video == v].set_index("image")
                names = [os.path.basename(str(i).replace("\\", "/")) for i in df.index]
                for k, nm in enumerate(names):
                    if nm in ps.index and ps.loc[nm, "hl"] >= a.pseudo_lik:
                        df.iloc[k, df.columns.get_loc((a.scorer, "HEAD", "x"))] = ps.loc[nm, "hx"]
                        df.iloc[k, df.columns.get_loc((a.scorer, "HEAD", "y"))] = ps.loc[nm, "hy"]
            if ext == "h5":
                df.to_hdf(p, key="df_with_missing", mode="w")
            else:
                df.to_csv(p)
    frame_dir = os.path.join(os.path.dirname(os.path.abspath(a.labels)), "frames")
    for v in vids:
        d = os.path.join(a.dest, "labeled-data", v)
        os.makedirs(d)
        vf = [f for f in frames if f["video"] == v]
        for f in vf:
            shutil.copy2(os.path.join(frame_dir, v, f["image"]), os.path.join(d, f["image"]))
        df = labels_frame(vf, a.scorer)
        df.to_csv(os.path.join(d, "CollectedData_{}.csv".format(a.scorer)))
        df.to_hdf(os.path.join(d, "CollectedData_{}.h5".format(a.scorer)), key="df_with_missing", mode="w")
    with open(a.base_config) as fh:
        c = yaml.safe_load(fh)
    c["project_path"] = os.path.abspath(a.dest)
    c["iteration"] = 0
    c["scorer"] = a.scorer
    if a.new_only:
        c["video_sets"] = {}
    if a.train_fraction:
        c["TrainingFraction"] = [a.train_fraction]
    for v in vids:
        c["video_sets"][os.path.join(a.video_dir, v + ".mp4")] = {"crop": "0, {}, 0, {}".format(a.width, a.height)}
    with open(os.path.join(a.dest, "config.yaml"), "w") as fh:
        yaml.safe_dump(c, fh)
    LOG.info("project built: %s", a.dest)


def cmd_train(a):
    """Create the training set of ``--iteration`` and train from ``--init-weights`` (steps relative to that snapshot)."""
    with open(a.config) as fh:
        c = yaml.safe_load(fh)
    sched = json.loads(a.lr_schedule)
    LOG.info(
        "iteration %s -> %s; init %s; schedule %s; %d steps", c["iteration"], a.iteration, a.init_weights, sched, a.max_iters
    )
    if a.dry_run:
        return
    if c["iteration"] != a.iteration:
        shutil.copy2(a.config, a.config + ".before-iteration-{}".format(a.iteration))
        c["iteration"] = a.iteration
        with open(a.config, "w") as fh:
            yaml.safe_dump(c, fh)
    import deeplabcut

    deeplabcut.create_training_dataset(
        a.config, num_shuffles=1, Shuffles=[a.shuffle], net_type=a.net_type, augmenter_type=a.augmenter
    )
    proj = os.path.dirname(os.path.abspath(a.config))
    tr = sorted(glob.glob(os.path.join(proj, "dlc-models", "iteration-{}".format(a.iteration), "*", "train")))[0]
    pc_path = os.path.join(tr, "pose_cfg.yaml")
    with open(pc_path) as fh:
        pc = yaml.safe_load(fh)
    # multi_step counts from the loaded snapshot: DeepLabCut adds the start iteration to max_iter and uses it - start for the rate
    pc.update(init_weights=a.init_weights, multi_step=sched, save_iters=a.save_every, display_iters=a.display_every)
    with open(pc_path, "w") as fh:
        yaml.safe_dump(pc, fh)
    deeplabcut.train_network(
        a.config,
        shuffle=a.shuffle,
        trainingsetindex=0,
        max_snapshots_to_keep=a.keep,
        displayiters=a.display_every,
        saveiters=a.save_every,
        maxiters=a.max_iters,
        allow_growth=True,
    )


def parser():
    ap = argparse.ArgumentParser(prog="runner.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--config", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--dest-dir", required=True)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--gpu", type=int, default=-1, help="GPU index; -1 lets DeepLabCut choose")
    p.add_argument("--ext", default=".mp4,.MP4,.avi,.AVI")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pseudo-labels")
    p.add_argument("--config", required=True)
    p.add_argument("--labeled-data", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build-project")
    p.add_argument("--base-config", required=True)
    p.add_argument("--labels", required=True, help="labeller export (json), next to its frames/ folder")
    p.add_argument("--dest", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--base-head", choices=["keep", "nan", "pseudo"], default="nan")
    p.add_argument("--pseudo-csv", default=None)
    p.add_argument("--pseudo-lik", type=float, default=0.9)
    p.add_argument("--holdout", default="")
    p.add_argument("--only-videos", default="")
    p.add_argument("--new-only", action="store_true", help="do not copy the base project's labelled data")
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=272)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--init-weights", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--lr-schedule", required=True, help='json, e.g. "[[0.001, 10000], [0.0001, 30000]]"')
    p.add_argument("--max-iters", type=int, required=True)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--display-every", type=int, default=500)
    p.add_argument("--keep", type=int, default=7)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--net-type", default="resnet_50")
    p.add_argument("--augmenter", default="default")
    p.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    a = parser().parse_args(argv)
    {"analyze": cmd_analyze, "pseudo-labels": cmd_pseudo, "build-project": cmd_build, "train": cmd_train}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
