"""Build the review pages. Pages are static HTML with relative image links; choices are kept in the
browser's localStorage and exported as JSON (download) or copied as text."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from ..common import list_coord_files, load_coords
from ..config import Config
from ..crop.tool import template
from ..signal import compute_signals
from ..urls import relative_or_file_url

LOG = logging.getLogger("ataagent")
PICKER_NUM = ("head_on_min", "end_min", "head_lik_lt05", "head_step_p999", "bodylen_violation")
REVIEW_NUM = (
    "head_on_min",
    "end_min",
    "chest_phase_min",
    "head_phase_min",
    "manual_last_s",
    "end_minus_manual_s",
    "head_lik_lt05",
    "head_step_p999",
    "bodylen_violation",
)


def _num(v: object) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def _script_json(obj: object) -> str:
    """JSON safe to embed inside a ``<script>`` element."""
    return json.dumps(obj).replace("</", "<\\/")


def picker_data(screen: pd.DataFrame, out_dir: str | Path) -> list[dict]:
    """Card records of the checkpoint picker (one per animal and tag) from ``source_checkpoint-screen.csv``."""
    rows = []
    t = screen.sort_values(["animal", "set_order"])
    for _, r in t.iterrows():
        png = f"png/{r.animal}/{r.tag}.png"
        if not (Path(out_dir) / png).exists():
            raise FileNotFoundError(png)
        d = dict(
            animal=str(r.animal),
            tag=str(r.tag),
            family=str(r.family),
            iteration=None if pd.isna(r.iteration) else int(r.iteration),
            png=png,
            is_default=bool(r.is_default),
        )
        d.update({k: _num(r[k]) for k in PICKER_NUM})
        d["head_onset_min"] = d.pop("head_on_min")
        rows.append(d)
    return rows


def build_picker(
    screen_dir: str | Path,
    store_key: str = "ata-checkpoint-picker",
    family_notes: dict[str, str] | None = None,
    preselect: dict[str, list[str]] | None = None,
    display_prefix: dict[str, str] | None = None,
) -> Path:
    """Write ``checkpoint-picker.html`` into the ``ata screen`` output folder.

    One section per animal, one card per checkpoint (twitching plot, onset / end, label-free
    metrics). Several cards per animal can be ticked, with a note; the header tallies how many
    animals ticked each checkpoint. Export: JSON download (``checkpoint-choices.json``) or text.
    """
    out = Path(screen_dir)
    data = picker_data(pd.read_csv(out / "source_checkpoint-screen.csv"), out)
    page = (
        template("checkpoint_picker.html")
        .replace("__DATA__", _script_json(data))
        .replace("__FAMNOTE__", _script_json(family_notes or {}))
        .replace("__PRE__", _script_json(preselect or {}))
        .replace("__KEY__", store_key)
        .replace("__DISP__", _script_json(display_prefix or {}))
        .replace("__DOWNLOAD__", "checkpoint-choices.json")
    )
    p = out / "checkpoint-picker.html"
    p.write_text(page, encoding="utf-8")
    LOG.info("checkpoint picker: %d cards, %d animals", len(data), len({d["animal"] for d in data}))
    return p


def choices_to_csv(choices_json: str | Path, out_csv: str | Path, rule: str = "last") -> pd.DataFrame:
    """Turn the picker export into ``animal, checkpoint_tag`` (``inputs.checkpoint_choice``).

    With several ticks for one animal, ``rule="last"`` takes the last ticked checkpoint and
    ``rule="first"`` the first; animals without ticks are left out (they keep the default set).
    """
    doc = json.loads(Path(choices_json).read_text(encoding="utf-8"))
    rows = []
    for c in doc["choices"]:
        ticks = [t["tag"] for t in c.get("ticked", [])]
        if ticks:
            rows.append(
                dict(
                    animal=c["animal"],
                    checkpoint_tag=ticks[-1] if rule == "last" else ticks[0],
                    ticked=";".join(ticks),
                    note=c.get("note", ""),
                )
            )
    t = pd.DataFrame(rows, columns=["animal", "checkpoint_tag", "ticked", "note"])
    t.to_csv(out_csv, index=False, encoding="utf-8")
    return t


def review_data(table: pd.DataFrame, out_dir: str | Path) -> list[dict]:
    """Card records of the twitching review page from ``source_phase-per-animal.csv``."""
    rows = []
    for _, r in table.iterrows():
        png = f"twitching/{r.animal}.png"
        if not (Path(out_dir) / png).exists():
            raise FileNotFoundError(png)
        d = dict(
            animal=str(r.animal),
            group=str(r.group) if isinstance(r.group, str) else "",
            checkpoint_tag=str(r.checkpoint_tag),
            metric_flags=r.metric_flags if isinstance(r.metric_flags, str) else "",
            note="",
            png=png,
        )
        d.update({k: _num(r.get(k)) for k in REVIEW_NUM})
        rows.append(d)
    return rows


def build_review(phase_dir: str | Path, set_name: str, store_key: str | None = None) -> Path:
    """Write ``twitching-review.html`` into the ``ata phase`` output folder.

    One card per animal (twitching plot, group, onset / end, manual last twitching, quality
    metrics, flags, checkpoint). Click the plot to mark a problem, add a note; "open video" shows
    ``viewer/viewer.html`` (``ata review viewer``) for that animal in a side panel.
    """
    out = Path(phase_dir)
    t = pd.read_csv(out / "source_phase-per-animal.csv")
    data = review_data(t, out)
    page = (
        template("twitching_review.html")
        .replace("__DATA__", _script_json(data))
        .replace("__KEY__", store_key or f"ata-review-{set_name}")
        .replace("__SET__", set_name)
        .replace("__DOWNLOAD__", f"twitching-review-{set_name}.json")
    )
    p = out / "twitching-review.html"
    p.write_text(page, encoding="utf-8")
    LOG.info("review page: %d animals", len(data))
    return p


def build_viewer(cfg: Config, phase_dir: str | Path, video_dir: str, frame_w: int = 480) -> Path:
    """Synchronised viewer: video with CHEST / HEAD pointers above the amplitude trace, cursor follows the video.

    Writes ``<phase_dir>/viewer/viewer.html`` and ``viewer/data/<animal>.js``; videos are linked, never copied.
    """
    out = Path(phase_dir) / "viewer"
    (out / "data").mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(Path(phase_dir) / "source_phase-per-animal.csv")
    sets = {s.tag: list_coord_files(s.dir, cfg.inputs.coord_glob, cfg.inputs.name_separator) for s in cfg.inputs.coord_sets}

    def r(a, k: int = 2) -> list[float]:
        return [round(float(x), k) for x in a]

    index = []
    for _, x in t.iterrows():
        vid = Path(video_dir) / f"{x.animal}{cfg.inputs.video_suffix}"
        if not vid.exists():
            LOG.warning("missing video for %s", x.animal)
            continue
        d = load_coords(sets[x.checkpoint_tag][x.animal], cfg.signal.chest_bodypart, cfg.signal.head_bodypart)
        s = compute_signals(d, cfg.signal, cfg.phase.bin_s)
        js = dict(
            sample=x.animal,
            fps=s.fps,
            n=len(d),
            cx=r(d.cx),
            cy=r(d.cy),
            hx=r(d.hx),
            hy=r(d.hy),
            hl=r(d.hl),
            ca=r(s.chest, 3),
            ha=r(s.head, 3),
            cpk=r(s.chest_peaks_s),
            hpk=r(s.head_peaks_s),
            hon=None if pd.isna(x.head_on_min) else float(x.head_on_min) * 60,
            end=None if pd.isna(x.end_min) else float(x.end_min) * 60,
        )
        (out / "data" / f"{x.animal}.js").write_text(f"window.VD={json.dumps(js)};", encoding="utf-8")
        grp = x.group if isinstance(x.group, str) else ""
        index.append(dict(cohort=grp, sample=x.animal, video=relative_or_file_url(vid, out, cfg.crop.url_path_map)))
    sc = cfg.signal
    sig_txt = (
        f"chest signal {sc.chest_mode}, head signal {sc.head_mode}, "
        f"peak threshold {sc.peak_threshold:g}, likelihood mask {sc.likelihood_min:g}"
    )
    html = (
        template("viewer.html")
        .replace("/*INDEX*/[]", _script_json(index))
        .replace("__TAG__", ", ".join(sorted(set(t.checkpoint_tag))))
        .replace("__SIG__", sig_txt)
        .replace("__FRAME_W__", str(frame_w))
    )
    p = out / "viewer.html"
    p.write_text(html, encoding="utf-8")
    LOG.info("viewer: %d of %d animals", len(index), len(t))
    return p
