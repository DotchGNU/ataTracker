import json
from html.parser import HTMLParser

import pandas as pd
import pytest

from atatracker.common import OutputExistsError
from atatracker.config import config_from_dict
from atatracker.pipeline import run_phase, run_screen, split_tag
from atatracker.review.pages import build_picker, build_review, choices_to_csv


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.tags, self._in = [], [], False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self._in = tag == "script"

    def handle_endtag(self, tag):
        self._in = False

    def handle_data(self, data):
        if self._in:
            self.scripts.append(data)


def _embedded(html: str, name: str):
    p = _Collect()
    p.feed(html)
    assert p.tags[0] == "html" and "body" in p.tags
    js = "".join(p.scripts)
    start = js.index(name + "=") + len(name) + 1
    return json.JSONDecoder().raw_decode(js[start:])[0]


def test_phase_outputs_manifest_and_overwrite_guard(config_dict, tmp_path):
    cfg = config_from_dict(config_dict)
    out = tmp_path / "phase"
    t = run_phase(cfg, out, plots=True)
    assert len(t) == 1 and t.animal[0] == "animalA"
    m = json.loads((out / "run_manifest.json").read_text())
    assert m["step"] == "phase" and m["inputs"] == {"reference": 6, "targets": 1}
    assert m["versions"]["hmmlearn"]
    for f in (
        "phase_model.json",
        "source_phase-per-animal.csv",
        "twitching/animalA.png",
        "twitching/animalA.svg",
        "twitching/source_animalA.csv",
    ):
        assert (out / f).exists(), f
    src = pd.read_csv(out / "twitching" / "source_animalA.csv")
    assert set(src.columns) >= {
        "t_s",
        "chest_signal",
        "head_signal",
        "chest_display",
        "head_display",
        "head_onset_s",
        "last_twitching_s",
    }
    with pytest.raises(OutputExistsError):
        run_phase(cfg, out)
    run_phase(cfg, out, overwrite=True, plots=False)


def test_leave_one_out_for_reference_targets(config_dict, coord_dirs, tmp_path):
    config_dict["inputs"]["coord_sets"] = [dict(tag="ref", dir=str(coord_dirs["reference"]))]
    t = run_phase(config_from_dict(config_dict), tmp_path / "p", plots=False)
    assert t.hmm_leave_one_out.all() and len(t) == 6


def test_review_page_schema(config_dict, tmp_path):
    cfg = config_from_dict(config_dict)
    out = tmp_path / "phase"
    run_phase(cfg, out)
    html = build_review(out, "demo").read_text(encoding="utf-8")
    data = _embedded(html, "const DATA")
    assert isinstance(data, list) and len(data) == 1
    keys = {
        "animal",
        "group",
        "checkpoint_tag",
        "metric_flags",
        "note",
        "png",
        "head_on_min",
        "end_min",
        "chest_phase_min",
        "head_phase_min",
        "manual_last_s",
        "end_minus_manual_s",
        "head_lik_lt05",
        "head_step_p999",
        "bodylen_violation",
    }
    assert set(data[0]) == keys
    assert "__" not in html.replace("__proto__", "")


def test_screen_picker_and_choices(config_dict, coord_dirs, tmp_path):
    config_dict["inputs"]["coord_sets"].append(dict(tag="model-2k", dir=str(coord_dirs["targets"])))
    cfg = config_from_dict(config_dict)
    out = tmp_path / "screen"
    t = run_screen(cfg, out)
    assert len(t) == 2 and set(t.tag) == {"model-1k", "model-2k"}
    html = build_picker(out).read_text(encoding="utf-8")
    data = _embedded(html, "DATA")
    assert {d["tag"] for d in data} == {"model-1k", "model-2k"}
    assert set(data[0]) == {
        "animal",
        "tag",
        "family",
        "iteration",
        "png",
        "is_default",
        "head_onset_min",
        "end_min",
        "head_lik_lt05",
        "head_step_p999",
        "bodylen_violation",
    }
    exp = tmp_path / "choices.json"
    exp.write_text(
        json.dumps({"choices": [{"animal": "animalA", "note": "", "ticked": [{"tag": "model-1k"}, {"tag": "model-2k"}]}]})
    )
    c = choices_to_csv(exp, tmp_path / "choice.csv")
    assert c.checkpoint_tag.tolist() == ["model-2k"]


def test_split_tag():
    assert split_tag("family-a-250k") == ("family-a", 250000)
    assert split_tag("archived") == ("archived", None)
