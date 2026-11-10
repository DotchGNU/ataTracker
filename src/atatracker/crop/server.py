"""Local helper for the browser tools (localhost only): ``/detect``, ``/state`` and ``/save_json``.

The HTML pages stay on ``file://``; this server answers on ``127.0.0.1`` with CORS so the
pages can call it. Keep it running while a page is in use.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import CropConfig
from .detect import detect_boxes
from .media import grey_frames

LOG = logging.getLogger("atatracker")


def make_handler(cfg: CropConfig, save_root: str | Path):
    """Request handler bound to a configuration; ``/save_json`` writes only below ``save_root``."""
    state = Path(cfg.tool_dir) / "state.json"
    root = os.path.realpath(save_root)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: object) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(200, {})

        def do_GET(self) -> None:  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path == "/ping":
                    return self._send(200, {"ok": True})
                if u.path == "/state":
                    return self._send(200, json.loads(state.read_text(encoding="utf-8")) if state.exists() else {})
                if u.path == "/detect":
                    path = os.path.join(cfg.raw_dir, os.path.basename(q["raw"]))
                    start, end = float(q["start"]), float(q["end"])
                    frames = grey_frames(cfg.ffmpeg, path, f"{cfg.detect_fps:g}", cfg.detect_w, cfg.detect_h, start, end - start)
                    return self._send(
                        200,
                        detect_boxes(
                            frames,
                            int(q.get("n", 2)),
                            int(q["w"]),
                            int(q["h"]),
                            float(q.get("crop_w", cfg.crop_w)),
                            cfg.out_w,
                            cfg.out_h,
                        ),
                    )
                self._send(404, {"error": "not found"})
            except Exception as e:  # reported to the page
                self._send(500, {"error": str(e)})

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if self.path == "/state":
                    state.parent.mkdir(parents=True, exist_ok=True)
                    state.write_bytes(body)
                    return self._send(200, {"ok": True})
                if self.path == "/save_json":
                    j = json.loads(body)
                    d = os.path.realpath(j["dir"])
                    if not (d == root or d.startswith(root + os.sep)):
                        raise ValueError("folder outside the allowed save root")
                    os.makedirs(d, exist_ok=True)
                    p = os.path.join(d, os.path.basename(j["name"]))
                    if os.path.exists(p):
                        raise FileExistsError(f"{os.path.basename(p)} exists; choose another name")
                    with open(p, "w", encoding="utf-8") as fh:
                        json.dump(j["data"], fh, indent=1)
                    return self._send(200, {"ok": True, "path": p})
                self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": str(e)})

        def log_message(self, *a: object) -> None:
            pass

    return Handler


def serve(cfg: CropConfig, save_root: str | Path) -> None:
    """Run the helper on ``127.0.0.1:server_port`` until interrupted."""
    LOG.info("helper on http://localhost:%d (Ctrl+C to stop)", cfg.server_port)
    ThreadingHTTPServer(("127.0.0.1", cfg.server_port), make_handler(cfg, save_root)).serve_forever()
