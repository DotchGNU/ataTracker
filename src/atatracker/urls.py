"""File URLs for local HTML pages."""

from __future__ import annotations

import urllib.parse
from pathlib import Path


def file_url(path: str | Path, path_map: dict[str, str] | None = None) -> str:
    """``file://`` URL of a local file.

    ``path_map`` rewrites a path prefix first, e.g. ``{"/data/": "D:/data/"}`` when the page is opened
    by a browser that sees the files under another path.
    """
    p = str(Path(path).resolve()).replace("\\", "/")
    for a, b in (path_map or {}).items():
        if p.startswith(a):
            p = b + p[len(a) :]
            break
    if not p.startswith("/"):
        p = "/" + p
    return "file://" + urllib.parse.quote(p, safe="/:")


def relative_or_file_url(path: str | Path, page_dir: str | Path, path_map: dict[str, str] | None = None) -> str:
    """Relative URL when ``path`` lies under ``page_dir``, otherwise a ``file://`` URL."""
    p, d = Path(path).resolve(), Path(page_dir).resolve()
    try:
        return urllib.parse.quote(p.relative_to(d).as_posix())
    except ValueError:
        return file_url(p, path_map)
