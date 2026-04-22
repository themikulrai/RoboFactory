"""Assemble the static site: copy shared assets + render index.html from template."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

THIS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = THIS_DIR / "templates"
ASSETS_DIR = THIS_DIR / "site_assets"
SITE_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/site")

# Search paths for plotly.min.js — any conda env that has plotly installed works.
PLOTLY_SEARCH = [
    "/iris/u/mikulrai/data/miniforge3/lib/python3.13/site-packages/plotly/package_data/plotly.min.js",
]


def _find_plotly_js() -> Path:
    for p in PLOTLY_SEARCH:
        if Path(p).exists():
            return Path(p)
    # fall back to glob in any conda env
    for p in Path("/iris/u/mikulrai/data/miniforge3").rglob("plotly/package_data/plotly.min.js"):
        return p
    raise FileNotFoundError(
        "plotly.min.js not found; `pip install plotly` in any env, or edit PLOTLY_SEARCH."
    )


def build() -> None:
    shared_dir = SITE_ROOT / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(_find_plotly_js(), shared_dir / "plotly.min.js")
    shutil.copy(ASSETS_DIR / "site.css", shared_dir / "site.css")
    shutil.copy(ASSETS_DIR / "app.js", shared_dir / "app.js")

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape())
    html = env.get_template("index.html.j2").render(build_id=int(time.time()))
    (SITE_ROOT / "index.html").write_text(html)
    print(f"site: {SITE_ROOT / 'index.html'}")


if __name__ == "__main__":
    build()
