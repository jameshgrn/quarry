"""Quarry Explorer — FastAPI application entry point.

Lane: adapter

Visual inspection UI for Quarry connectors. Serves a Leaflet map with
Planet basemap tiles and connector output overlays.

Usage:
    uv run uvicorn quarry_explorer.serve:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from quarry_explorer.api import SCENE_CATALOG, discover_scenes
from quarry_explorer.api import router as api_router

logger = logging.getLogger(__name__)

# Default data root — override via QUARRY_EXPLORER_DATA env var
_DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data" / "explorer"
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


def _initialize() -> None:
    """Discover scenes on startup."""
    import os

    import quarry_explorer.api as api_module

    data_root = Path(os.environ.get("QUARRY_EXPLORER_DATA", str(_DEFAULT_DATA)))
    api_module.DATA_ROOT = data_root

    SCENE_CATALOG.clear()
    catalog = discover_scenes(data_root)
    SCENE_CATALOG.update(catalog)

    logger.info(
        "Explorer ready — %d scene(s), data: %s",
        len(SCENE_CATALOG),
        data_root,
    )
    if not SCENE_CATALOG:
        logger.warning("No scenes found. Set QUARRY_EXPLORER_DATA to a directory with GeoTIFFs.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _initialize()
    yield


app = FastAPI(
    title="Quarry Explorer",
    version="0.1.0",
    description="Visual inspection UI for Quarry connectors with Planet basemap.",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Width", "X-Height"],
)

app.include_router(api_router)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Serve the explorer UI."""
    html_path = STATIC_DIR / "explorer.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Explorer UI not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())
