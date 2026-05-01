"""Explorer API router.

Endpoints:
    GET /api/planet/{z}/{y}/{x}              Proxy Planet basemap tiles
    GET /api/scenes                           List discovered scenes
    GET /api/overlay/{scene_id}/{layer_id}   Render layer as RGBA PNG
    GET /api/bounds/{scene_id}               Geographic bounds for overlay
    POST /api/process                         Process SLC HDF5 → geocoded σ0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from rasterio.warp import transform_bounds

from quarry_explorer.render import render_layer, rgba_to_png

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

PLANET_TILE_URL = (
    "https://tiles.planet.com/basemaps/v1/planet-tiles/{mosaic}/gmap/{z}/{x}/{y}.png?api_key={key}"
)
DEFAULT_MOSAIC = "global_monthly_2024_01_mosaic"


@dataclass(frozen=True)
class LayerInfo:
    id: str
    name: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class SceneInfo:
    id: str
    layers: tuple[LayerInfo, ...]
    bounds: tuple[float, float, float, float]  # west, south, east, north in 4326


# Populated by serve.py
SCENE_CATALOG: dict[str, SceneInfo] = {}
DATA_ROOT: Path = Path("data")

# Layer display names
LAYER_NAMES: dict[str, str] = {
    "sigma0_plus": "Sigma0 (Plus Y)",
    "sigma0_minus": "Sigma0 (Minus Y)",
    "sigma0": "Sigma0 Backscatter",
    "ifgram_power": "Interferogram Power",
    "ifgram_phase": "Interferometric Phase",
    "slc_phase": "SLC Phase",
    "rgb": "Planet RGB",
}


# ── Planet basemap proxy ─────────────────────────────────────────────────────


def _planet_key() -> str:
    key = os.environ.get("PLANET_API_KEY", "")
    if not key:
        raise HTTPException(503, detail="PLANET_API_KEY not set")
    return key


@router.get("/planet/{z}/{y}/{x}")
def planet_tile(
    z: int,
    y: int,
    x: int,
    mosaic: str = Query(default=DEFAULT_MOSAIC),
):
    """Proxy Planet basemap tiles to avoid exposing API key in frontend."""
    key = _planet_key()
    url = PLANET_TILE_URL.format(mosaic=mosaic, z=z, x=x, y=y, key=key)

    try:
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            tile_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "image/png")
    except Exception as exc:
        logger.warning("Planet tile fetch failed: %s", exc)
        raise HTTPException(502, detail="Planet tile unavailable") from exc

    return Response(
        content=tile_bytes,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Scene catalog ────────────────────────────────────────────────────────────


@router.get("/scenes")
def list_scenes():
    """List all discovered scenes."""
    scenes = []
    for scene_id, scene in sorted(SCENE_CATALOG.items()):
        scenes.append(
            {
                "id": scene.id,
                "bounds": list(scene.bounds),
                "layers": [
                    {"id": ly.id, "name": ly.name, "width": ly.width, "height": ly.height}
                    for ly in scene.layers
                ],
            }
        )
    return {"scenes": scenes, "total": len(scenes)}


# ── Overlay rendering ────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _read_layer(path: str) -> np.ndarray:
    """Read raster data, cached. Returns (H, W) for single-band, (bands, H, W) for multi."""
    with rasterio.open(path) as src:
        if src.count >= 3:
            return src.read().astype(np.float32)
        return src.read(1).astype(np.float32)


@router.get("/overlay/{scene_id}/{layer_id}")
def get_overlay(scene_id: str, layer_id: str):
    """Render a scene layer as an RGBA PNG image overlay."""
    scene = SCENE_CATALOG.get(scene_id)
    if scene is None:
        raise HTTPException(404, detail=f"Scene not found: {scene_id}")

    layer = next((ly for ly in scene.layers if ly.id == layer_id), None)
    if layer is None:
        raise HTTPException(404, detail=f"Layer not found: {layer_id}")

    data = _read_layer(str(layer.path))
    rgba = render_layer(data, layer_id)
    png = rgba_to_png(rgba)

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Width": str(layer.width),
            "X-Height": str(layer.height),
        },
    )


@router.get("/bounds/{scene_id}")
def get_bounds(scene_id: str):
    """Return geographic bounds for a scene."""
    scene = SCENE_CATALOG.get(scene_id)
    if scene is None:
        raise HTTPException(404, detail=f"Scene not found: {scene_id}")
    return {"bounds": list(scene.bounds)}


# ── Discovery ────────────────────────────────────────────────────────────────


def discover_scenes(data_root: Path) -> dict[str, SceneInfo]:
    """Scan data directory for scenes with GeoTIFF layers.

    Supports two layouts:
        data_root/{scene_id}/sigma0_plus.tif   (quarry SLCConnector output)
        data_root/geo/{scene_id}/sigma0.tif    (reverse_eng_swot layout)
    """
    catalog: dict[str, SceneInfo] = {}

    search_dirs: list[Path] = []
    geo_dir = data_root / "geo"
    if geo_dir.is_dir():
        search_dirs.extend(d for d in geo_dir.iterdir() if d.is_dir())
    if data_root.is_dir():
        search_dirs.extend(d for d in data_root.iterdir() if d.is_dir() and d.name != "geo")

    for scene_dir in search_dirs:
        layers: list[LayerInfo] = []
        scene_bounds: tuple[float, float, float, float] | None = None

        for tif in sorted(scene_dir.glob("*.tif")):
            layer_id = tif.stem
            try:
                with rasterio.open(tif) as src:
                    w, h = src.width, src.height
                    if src.crs is not None:
                        left, bottom, right, top = transform_bounds(
                            src.crs,
                            "EPSG:4326",
                            *src.bounds,
                        )
                        if scene_bounds is None:
                            scene_bounds = (left, bottom, right, top)
                        else:
                            scene_bounds = (
                                min(scene_bounds[0], left),
                                min(scene_bounds[1], bottom),
                                max(scene_bounds[2], right),
                                max(scene_bounds[3], top),
                            )
            except Exception:
                logger.warning("Failed to read header: %s", tif)
                continue

            layers.append(
                LayerInfo(
                    id=layer_id,
                    name=LAYER_NAMES.get(layer_id, layer_id),
                    path=tif,
                    width=w,
                    height=h,
                )
            )

        if not layers or scene_bounds is None:
            continue

        catalog[scene_dir.name] = SceneInfo(
            id=scene_dir.name,
            layers=tuple(layers),
            bounds=scene_bounds,
        )

    logger.info("Discovered %d scenes", len(catalog))
    return catalog


# ── PIXC ingest ──────────────────────────────────────────────────────────────


PIXC_LAYERS: dict[str, str] = {
    "sig0": "σ0 Backscatter",
    "water_frac": "Water Fraction",
    "ifgram_power": "Coherent Power",
    "height": "Water Surface Height",
    "classification": "Classification",
}


@router.post("/process")
def process_pixc(
    pixc_path: str = Query(..., description="Path to SWOT L2 HR PIXC NetCDF"),
    resolution_m: float = Query(default=15.0, description="Target resolution in meters"),
    layers: str = Query(default="sig0", description="Comma-separated layer names"),
):
    """Rasterize PIXC pixel cloud directly at JPL's lat/lon.

    No SLC processing, no GCPs, no warping. Just scatter-to-grid from the
    PIXC lat/lon/value arrays. Perfect geolocation from JPL's operational
    pipeline with full corrections (terrain, tropo, iono, tides).
    """
    import h5py

    pixc = Path(pixc_path).resolve()
    if not pixc.exists():
        raise HTTPException(404, detail=f"PIXC file not found: {pixc}")

    scene_id = pixc.stem
    out_dir = DATA_ROOT / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [name.strip() for name in layers.split(",")]
    resolution_deg = resolution_m / 111320.0

    logger.info("Rasterizing PIXC: %s (layers: %s)", pixc, requested)

    with h5py.File(str(pixc), "r") as ds:
        pc = ds["pixel_cloud"]
        lat = pc["latitude"][:]
        lon = pc["longitude"][:]

        valid_geo = np.isfinite(lat) & np.isfinite(lon)

        # Compute output grid from PIXC extent
        west = float(lon[valid_geo].min()) - resolution_deg
        east = float(lon[valid_geo].max()) + resolution_deg
        south = float(lat[valid_geo].min()) - resolution_deg
        north = float(lat[valid_geo].max()) + resolution_deg
        grid_w = int(np.ceil((east - west) / resolution_deg))
        grid_h = int(np.ceil((north - south) / resolution_deg))

        from rasterio.transform import from_bounds

        transform = from_bounds(west, south, east, north, grid_w, grid_h)

        # Pixel indices into the output grid
        cols = ((lon - west) / resolution_deg).astype(np.int32)
        rows = ((north - lat) / resolution_deg).astype(np.int32)
        in_grid = valid_geo & (cols >= 0) & (cols < grid_w) & (rows >= 0) & (rows < grid_h)

        result_layers: list[LayerInfo] = []

        for layer_name in requested:
            if layer_name not in pc:
                logger.warning("PIXC layer '%s' not found, skipping", layer_name)
                continue

            values = pc[layer_name][:]
            valid = in_grid & np.isfinite(values.astype(np.float64))

            grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
            grid[rows[valid], cols[valid]] = values[valid].astype(np.float32)

            out_path = out_dir / f"{layer_name}.tif"
            profile = {
                "driver": "GTiff",
                "height": grid_h,
                "width": grid_w,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": transform,
                "compress": "deflate",
                "nodata": float("nan"),
            }
            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(grid, 1)

            pct = 100 * np.isfinite(grid).sum() / grid.size
            logger.info("  %s: %dx%d, %.1f%% coverage", layer_name, grid_w, grid_h, pct)

            result_layers.append(
                LayerInfo(
                    id=layer_name,
                    name=PIXC_LAYERS.get(layer_name, layer_name),
                    path=out_path,
                    width=grid_w,
                    height=grid_h,
                )
            )

    if not result_layers:
        raise HTTPException(500, detail="No valid layers produced")

    scene_bounds = (west, south, east, north)
    scene = SceneInfo(
        id=scene_id,
        layers=tuple(result_layers),
        bounds=scene_bounds,
    )
    SCENE_CATALOG[scene_id] = scene
    _read_layer.cache_clear()

    return {
        "scene_id": scene_id,
        "bounds": list(scene_bounds),
        "layers": [ly.id for ly in result_layers],
        "message": f"Rasterized {len(result_layers)} PIXC layers at {resolution_m}m",
    }
