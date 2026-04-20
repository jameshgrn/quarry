"""Generate preview images from geospatial artifacts."""

from pathlib import Path
from typing import Any

import fiona
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image


def preview_artifact(input_path: str, output_path: str) -> dict[str, Any]:
    """Generate a preview PNG from a geospatial file.

    Args:
        input_path: Path to input raster or vector file
        output_path: Path for output PNG

    Returns:
        Dict with preview metadata for registration
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Detect type by extension
    ext = input_path.suffix.lower()

    if ext in (".geojson", ".json", ".shp", ".gpkg"):
        return _preview_vector(input_path, output_path)
    elif ext in (".tif", ".tiff", ".geotiff"):
        return _preview_raster(input_path, output_path)
    else:
        raise ValueError(f"unsupported extension for preview: {ext}")


def _preview_raster(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Generate PNG preview from raster."""
    with rasterio.open(input_path) as src:
        # Read first band for grayscale, or first 3 for RGB
        if src.count >= 3:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

            # Handle different dtypes
            def normalize(arr):
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    mask = ~np.isnan(arr)
                    if mask.any():
                        vmin, vmax = arr[mask].min(), arr[mask].max()
                        if vmax > vmin:
                            return np.clip((arr - vmin) / (vmax - vmin) * 255, 0, 255).astype(
                                np.uint8
                            )
                    return np.zeros_like(arr, dtype=np.uint8)
                else:
                    vmin, vmax = arr.min(), arr.max()
                    if vmax > vmin:
                        return ((arr.astype(np.float32) - vmin) / (vmax - vmin) * 255).astype(
                            np.uint8
                        )
                    return np.zeros_like(arr, dtype=np.uint8)

            rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")
        else:
            band = src.read(1)

            # Normalize to 0-255
            if band.dtype == np.float32 or band.dtype == np.float64:
                mask = ~np.isnan(band)
                if mask.any():
                    vmin, vmax = band[mask].min(), band[mask].max()
                else:
                    vmin, vmax = 0, 1
            else:
                vmin, vmax = band.min(), band.max()

            if vmax > vmin:
                normalized = ((band.astype(np.float32) - vmin) / (vmax - vmin) * 255).astype(
                    np.uint8
                )
            else:
                normalized = np.zeros_like(band, dtype=np.uint8)

            img = Image.fromarray(normalized, mode="L")

        # Resize if too large (max 1024px)
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        img.save(output_path, "PNG")

    return {
        "name": output_path.stem,
        "crs": None,  # Preview has no CRS
        "driver": "PNG",
        "width": img.width,
        "height": img.height,
    }


def _preview_vector(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Generate PNG preview from vector."""
    fig, ax = plt.subplots(figsize=(8, 8))

    with fiona.open(input_path) as src:
        bounds = src.bounds

        # Collect geometries
        for feat in src:
            geom = feat.geometry
            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]  # Exterior ring
                poly = Polygon(coords, fill=True, alpha=0.5, edgecolor="black", facecolor="blue")
                ax.add_patch(poly)
            elif geom["type"] == "Point":
                x, y = geom["coordinates"]
                ax.plot(x, y, "ko", markersize=5)
            elif geom["type"] == "LineString":
                coords = np.array(geom["coordinates"])
                ax.plot(coords[:, 0], coords[:, 1], "k-", linewidth=1)

    # Set bounds with small padding
    xmin, ymin, xmax, ymax = bounds
    padding = max(xmax - xmin, ymax - ymin) * 0.05
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)

    # Remove axes for clean preview
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Get dimensions from saved file
    with Image.open(output_path) as img:
        width, height = img.size

    return {
        "name": output_path.stem,
        "crs": None,
        "driver": "PNG",
        "width": width,
        "height": height,
    }
