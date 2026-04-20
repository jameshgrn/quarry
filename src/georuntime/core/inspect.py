"""File inspection logic for geospatial artifacts."""

from pathlib import Path

from georuntime.registry import Registry

RASTER_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".jp2", ".hgt"}
VECTOR_EXTENSIONS = {".shp", ".geojson", ".gpkg", ".kml", ".gml"}


def inspect_file(path: str, workspace: str | None = None) -> dict:
    """Inspect a geospatial file and register it.

    Args:
        path: Path to the file to inspect.
        workspace: Optional workspace directory for registry.

    Returns:
        The registered artifact dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file type is not supported.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = file_path.suffix.lower()

    if ext in RASTER_EXTENSIONS:
        return _inspect_raster(path, workspace)
    elif ext in VECTOR_EXTENSIONS:
        return _inspect_vector(path, workspace)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _inspect_raster(path: str, workspace: str | None = None) -> dict:
    """Inspect a raster file using rasterio."""
    import rasterio

    with rasterio.open(path) as src:
        bounds = src.bounds
        extent = {
            "xmin": bounds.left,
            "ymin": bounds.bottom,
            "xmax": bounds.right,
            "ymax": bounds.top,
        }

        artifact = {
            "name": Path(path).stem,
            "artifact_type": "raster",
            "path": str(path),
            "crs": str(src.crs) if src.crs else None,
            "extent": extent,
            "band_count": src.count,
            "feature_count": None,
            "driver": src.driver,
            "source_operation": "inspect",
            "source_inputs": None,
            "metadata": dict(src.tags()),
        }

    reg = Registry(workspace)
    artifact_id = reg.register(artifact)
    return reg.get(artifact_id)


def _inspect_vector(path: str, workspace: str | None = None) -> dict:
    """Inspect a vector file using fiona."""
    import fiona

    with fiona.open(path) as src:
        bounds = src.bounds
        extent = {
            "xmin": bounds[0],
            "ymin": bounds[1],
            "xmax": bounds[2],
            "ymax": bounds[3],
        }

        artifact = {
            "name": Path(path).stem,
            "artifact_type": "vector",
            "path": str(path),
            "crs": str(src.crs) if src.crs else None,
            "extent": extent,
            "band_count": None,
            "feature_count": len(src),
            "driver": src.driver,
            "source_operation": "inspect",
            "source_inputs": None,
            "metadata": dict(src.schema),
        }

    reg = Registry(workspace)
    artifact_id = reg.register(artifact)
    return reg.get(artifact_id)
