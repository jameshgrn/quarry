"""Reproject geospatial files to a new CRS."""

from pathlib import Path
from typing import Any

import fiona
import rasterio
from fiona.crs import CRS as FionaCRS
from fiona.transform import transform_geom
from rasterio.crs import CRS as RasterioCRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


def reproject_file(
    input_path: str,
    output_path: str,
    target_crs: str,
) -> dict[str, Any]:
    """Reproject a geospatial file to a new CRS.

    Args:
        input_path: Path to input file
        output_path: Path for output file
        target_crs: Target CRS (EPSG:XXXX format or proj string)

    Returns:
        Dict with output metadata for registration
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Detect type by extension
    ext = input_path.suffix.lower()

    if ext in (".geojson", ".json", ".shp", ".gpkg"):
        return _reproject_vector(input_path, output_path, target_crs)
    elif ext in (".tif", ".tiff", ".geotiff"):
        return _reproject_raster(input_path, output_path, target_crs)
    else:
        raise ValueError(f"unsupported extension: {ext}")


def _reproject_vector(input_path: Path, output_path: Path, target_crs: str) -> dict[str, Any]:
    """Reproject a vector file."""
    target = FionaCRS.from_string(target_crs)

    with fiona.open(input_path) as src:
        src_crs = src.crs
        schema = src.schema.copy()

        # Transform all features
        features = []
        for feat in src:
            new_geom = transform_geom(src_crs, target, feat.geometry)
            features.append(
                {
                    "type": "Feature",
                    "geometry": new_geom,
                    "properties": feat.properties,
                }
            )

        # Write output
        driver = "GeoJSON" if output_path.suffix == ".geojson" else src.driver
        with fiona.open(
            output_path,
            "w",
            driver=driver,
            crs=target,
            schema=schema,
        ) as dst:
            for feat in features:
                dst.write(feat)

    # Get EPSG code if available
    epsg = target.to_epsg()
    crs_str = f"EPSG:{epsg}" if epsg else str(target)

    return {
        "name": output_path.stem,
        "crs": crs_str,
        "feature_count": len(features),
        "driver": driver,
    }


def _reproject_raster(input_path: Path, output_path: Path, target_crs: str) -> dict[str, Any]:
    """Reproject a raster file."""
    target = RasterioCRS.from_string(target_crs)

    with rasterio.open(input_path) as src:
        # Calculate new transform and dimensions
        transform, width, height = calculate_default_transform(
            src.crs, target, src.width, src.height, *src.bounds
        )

        # Update kwargs for output
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target,
                    resampling=Resampling.nearest,
                )

    # Return metadata for registration
    with rasterio.open(output_path) as dst:
        return {
            "name": output_path.stem,
            "crs": dst.crs.to_epsg() and f"EPSG:{dst.crs.to_epsg()}" or str(dst.crs),
            "band_count": dst.count,
            "driver": dst.driver,
        }
