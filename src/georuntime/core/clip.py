"""Clip geospatial files to a bounding box or mask."""

from pathlib import Path
from typing import Any

import fiona
import rasterio
import shapely.geometry
from rasterio.mask import mask as raster_mask


def clip_file(
    input_path: str,
    output_path: str,
    bounds: tuple[float, float, float, float] | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    """Clip a geospatial file to bounds or a mask geometry.

    Args:
        input_path: Path to input file
        output_path: Path for output file
        bounds: (xmin, ymin, xmax, ymax) for clipping
        mask_path: Path to vector file to use as clip mask

    Returns:
        Dict with output metadata for registration
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if bounds is None and mask_path is None:
        raise ValueError("must provide either bounds or mask_path")

    # Detect type by extension
    ext = input_path.suffix.lower()

    if ext in (".geojson", ".json", ".shp", ".gpkg"):
        return _clip_vector(input_path, output_path, bounds, mask_path)
    elif ext in (".tif", ".tiff", ".geotiff"):
        return _clip_raster(input_path, output_path, bounds, mask_path)
    else:
        raise ValueError(f"unsupported extension: {ext}")


def _clip_vector(
    input_path: Path,
    output_path: Path,
    bounds: tuple[float, float, float, float] | None,
    mask_path: str | None,
) -> dict[str, Any]:
    """Clip vector features to bounds or mask."""
    # Build clip geometry
    if mask_path:
        # Use mask file bounds as clip region
        with fiona.open(mask_path) as mask_src:
            mask_bounds = mask_src.bounds
            clip_geom = shapely.geometry.box(*mask_bounds)
    else:
        clip_geom = shapely.geometry.box(*bounds)

    with fiona.open(input_path) as src:
        src_crs = src.crs
        schema = src.schema.copy()

        # Filter features intersecting clip geometry
        features = []
        for feat in src:
            geom = shapely.geometry.shape(feat.geometry)
            if geom.intersects(clip_geom):
                # Clip geometry to bounds
                clipped = geom.intersection(clip_geom)
                if not clipped.is_empty:
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": shapely.geometry.mapping(clipped),
                            "properties": feat.properties,
                        }
                    )

        if not features:
            raise ValueError("no features intersect clip bounds")

        # Write output
        driver = "GeoJSON" if output_path.suffix == ".geojson" else src.driver
        with fiona.open(
            output_path,
            "w",
            driver=driver,
            crs=src_crs,
            schema=schema,
        ) as dst:
            for feat in features:
                dst.write(feat)

    epsg = src_crs.to_epsg() if hasattr(src_crs, "to_epsg") else None
    crs_str = f"EPSG:{epsg}" if epsg else str(src_crs)

    return {
        "name": output_path.stem,
        "crs": crs_str,
        "feature_count": len(features),
        "driver": driver,
    }


def _clip_raster(
    input_path: Path,
    output_path: Path,
    bounds: tuple[float, float, float, float] | None,
    mask_path: str | None,
) -> dict[str, Any]:
    """Clip raster to bounds or mask."""
    with rasterio.open(input_path) as src:
        if mask_path:
            # Use mask file geometry
            with fiona.open(mask_path) as mask_src:
                shapes = [feat.geometry for feat in mask_src]
                out_image, out_transform = raster_mask(src, shapes, crop=True)
        else:
            # Use bounds box

            geom = shapely.geometry.box(*bounds)
            shapes = [shapely.geometry.mapping(geom)]
            out_image, out_transform = raster_mask(src, shapes, crop=True)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

    # Return metadata
    with rasterio.open(output_path) as dst:
        return {
            "name": output_path.stem,
            "crs": dst.crs.to_epsg() and f"EPSG:{dst.crs.to_epsg()}" or str(dst.crs),
            "band_count": dst.count,
            "driver": dst.driver,
        }
