"""RasterizeVectorOperator — burn vector geometries into a raster grid.

Lane: operator
Accepts: one vector artifact (Point, LineString, Polygon, or Multi variants)
Produces: one raster artifact (GeoTIFF)
Burn modes: constant value OR single numeric attribute per feature
Checks: crs_valid, dimensions_sane, nodata_background
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    Lineage,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import (
    OperatorError,
    OperatorParams,
    OperatorResult,
    OperatorSpec,
    ResourceScale,
)
from rasterio.features import rasterize as _rasterize
from rasterio.transform import Affine, from_bounds
from shapely.geometry import mapping, shape

# Supported rasterio/numpy dtype strings
_VALID_DTYPES = frozenset(
    {
        "uint8",
        "uint16",
        "uint32",
        "int16",
        "int32",
        "float32",
        "float64",
    }
)


@dataclass(frozen=True)
class RasterizeVectorParams(OperatorParams):
    """Parameters for vector rasterization."""

    output_path: str | None = None
    resolution: tuple[float, float] | None = None  # (x_res, y_res) in CRS units
    extent: tuple[float, float, float, float] | None = None  # xmin, ymin, xmax, ymax
    burn_value: float = 1.0  # constant burn; ignored if burn_attribute set
    burn_attribute: str | None = None  # feature property name for per-feature burn
    nodata: float = 0.0  # background / nodata value
    dtype: str = "float32"
    all_touched: bool = False  # rasterio rasterize all_touched semantics; True for thin features
    burn_attributes: tuple[str, ...] | None = None  # multi-band: one band per attribute name;
    # mutually exclusive with burn_attribute


if TYPE_CHECKING:
    import fiona


def _rasterize_single_band(
    src: fiona.Collection,
    transform: Affine,
    params: RasterizeVectorParams,
    height: int,
    width: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Build shapes and rasterize a single band.

    Returns (burned_2d_array, shapes_burned_count).
    """
    shapes: list[tuple[dict, float]] = []
    for feature in src:
        geom = shape(feature["geometry"])
        if geom is None or geom.is_empty:
            continue

        geojson = mapping(geom)

        if params.burn_attribute is not None:
            props = feature.get("properties", {})
            raw = props.get(params.burn_attribute)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
        else:
            val = float(params.burn_value)

        shapes.append((geojson, val))

    if shapes:
        burned = _rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=params.nodata,
            dtype=dtype,
            all_touched=params.all_touched,
        )
    else:
        burned = np.full((height, width), params.nodata, dtype=dtype)

    return burned, len(shapes)


def _rasterize_multi_band(
    src: fiona.Collection,
    transform: Affine,
    params: RasterizeVectorParams,
    height: int,
    width: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Build shapes per band and rasterize multiple bands.

    Returns (burned_3d_array_of_shape_NHW, total_shapes_burned_count).
    """
    band_count = len(params.burn_attributes)
    shapes_per_band: list[list[tuple[dict, float]]] = [[] for _ in range(band_count)]

    for feature in src:
        geom = shape(feature["geometry"])
        if geom is None or geom.is_empty:
            continue

        geojson = mapping(geom)
        props = feature.get("properties", {})

        for band_idx, attr in enumerate(params.burn_attributes):
            raw = props.get(attr)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            shapes_per_band[band_idx].append((geojson, val))

    band_arrays = []
    total_shapes = 0
    for band_shapes in shapes_per_band:
        if band_shapes:
            band_arr = _rasterize(
                band_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=params.nodata,
                dtype=dtype,
                all_touched=params.all_touched,
            )
        else:
            band_arr = np.full((height, width), params.nodata, dtype=dtype)
        band_arrays.append(band_arr)
        total_shapes += len(band_shapes)

    burned = np.stack(band_arrays, axis=0)
    return burned, total_shapes


class RasterizeVectorOperator:
    """Burn vector geometries (Point, LineString, Polygon, or Multi variants) into a raster grid.

    Input 0: vector artifact (any GeoJSON geometry type supported by rasterio.features.rasterize)

    Output: raster artifact (single-band or multi-band GeoTIFF). Each pixel
    covered by a geometry receives either a constant ``burn_value`` or the value
    of ``burn_attribute`` from that feature. Pixels not covered by any geometry
    receive ``nodata``. ``burn_attributes`` produces a multi-band GeoTIFF (one
    band per listed attribute). ``all_touched=True`` includes every pixel touched
    by a geometry edge.

    When ``extent`` is None the bounding box of the input vector is used.
    Resolution must be explicitly provided — no guessing.
    """

    @property
    def name(self) -> str:
        return "rasterize_vector"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.VECTOR,),
            output_type=ArtifactType.RASTER,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 1:
            errors.append(f"Exactly 1 vector input required, got {len(inputs)}")
            return errors

        vector = inputs[0]

        if vector.type != ArtifactType.VECTOR:
            errors.append(f"Input must be vector, got {vector.type.value}")
        if not vector.is_materialized:
            errors.append("Input vector is not materialized")

        if not isinstance(params, RasterizeVectorParams):
            errors.append("Params must be RasterizeVectorParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        if params.resolution is None:
            errors.append("resolution is required")
        else:
            rx, ry = params.resolution
            if rx <= 0 or ry <= 0:
                errors.append(f"resolution must be positive (x_res, y_res), got ({rx}, {ry})")

        if params.dtype not in _VALID_DTYPES:
            errors.append(f"Unsupported dtype '{params.dtype}'; valid: {sorted(_VALID_DTYPES)}")

        if params.extent is not None:
            xmin, ymin, xmax, ymax = params.extent
            if xmin >= xmax or ymin >= ymax:
                errors.append(f"Invalid extent: xmin >= xmax or ymin >= ymax ({params.extent})")

        if params.burn_attribute is not None and params.burn_attributes is not None:
            errors.append("burn_attribute and burn_attributes are mutually exclusive — use one")

        if params.burn_attributes is not None and len(params.burn_attributes) == 0:
            errors.append("burn_attributes must be non-empty when set")

        if params.burn_attributes is not None:
            for attr in params.burn_attributes:
                if not isinstance(attr, str):
                    errors.append("burn_attributes entries must be strings")
                    break

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, RasterizeVectorParams):
            raise OperatorError(self.name, "Params must be RasterizeVectorParams")
        import time
        import fiona
        import rasterio

        t0 = time.monotonic()
        vector_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rx, ry = params.resolution

        try:
            with fiona.open(vector_artifact.backing.uri) as src:
                vector_crs = str(src.crs) if src.crs else None
                if params.extent is not None:
                    xmin, ymin, xmax, ymax = params.extent
                else:
                    xmin, ymin, xmax, ymax = src.bounds
                width = max(1, int(np.ceil((xmax - xmin) / rx)))
                height = max(1, int(np.ceil((ymax - ymin) / ry)))
                transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
                dtype = np.dtype(params.dtype)
                if params.burn_attributes is not None:
                    band_count = len(params.burn_attributes)
                    burned, shapes_burned = _rasterize_multi_band(
                        src, transform, params, height, width, dtype
                    )
                else:
                    band_count = 1
                    burned, shapes_burned = _rasterize_single_band(
                        src, transform, params, height, width, dtype
                    )
            profile = {
                "driver": "GTiff",
                "dtype": dtype.name,
                "width": width,
                "height": height,
                "count": band_count,
                "crs": vector_crs,
                "transform": transform,
                "nodata": params.nodata,
            }
            with rasterio.open(output_path, "w", **profile) as dst:
                if band_count == 1:
                    dst.write(burned, 1)
                else:
                    for i in range(band_count):
                        dst.write(burned[i], i + 1)
        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name, f"Rasterization failed: {e}", inputs=[a.id for a in inputs]
            ) from e

        elapsed = time.monotonic() - t0
        with rasterio.open(output_path) as src:
            out_bounds = src.bounds
            out_crs = str(src.crs) if src.crs else None
            out_width = src.width
            out_height = src.height
            out_res = src.res
            out_nodata = src.nodata

        output_artifact = Artifact(
            type=ArtifactType.RASTER,
            name=output_path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs=out_crs,
                extent=(
                    out_bounds.left,
                    out_bounds.bottom,
                    out_bounds.right,
                    out_bounds.top,
                ),
                resolution=out_res,
                band_count=band_count,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in inputs),
                params={
                    "resolution": list(params.resolution),
                    "extent": list(params.extent) if params.extent else None,
                    "burn_value": params.burn_value,
                    "burn_attribute": params.burn_attribute,
                    "nodata": params.nodata,
                    "dtype": params.dtype,
                    "all_touched": params.all_touched,
                    "burn_attributes": list(params.burn_attributes)
                    if params.burn_attributes is not None
                    else None,
                },
            ),
            metadata={
                "format": "geotiff",
                "width": out_width,
                "height": out_height,
                "nodata": out_nodata,
                "shapes_burned": shapes_burned,
            },
        )
        checks = self._run_checks(output_artifact, inputs, out_width, out_height)
        return OperatorResult(artifact=output_artifact, checks=checks, timing_seconds=elapsed)

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "dimensions_sane", "nodata_background"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        inputs: list[Artifact],
        width: int,
        height: int,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

        # CRS valid — output must carry a CRS
        if output.spatial.crs:
            results.append(
                CheckResult(
                    check_name="crs_valid",
                    state=ValidationState.VALID,
                    message=f"CRS: {output.spatial.crs}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="crs_valid",
                    state=ValidationState.INVALID,
                    message="Output has no CRS",
                )
            )

        # Dimensions sane — width and height must be positive and bounded
        max_dim = 100_000  # sanity ceiling
        if 0 < width <= max_dim and 0 < height <= max_dim:
            results.append(
                CheckResult(
                    check_name="dimensions_sane",
                    state=ValidationState.VALID,
                    message=f"Grid dimensions: {width}x{height}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="dimensions_sane",
                    state=ValidationState.INVALID,
                    message=(f"Grid dimensions out of range: {width}x{height} (max {max_dim})"),
                )
            )

        # Nodata / background — output nodata must be set
        if output.metadata.get("nodata") is not None:
            results.append(
                CheckResult(
                    check_name="nodata_background",
                    state=ValidationState.VALID,
                    message=f"Nodata value: {output.metadata['nodata']}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="nodata_background",
                    state=ValidationState.WARN,
                    message="No nodata value set on output raster",
                )
            )

        return results
