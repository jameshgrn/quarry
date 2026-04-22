"""RasterizeVectorOperator — burn vector polygons into a raster grid.

Lane: operator
Accepts: one vector artifact (polygons)
Produces: one raster artifact (GeoTIFF)
Burn modes: constant value OR single numeric attribute per feature
Checks: crs_valid, dimensions_sane, nodata_background
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    resolution: tuple[float, float] = (0.0, 0.0)  # (x_res, y_res) in CRS units
    extent: tuple[float, float, float, float] | None = None  # xmin, ymin, xmax, ymax
    burn_value: float = 1.0  # constant burn; ignored if burn_attribute set
    burn_attribute: str | None = None  # feature property name for per-feature burn
    nodata: float = 0.0  # background / nodata value
    dtype: str = "float32"


class RasterizeVectorOperator:
    """Burn vector polygon geometries into a raster grid.

    Input 0: vector artifact (polygon geometries)

    Output: raster artifact (single-band GeoTIFF). Each pixel covered by a
    polygon receives either a constant ``burn_value`` or the value of
    ``burn_attribute`` from that feature. Pixels not covered by any
    polygon receive ``nodata``.

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

        rx, ry = params.resolution
        if rx <= 0 or ry <= 0:
            errors.append(f"resolution must be positive (x_res, y_res), got ({rx}, {ry})")

        if params.dtype not in _VALID_DTYPES:
            errors.append(f"Unsupported dtype '{params.dtype}'; valid: {sorted(_VALID_DTYPES)}")

        if params.extent is not None:
            xmin, ymin, xmax, ymax = params.extent
            if xmin >= xmax or ymin >= ymax:
                errors.append(f"Invalid extent: xmin >= xmax or ymin >= ymax ({params.extent})")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, RasterizeVectorParams):
            raise OperatorError(self.name, "Params must be RasterizeVectorParams")

        import time

        import fiona
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from shapely.geometry import shape

        t0 = time.monotonic()

        vector_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rx, ry = params.resolution

        try:
            with fiona.open(vector_artifact.backing.uri) as src:
                vector_crs = str(src.crs) if src.crs else None

                # Determine extent
                if params.extent is not None:
                    xmin, ymin, xmax, ymax = params.extent
                else:
                    bounds = src.bounds
                    xmin, ymin, xmax, ymax = bounds

                # Compute grid dimensions — snap to whole pixels
                width = max(1, int(np.ceil((xmax - xmin) / rx)))
                height = max(1, int(np.ceil((ymax - ymin) / ry)))

                transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

                # Build (geometry, value) pairs
                shapes: list[tuple[dict, float]] = []
                for feature in src:
                    geom = shape(feature["geometry"])
                    if geom is None or geom.is_empty:
                        continue

                    from shapely.geometry import mapping

                    geojson = mapping(geom)

                    if params.burn_attribute is not None:
                        props = feature.get("properties", {})
                        raw = props.get(params.burn_attribute)
                        if raw is None:
                            continue  # skip features missing the attribute
                        try:
                            val = float(raw)
                        except (TypeError, ValueError):
                            continue  # skip non-numeric
                    else:
                        val = float(params.burn_value)

                    shapes.append((geojson, val))

            # Rasterize
            dtype = np.dtype(params.dtype)
            if shapes:
                burned = rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=params.nodata,
                    dtype=dtype,
                )
            else:
                burned = np.full((height, width), params.nodata, dtype=dtype)

            # Write GeoTIFF
            profile = {
                "driver": "GTiff",
                "dtype": dtype.name,
                "width": width,
                "height": height,
                "count": 1,
                "crs": vector_crs,
                "transform": transform,
                "nodata": params.nodata,
            }
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(burned, 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Rasterization failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        elapsed = time.monotonic() - t0

        # Read back for fresh metadata
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
                band_count=1,
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
                },
            ),
            metadata={
                "format": "geotiff",
                "width": out_width,
                "height": out_height,
                "nodata": out_nodata,
                "shapes_burned": len(shapes),
            },
        )

        checks = self._run_checks(output_artifact, inputs, out_width, out_height)

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

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
