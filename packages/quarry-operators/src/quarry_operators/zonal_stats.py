"""ZonalStatsOperator — compute raster statistics per vector polygon zone.

Lane: operator
Accepts: one raster artifact + one vector artifact (polygons)
Produces: one table artifact (CSV with per-zone statistics)
Checks: row_count_matches, schema_complete, crs_valid
"""

from __future__ import annotations

import csv
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

STAT_COLUMNS = ("count", "sum", "mean", "min", "max", "std")


@dataclass(frozen=True)
class ZonalStatsParams(OperatorParams):
    """Parameters for zonal statistics."""

    output_path: str | None = None
    band: int = 1
    zone_id_field: str | None = None


class ZonalStatsOperator:
    """Compute raster statistics per vector polygon zone.

    Input 0: raster artifact (single band used, selected by ``band`` param)
    Input 1: vector artifact (polygon geometries define zones)

    Output: table artifact (CSV) with one row per zone and columns for each
    statistic.  Row order matches feature order in the input vector.
    """

    @property
    def name(self) -> str:
        return "zonal_stats"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER, ArtifactType.VECTOR),
            output_type=ArtifactType.TABLE,
            min_inputs=2,
            max_inputs=2,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 2:
            errors.append(f"Exactly 2 inputs required (raster + vector), got {len(inputs)}")
            return errors

        raster, vector = inputs

        if raster.type != ArtifactType.RASTER:
            errors.append(f"First input must be raster, got {raster.type.value}")
        if vector.type != ArtifactType.VECTOR:
            errors.append(f"Second input must be vector, got {vector.type.value}")

        if not raster.is_materialized:
            errors.append("Raster input is not materialized")
        if not vector.is_materialized:
            errors.append("Vector input is not materialized")

        # CRS must match
        raster_crs = raster.spatial.crs
        vector_crs = vector.spatial.crs
        if raster_crs and vector_crs and raster_crs != vector_crs:
            errors.append(f"CRS mismatch: raster={raster_crs}, vector={vector_crs}")

        if not isinstance(params, ZonalStatsParams):
            errors.append("Params must be ZonalStatsParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        if params.band < 1:
            errors.append(f"band must be >= 1, got {params.band}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, ZonalStatsParams):
            raise OperatorError(self.name, "Params must be ZonalStatsParams")

        import time

        import fiona
        import rasterio
        from shapely.geometry import shape

        t0 = time.monotonic()

        raster_artifact, vector_artifact = inputs
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(raster_artifact.backing.uri) as src:
                band_data = src.read(params.band)
                transform = src.transform
                nodata = src.nodata
                raster_crs = str(src.crs) if src.crs else None

                rows: list[dict] = []

                with fiona.open(vector_artifact.backing.uri) as vec:
                    for i, feature in enumerate(vec):
                        geom = shape(feature["geometry"])
                        zone_id = self._resolve_zone_id(feature, params.zone_id_field, i)
                        stats = self._compute_zone_stats(
                            geom, band_data, transform, nodata, src.height, src.width
                        )
                        rows.append({"zone_id": zone_id, **stats})

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Zonal stats failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        # Write CSV
        fieldnames = ["zone_id", *STAT_COLUMNS]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        elapsed = time.monotonic() - t0

        output_artifact = Artifact(
            type=ArtifactType.TABLE,
            name=output_path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs=raster_crs,
                feature_count=len(rows),
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in inputs),
                params={
                    "band": params.band,
                    "zone_id_field": params.zone_id_field,
                    "stat_columns": list(STAT_COLUMNS),
                },
            ),
            metadata={
                "format": "csv",
                "stat_columns": list(STAT_COLUMNS),
                "zone_count": len(rows),
            },
        )

        checks = self._run_checks(output_artifact, inputs, rows)

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["row_count_matches", "schema_complete", "crs_valid"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_zone_id(feature: dict, zone_id_field: str | None, index: int) -> str:
        """Extract zone identifier from feature properties or fall back to index."""
        if zone_id_field and zone_id_field in feature.get("properties", {}):
            return str(feature["properties"][zone_id_field])
        return str(index)

    @staticmethod
    def _compute_zone_stats(
        geom,
        band_data,
        transform,
        nodata,
        height: int,
        width: int,
    ) -> dict:
        """Compute statistics for pixels within a single zone geometry."""
        import numpy as np
        from rasterio.features import geometry_mask
        from shapely.geometry import mapping

        nan_row = {col: float("nan") for col in STAT_COLUMNS}

        if geom.is_empty:
            return nan_row

        geojson = mapping(geom)
        try:
            mask = geometry_mask(
                [geojson],
                out_shape=(height, width),
                transform=transform,
                invert=True,  # True inside geometry
            )
        except Exception:
            return nan_row

        pixels = band_data[mask]

        # Exclude nodata
        if nodata is not None:
            if np.isnan(nodata):
                pixels = pixels[~np.isnan(pixels)]
            else:
                pixels = pixels[pixels != nodata]

        if len(pixels) == 0:
            return nan_row

        pixels = pixels.astype(np.float64)
        return {
            "count": int(len(pixels)),
            "sum": float(np.sum(pixels)),
            "mean": float(np.mean(pixels)),
            "min": float(np.min(pixels)),
            "max": float(np.max(pixels)),
            "std": float(np.std(pixels)),
        }

    def _run_checks(
        self,
        output: Artifact,
        inputs: list[Artifact],
        rows: list[dict],
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

        # Row count matches input feature count
        vector = inputs[1]
        expected = vector.spatial.feature_count
        actual = len(rows)
        if expected is not None and actual == expected:
            results.append(
                CheckResult(
                    check_name="row_count_matches",
                    state=ValidationState.VALID,
                    message=f"Output rows ({actual}) match input features ({expected})",
                )
            )
        elif expected is not None:
            results.append(
                CheckResult(
                    check_name="row_count_matches",
                    state=ValidationState.INVALID,
                    message=f"Output rows ({actual}) != input features ({expected})",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="row_count_matches",
                    state=ValidationState.WARN,
                    message="Cannot verify row count — input feature_count unknown",
                )
            )

        # Schema complete
        expected_cols = {"zone_id", *STAT_COLUMNS}
        if rows:
            actual_cols = set(rows[0].keys())
            if actual_cols == expected_cols:
                results.append(
                    CheckResult(
                        check_name="schema_complete",
                        state=ValidationState.VALID,
                        message=f"All expected columns present: {sorted(expected_cols)}",
                    )
                )
            else:
                missing = expected_cols - actual_cols
                results.append(
                    CheckResult(
                        check_name="schema_complete",
                        state=ValidationState.INVALID,
                        message=f"Missing columns: {sorted(missing)}",
                    )
                )
        else:
            results.append(
                CheckResult(
                    check_name="schema_complete",
                    state=ValidationState.WARN,
                    message="No rows produced — cannot verify schema",
                )
            )

        # CRS valid
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

        return results
