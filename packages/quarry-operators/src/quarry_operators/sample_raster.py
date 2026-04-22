"""SampleRasterOperator — sample raster values at point locations.

Lane: operator
Accepts: one raster artifact + one vector artifact (points)
Produces: one table artifact (CSV with sampled band values per point)
Checks: row_count_matches, schema_complete, crs_valid
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class SampleRasterParams(OperatorParams):
    """Parameters for raster point sampling."""

    output_path: str | None = None
    bands: list[int] = field(default_factory=list)  # empty = all bands
    nodata_value: float | None = None  # override; None = use raster native


class SampleRasterOperator:
    """Sample raster cell values at point locations.

    Input 0: raster artifact
    Input 1: vector artifact (point geometries)

    Output: table artifact (CSV) with one row per point and one column per
    requested band.  Points outside the raster extent or landing on nodata
    pixels get NaN for those bands.  Row count always equals input point count.
    """

    @property
    def name(self) -> str:
        return "sample_raster"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER, ArtifactType.VECTOR),
            output_type=ArtifactType.TABLE,
            min_inputs=2,
            max_inputs=2,
            resource_scale=ResourceScale.LIGHT,
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

        raster_crs = raster.spatial.crs
        vector_crs = vector.spatial.crs
        if raster_crs and vector_crs and raster_crs != vector_crs:
            errors.append(f"CRS mismatch: raster={raster_crs}, vector={vector_crs}")

        if not isinstance(params, SampleRasterParams):
            errors.append("Params must be SampleRasterParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        for b in params.bands:
            if b < 1:
                errors.append(f"Band index must be >= 1, got {b}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, SampleRasterParams):
            raise OperatorError(self.name, "Params must be SampleRasterParams")

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
                total_bands = src.count
                band_indices = params.bands if params.bands else list(range(1, total_bands + 1))
                nodata = params.nodata_value if params.nodata_value is not None else src.nodata
                raster_crs = str(src.crs) if src.crs else None

                rows: list[dict] = []

                with fiona.open(vector_artifact.backing.uri) as vec:
                    for i, feature in enumerate(vec):
                        geom = shape(feature["geometry"])
                        row = self._sample_point(
                            geom,
                            src,
                            band_indices,
                            nodata,
                            i,
                        )
                        rows.append(row)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Raster sampling failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        band_columns = [f"band_{b}" for b in band_indices]
        fieldnames = ["point_id", *band_columns]

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
                    "bands": band_indices,
                    "nodata_value": nodata,
                },
            ),
            metadata={
                "format": "csv",
                "band_columns": band_columns,
                "point_count": len(rows),
            },
        )

        checks = self._run_checks(output_artifact, inputs, rows, band_columns)

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
    def _sample_point(
        geom,
        src,
        band_indices: list[int],
        nodata,
        index: int,
    ) -> dict:
        """Sample raster at a single point geometry."""
        nan = float("nan")
        band_columns = [f"band_{b}" for b in band_indices]
        nan_row = {"point_id": str(index), **{col: nan for col in band_columns}}

        if geom.is_empty:
            return nan_row

        x, y = geom.x, geom.y

        # Check if point is within raster bounds
        bounds = src.bounds
        if x < bounds.left or x > bounds.right or y < bounds.bottom or y > bounds.top:
            return nan_row

        row_idx, col_idx = src.index(x, y)

        # Bounds check after pixel index conversion (edge cases)
        if row_idx < 0 or row_idx >= src.height or col_idx < 0 or col_idx >= src.width:
            return nan_row

        result = {"point_id": str(index)}
        for band_i, col_name in zip(band_indices, band_columns):
            val = float(
                src.read(band_i, window=((row_idx, row_idx + 1), (col_idx, col_idx + 1)))[0, 0]
            )
            if nodata is not None and ((math.isnan(nodata) and math.isnan(val)) or val == nodata):
                result[col_name] = nan
            else:
                result[col_name] = val
        return result

    def _run_checks(
        self,
        output: Artifact,
        inputs: list[Artifact],
        rows: list[dict],
        band_columns: list[str],
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
                    message=f"Output rows ({actual}) match input points ({expected})",
                )
            )
        elif expected is not None:
            results.append(
                CheckResult(
                    check_name="row_count_matches",
                    state=ValidationState.INVALID,
                    message=f"Output rows ({actual}) != input points ({expected})",
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
        expected_cols = {"point_id", *band_columns}
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
