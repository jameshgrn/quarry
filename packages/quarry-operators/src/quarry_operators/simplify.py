"""SimplifyOperator — vector geometry simplification via Douglas-Peucker.

Lane: operator
Accepts: one vector artifact
Produces: one vector artifact (GeoJSON) with simplified geometries
Algorithm: Douglas-Peucker (Shapely simplify)
Checks: crs_valid, feature_count_preserved, geometry_valid
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


@dataclass(frozen=True)
class SimplifyParams(OperatorParams):
    """Parameters for geometry simplification."""

    output_path: str | None = None
    tolerance: float = 0.0
    preserve_topology: bool = True


class SimplifyOperator:
    """Simplify vector geometries using Douglas-Peucker algorithm.

    Input 0: vector artifact to simplify

    Output: vector artifact (GeoJSON) with simplified geometries.
    Feature count is always preserved — simplify never drops features.
    Points pass through unchanged (simplify is a no-op on points).
    Empty geometries stay empty.
    """

    @property
    def name(self) -> str:
        return "simplify"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.VECTOR,),
            output_type=ArtifactType.VECTOR,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.LIGHT,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 1:
            errors.append(f"Exactly 1 input required, got {len(inputs)}")
            return errors

        artifact = inputs[0]

        if artifact.type != ArtifactType.VECTOR:
            errors.append(f"Input must be vector, got {artifact.type.value}")

        if not artifact.is_materialized:
            errors.append("Input is not materialized")

        if not isinstance(params, SimplifyParams):
            errors.append("Params must be SimplifyParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        if params.tolerance < 0:
            errors.append(f"tolerance must be >= 0, got {params.tolerance}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, SimplifyParams):
            raise OperatorError(self.name, "Params must be SimplifyParams")

        import time

        import fiona
        from shapely.geometry import mapping, shape

        t0 = time.monotonic()

        input_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        collapsed_count = 0

        try:
            with fiona.open(input_artifact.backing.uri) as src:
                src_schema = dict(src.schema)
                src_crs = src.crs
                input_count = len(src)

                with fiona.open(
                    output_path,
                    "w",
                    driver="GeoJSON",
                    crs=src_crs,
                    schema=src_schema,
                ) as dst:
                    for feat in src:
                        if feat["geometry"] is None:
                            dst.write(feat)
                            continue

                        geom = shape(feat["geometry"])

                        if geom.is_empty:
                            dst.write(feat)
                            continue

                        simplified = geom.simplify(
                            params.tolerance,
                            preserve_topology=params.preserve_topology,
                        )

                        if simplified.is_empty:
                            collapsed_count += 1

                        out_feat = {
                            "geometry": mapping(simplified),
                            "properties": dict(feat.get("properties", {})),
                        }
                        dst.write(out_feat)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Simplification failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        elapsed = time.monotonic() - t0

        # Read back output for fresh metadata
        with fiona.open(output_path) as out_src:
            out_count = len(out_src)
            out_crs = str(out_src.crs) if out_src.crs else None
            if out_count > 0:
                out_bounds = out_src.bounds
                extent = (out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3])
            else:
                extent = None

        output_artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=output_path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs=out_crs,
                extent=extent,
                feature_count=out_count,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in inputs),
                params={
                    "tolerance": params.tolerance,
                    "preserve_topology": params.preserve_topology,
                },
            ),
            metadata={
                "format": "geojson",
                "input_feature_count": input_count,
                "output_feature_count": out_count,
                "collapsed_count": collapsed_count,
            },
        )

        checks = self._run_checks(output_artifact, input_count, out_count, collapsed_count)

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "feature_count_preserved", "geometry_valid"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        input_count: int,
        output_count: int,
        collapsed_count: int,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

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

        # Feature count preserved: simplify never drops features
        if output_count == input_count:
            results.append(
                CheckResult(
                    check_name="feature_count_preserved",
                    state=ValidationState.VALID,
                    message=f"Feature count preserved: {output_count}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="feature_count_preserved",
                    state=ValidationState.INVALID,
                    message=(f"Feature count mismatch: input={input_count}, output={output_count}"),
                )
            )

        # Geometry valid: warn if any non-empty geometries collapsed to empty
        if collapsed_count == 0:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.VALID,
                    message="No geometries collapsed to empty",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.WARN,
                    message=(
                        f"{collapsed_count} non-empty geometries collapsed to empty "
                        f"at tolerance={output.lineage.params.get('tolerance')}"
                    ),
                )
            )

        return results
