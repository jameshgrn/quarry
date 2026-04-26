"""BufferOperator — vector-to-vector geometry buffer.

Lane: operator
Accepts: one vector artifact
Produces: one vector artifact (GeoJSON) with every geometry buffered by distance
Distance: in CRS units; negative shrinks polygons (may collapse to empty)
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

_CAP_STYLES = {"round": 1, "flat": 2, "square": 3}
_JOIN_STYLES = {"round": 1, "mitre": 2, "bevel": 3}


@dataclass(frozen=True)
class BufferParams(OperatorParams):
    """Parameters for geometry buffer."""

    output_path: str | None = None
    distance: float = 0.0
    resolution: int = 16
    cap_style: str = "round"
    join_style: str = "round"


class BufferOperator:
    """Buffer every geometry in a vector dataset by a fixed distance.

    Input 0: vector artifact
    Output: vector artifact (GeoJSON) with buffered geometries.

    All feature properties are preserved unchanged. Feature count is
    preserved 1:1. Negative distance shrinks polygons and may produce
    empty geometries — that is valid behaviour.
    """

    @property
    def name(self) -> str:
        return "buffer"

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

        if not isinstance(params, BufferParams):
            errors.append("Params must be BufferParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        if params.distance == 0.0:
            errors.append("distance must not be zero")

        if params.cap_style not in _CAP_STYLES:
            errors.append(
                f"Invalid cap_style: {params.cap_style} (valid: {', '.join(_CAP_STYLES)})"
            )

        if params.join_style not in _JOIN_STYLES:
            errors.append(
                f"Invalid join_style: {params.join_style} (valid: {', '.join(_JOIN_STYLES)})"
            )

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, BufferParams):
            raise OperatorError(self.name, "Params must be BufferParams")

        import time

        import fiona
        from shapely.geometry import mapping, shape

        t0 = time.monotonic()

        input_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap_style_int = _CAP_STYLES[params.cap_style]
        join_style_int = _JOIN_STYLES[params.join_style]

        try:
            with fiona.open(input_artifact.backing.uri) as src:
                src_crs = src.crs
                src_schema_props = dict(src.schema.get("properties", {}))
                input_feature_count = len(src)

                # Buffer always produces Polygon regardless of input type
                # (point → circle, line → corridor, polygon → expanded polygon)
                out_schema = {
                    "geometry": "Polygon",
                    "properties": src_schema_props,
                }

                output_features: list[dict] = []
                for feat in src:
                    props = dict(feat.get("properties", {}))
                    geom = feat["geometry"]

                    if geom is None:
                        buffered_geom = None
                    else:
                        shp = shape(geom)
                        if shp.is_empty:
                            buffered = shp
                        else:
                            buffered = shp.buffer(
                                params.distance,
                                quad_segs=params.resolution,
                                cap_style=cap_style_int,
                                join_style=join_style_int,
                            )
                        buffered_geom = mapping(buffered) if not buffered.is_empty else None

                    output_features.append({"geometry": buffered_geom, "properties": props})

            with fiona.open(
                output_path,
                "w",
                driver="GeoJSON",
                crs=src_crs,
                schema=out_schema,
            ) as dst:
                for feat in output_features:
                    dst.write(feat)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Buffer failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        elapsed = time.monotonic() - t0

        # Read back output for fresh metadata
        with fiona.open(output_path) as out_src:
            try:
                out_bounds = out_src.bounds
            except Exception:
                out_bounds = None
            out_count = len(out_src)
            out_crs = str(out_src.crs) if out_src.crs else None

        # Count empty geometries in output
        empty_count = sum(1 for f in output_features if f["geometry"] is None)

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
                extent=(out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3])
                if out_bounds
                else None,
                feature_count=out_count,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in inputs),
                params={
                    "distance": params.distance,
                    "resolution": params.resolution,
                    "cap_style": params.cap_style,
                    "join_style": params.join_style,
                },
            ),
            metadata={
                "format": "geojson",
                "input_feature_count": input_feature_count,
                "output_feature_count": out_count,
                "empty_geometry_count": empty_count,
            },
        )

        checks = self._run_checks(
            output_artifact,
            input_feature_count,
            out_count,
            output_features,
        )

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
        output_features: list[dict],
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

        # Feature count preserved: output count must equal input count
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

        # Geometry valid: no empty geometries unless they were already empty
        # (negative buffer can collapse geometry to empty — that is acceptable)
        empty_geoms = [f for f in output_features if f["geometry"] is None]
        if not empty_geoms:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.VALID,
                    message="All output geometries are non-empty",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.WARN,
                    message=(
                        f"{len(empty_geoms)} empty geometries in output "
                        "(may result from negative buffer distance)"
                    ),
                )
            )

        return results
