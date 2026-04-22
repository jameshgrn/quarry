"""SpatialJoinOperator — vector-to-vector spatial join.

Lane: operator
Accepts: two vector artifacts (left, right)
Produces: one vector artifact (GeoJSON) with left geometry + joined attributes
Predicate: intersects (v1)
Join type: left join — all left features preserved; unmatched get null right attrs
Schema collision: right columns colliding with left get '_right' suffix
Checks: crs_valid, left_features_preserved, schema_no_collision
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
class SpatialJoinParams(OperatorParams):
    """Parameters for spatial join."""

    output_path: str | None = None
    predicate: str = "intersects"


class SpatialJoinOperator:
    """Join two vector datasets by spatial predicate.

    Input 0: left vector (geometry preserved in output)
    Input 1: right vector (attributes joined where predicate holds)

    Output: vector artifact (GeoJSON). Left join semantics — every left
    feature appears at least once. One-to-many: left feature duplicated
    per matching right feature. No match: right attributes are None.
    """

    @property
    def name(self) -> str:
        return "spatial_join"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.VECTOR, ArtifactType.VECTOR),
            output_type=ArtifactType.VECTOR,
            min_inputs=2,
            max_inputs=2,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors: list[str] = []

        if len(inputs) != 2:
            errors.append(f"Exactly 2 inputs required (left + right vector), got {len(inputs)}")
            return errors

        left, right = inputs

        if left.type != ArtifactType.VECTOR:
            errors.append(f"Left input must be vector, got {left.type.value}")
        if right.type != ArtifactType.VECTOR:
            errors.append(f"Right input must be vector, got {right.type.value}")

        if not left.is_materialized:
            errors.append("Left input is not materialized")
        if not right.is_materialized:
            errors.append("Right input is not materialized")

        left_crs = left.spatial.crs
        right_crs = right.spatial.crs
        if left_crs and right_crs and left_crs != right_crs:
            errors.append(f"CRS mismatch: left={left_crs}, right={right_crs}")

        if not isinstance(params, SpatialJoinParams):
            errors.append("Params must be SpatialJoinParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        if params.predicate != "intersects":
            errors.append(f"Unsupported predicate: {params.predicate} (v1 supports: intersects)")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, SpatialJoinParams):
            raise OperatorError(self.name, "Params must be SpatialJoinParams")

        import time

        import fiona
        from shapely.geometry import shape
        from shapely.prepared import prep

        t0 = time.monotonic()

        left_artifact, right_artifact = inputs
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read right features into memory (spatial index side)
            right_features: list[tuple[object, dict]] = []
            right_schema_props: dict[str, str] = {}
            with fiona.open(right_artifact.backing.uri) as right_src:
                right_schema_props = dict(right_src.schema.get("properties", {}))
                for feat in right_src:
                    geom = shape(feat["geometry"]) if feat["geometry"] else None
                    right_features.append((geom, dict(feat.get("properties", {}))))

            # Read left schema
            with fiona.open(left_artifact.backing.uri) as left_src:
                left_schema_props = dict(left_src.schema.get("properties", {}))
                left_crs = left_src.crs
                left_geom_type = left_src.schema["geometry"]

            # Resolve schema collisions: right columns that collide get '_right' suffix
            collision_renames: dict[str, str] = {}
            merged_schema_props = dict(left_schema_props)
            for rkey, rtype in right_schema_props.items():
                if rkey in merged_schema_props:
                    new_key = f"{rkey}_right"
                    collision_renames[rkey] = new_key
                    merged_schema_props[new_key] = rtype
                else:
                    merged_schema_props[rkey] = rtype

            # Build output schema
            out_schema = {
                "geometry": left_geom_type,
                "properties": merged_schema_props,
            }

            # Null right properties template
            null_right: dict[str, None] = {}
            for rkey in right_schema_props:
                out_key = collision_renames.get(rkey, rkey)
                null_right[out_key] = None

            # Perform spatial join and write output
            left_feature_count = 0
            output_feature_count = 0
            had_collision = bool(collision_renames)

            with fiona.open(left_artifact.backing.uri) as left_src:
                with fiona.open(
                    output_path,
                    "w",
                    driver="GeoJSON",
                    crs=left_crs,
                    schema=out_schema,
                ) as dst:
                    for left_feat in left_src:
                        left_feature_count += 1
                        left_geom = shape(left_feat["geometry"]) if left_feat["geometry"] else None
                        left_props = dict(left_feat.get("properties", {}))

                        matches = []
                        if left_geom is not None and not left_geom.is_empty:
                            prepared = prep(left_geom)
                            for right_geom, right_props in right_features:
                                if right_geom is None or right_geom.is_empty:
                                    continue
                                if prepared.intersects(right_geom):
                                    matches.append(right_props)

                        if not matches:
                            # No match — emit left feature with null right attrs
                            merged = {**left_props, **null_right}
                            dst.write(
                                {
                                    "geometry": left_feat["geometry"],
                                    "properties": merged,
                                }
                            )
                            output_feature_count += 1
                        else:
                            # One or more matches — emit one row per match
                            for right_props in matches:
                                renamed_right = {}
                                for rkey, rval in right_props.items():
                                    out_key = collision_renames.get(rkey, rkey)
                                    renamed_right[out_key] = rval
                                # Fill missing right keys with None
                                for nk, nv in null_right.items():
                                    if nk not in renamed_right:
                                        renamed_right[nk] = nv
                                merged = {**left_props, **renamed_right}
                                dst.write(
                                    {
                                        "geometry": left_feat["geometry"],
                                        "properties": merged,
                                    }
                                )
                                output_feature_count += 1

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Spatial join failed: {e}",
                inputs=[a.id for a in inputs],
            ) from e

        elapsed = time.monotonic() - t0

        # Read back output for fresh metadata
        with fiona.open(output_path) as out_src:
            out_bounds = out_src.bounds
            out_count = len(out_src)
            out_crs = str(out_src.crs) if out_src.crs else None

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
                extent=(out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3]),
                feature_count=out_count,
            ),
            lineage=Lineage(
                operation=self.name,
                inputs=tuple(a.id for a in inputs),
                params={
                    "predicate": params.predicate,
                    "collision_renames": collision_renames,
                },
            ),
            metadata={
                "format": "geojson",
                "left_feature_count": left_feature_count,
                "output_feature_count": output_feature_count,
                "collision_renames": collision_renames,
            },
        )

        checks = self._run_checks(
            output_artifact,
            left_feature_count,
            output_feature_count,
            had_collision,
            collision_renames,
        )

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "left_features_preserved", "schema_no_collision"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        left_count: int,
        output_count: int,
        had_collision: bool,
        collision_renames: dict[str, str],
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

        # Left features preserved: output count >= left count (left join guarantee)
        if output_count >= left_count:
            results.append(
                CheckResult(
                    check_name="left_features_preserved",
                    state=ValidationState.VALID,
                    message=(f"Output features ({output_count}) >= left input ({left_count})"),
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="left_features_preserved",
                    state=ValidationState.INVALID,
                    message=(
                        f"Output features ({output_count}) < left input ({left_count}) "
                        "— left join violated"
                    ),
                )
            )

        # Schema collision warning
        if had_collision:
            results.append(
                CheckResult(
                    check_name="schema_no_collision",
                    state=ValidationState.WARN,
                    message=f"Schema collisions resolved by rename: {collision_renames}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="schema_no_collision",
                    state=ValidationState.VALID,
                    message="No schema collisions",
                )
            )

        return results
