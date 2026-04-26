"""DissolveOperator — vector-to-vector geometry dissolve.

Lane: operator
Accepts: one vector artifact
Produces: one vector artifact (GeoJSON) with geometries merged per group
Grouping: by attribute field (or all features if no field specified)
Merge: unary_union per group
Checks: crs_valid, feature_count_reduced, geometry_valid
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
class DissolveParams(OperatorParams):
    """Parameters for dissolve."""

    output_path: str | None = None
    by: str | None = None


class DissolveOperator:
    """Dissolve vector features by grouping on an attribute field.

    If ``by`` is None, all features are dissolved into a single geometry.
    If ``by`` is set, features are grouped by that property value and each
    group is dissolved independently.  Features missing the grouping field
    are collected into a ``__null__`` group.

    Output properties per group:
    - the ``by`` field value (when grouping)
    - ``_dissolved_count``: number of input features merged
    """

    @property
    def name(self) -> str:
        return "dissolve"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.VECTOR,),
            output_type=ArtifactType.VECTOR,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
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

        if not isinstance(params, DissolveParams):
            errors.append("Params must be DissolveParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, DissolveParams):
            raise OperatorError(self.name, "Params must be DissolveParams")

        import time

        import fiona
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union

        t0 = time.monotonic()

        input_artifact = inputs[0]
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read input features and group them
            groups: dict[str, list[dict]] = {}
            input_feature_count = 0
            input_crs = None

            with fiona.open(input_artifact.backing.uri) as src:
                input_crs = src.crs
                for feat in src:
                    input_feature_count += 1
                    if params.by is None:
                        key = "__all__"
                    else:
                        props = feat.get("properties", {})
                        val = props.get(params.by)
                        key = str(val) if val is not None else "__null__"
                    groups.setdefault(key, []).append(feat)

            # Dissolve each group
            output_features: list[dict] = []
            had_empty_union = False

            for key, features in groups.items():
                geometries = [shape(f["geometry"]) for f in features if f["geometry"] is not None]

                if not geometries:
                    had_empty_union = True
                    continue

                merged = unary_union(geometries)

                if merged.is_empty:
                    # All input geometries in group were empty — allowed
                    had_empty_union = True
                    continue

                props: dict = {"_dissolved_count": len(features)}
                if params.by is not None:
                    props[params.by] = (
                        None if key == "__null__" else features[0]["properties"].get(params.by)
                    )

                output_features.append(
                    {
                        "geometry": mapping(merged),
                        "properties": props,
                    }
                )

            # Determine output geometry type from results
            if output_features:
                geom_types = {f["geometry"]["type"] for f in output_features}
                # Normalize: if mix of Polygon and MultiPolygon, use MultiPolygon
                if geom_types == {"Polygon", "MultiPolygon"} or geom_types == {"MultiPolygon"}:
                    out_geom_type = "MultiPolygon"
                elif geom_types == {"LineString", "MultiLineString"} or geom_types == {
                    "MultiLineString"
                }:
                    out_geom_type = "MultiLineString"
                elif geom_types == {"Point", "MultiPoint"} or geom_types == {"MultiPoint"}:
                    out_geom_type = "MultiPoint"
                elif len(geom_types) == 1:
                    out_geom_type = geom_types.pop()
                else:
                    # Mixed types — use GeometryCollection
                    out_geom_type = "GeometryCollection"
            else:
                out_geom_type = "Polygon"

            # Build output schema
            out_schema_props: dict[str, str] = {"_dissolved_count": "int"}
            if params.by is not None:
                out_schema_props[params.by] = "str"

            out_schema = {
                "geometry": out_geom_type,
                "properties": out_schema_props,
            }

            # Write output GeoJSON
            with fiona.open(
                output_path,
                "w",
                driver="GeoJSON",
                crs=input_crs,
                schema=out_schema,
            ) as dst:
                for feat in output_features:
                    dst.write(feat)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Dissolve failed: {e}",
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
                params={"by": params.by},
            ),
            metadata={
                "format": "geojson",
                "input_feature_count": input_feature_count,
                "output_feature_count": out_count,
            },
        )

        checks = self._run_checks(
            output_artifact,
            input_feature_count,
            out_count,
            had_empty_union,
        )

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "feature_count_reduced", "geometry_valid"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        input_count: int,
        output_count: int,
        had_empty_union: bool,
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

        # Feature count reduced: dissolve should reduce or maintain count
        if output_count <= input_count:
            results.append(
                CheckResult(
                    check_name="feature_count_reduced",
                    state=ValidationState.VALID,
                    message=f"Output features ({output_count}) <= input features ({input_count})",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="feature_count_reduced",
                    state=ValidationState.INVALID,
                    message=(
                        f"Output features ({output_count}) > input features ({input_count}) "
                        "— dissolve should not increase feature count"
                    ),
                )
            )

        # Geometry valid: no empty union results (unless all input geometries were empty)
        if not had_empty_union:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.VALID,
                    message="All groups produced non-empty geometries",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="geometry_valid",
                    state=ValidationState.WARN,
                    message="One or more groups had empty or null geometries and were omitted",
                )
            )

        return results
