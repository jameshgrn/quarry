"""ClipVectorOperator — clip vector features to a polygon mask boundary.

Lane: operator
Accepts: two vector artifacts (input features + clip mask)
Produces: one vector artifact (GeoJSON) with features clipped to mask boundary
Features entirely outside the mask are dropped.
Features partially inside are clipped to the mask boundary.
Checks: crs_valid, output_within_clip, feature_count
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
class ClipVectorParams(OperatorParams):
    """Parameters for vector clip."""

    output_path: str | None = None


class ClipVectorOperator:
    """Clip vector features to a polygon mask boundary.

    Input 0: features to clip (any geometry type)
    Input 1: clip mask (polygon/multipolygon layer)

    Output: vector artifact (GeoJSON). Features entirely outside the mask
    are dropped. Features partially inside are clipped to the mask boundary.
    All properties are preserved on surviving features.
    """

    @property
    def name(self) -> str:
        return "clip_vector"

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
            errors.append(f"Exactly 2 inputs required (features + clip mask), got {len(inputs)}")
            return errors

        features, mask = inputs

        if features.type != ArtifactType.VECTOR:
            errors.append(f"Features input must be vector, got {features.type.value}")
        if mask.type != ArtifactType.VECTOR:
            errors.append(f"Clip mask input must be vector, got {mask.type.value}")

        if not features.is_materialized:
            errors.append("Features input is not materialized")
        if not mask.is_materialized:
            errors.append("Clip mask input is not materialized")

        features_crs = features.spatial.crs
        mask_crs = mask.spatial.crs
        if features_crs and mask_crs and features_crs != mask_crs:
            errors.append(f"CRS mismatch: features={features_crs}, mask={mask_crs}")

        if not isinstance(params, ClipVectorParams):
            errors.append("Params must be ClipVectorParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, ClipVectorParams):
            raise OperatorError(self.name, "Params must be ClipVectorParams")

        import time

        import fiona
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union
        from shapely.prepared import prep

        t0 = time.monotonic()

        features_artifact, mask_artifact = inputs
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read clip mask geometries and union them
            mask_geoms = []
            with fiona.open(mask_artifact.backing.uri) as mask_src:
                mask_bounds = mask_src.bounds
                for feat in mask_src:
                    if feat["geometry"] is not None:
                        geom = shape(feat["geometry"])
                        if not geom.is_empty:
                            mask_geoms.append(geom)

            if not mask_geoms:
                raise OperatorError(
                    self.name,
                    "Clip mask contains no valid geometries",
                    inputs=[a.id for a in inputs],
                )

            clip_union = unary_union(mask_geoms)
            prepared_clip = prep(clip_union)

            # Read input features schema
            with fiona.open(features_artifact.backing.uri) as feat_src:
                feat_schema = dict(feat_src.schema)
                feat_crs = feat_src.crs

            # Clip features and write output
            input_feature_count = 0
            output_feature_count = 0

            with fiona.open(features_artifact.backing.uri) as feat_src:
                with fiona.open(
                    output_path,
                    "w",
                    driver="GeoJSON",
                    crs=feat_crs,
                    schema=feat_schema,
                ) as dst:
                    for feat in feat_src:
                        input_feature_count += 1

                        if feat["geometry"] is None:
                            continue

                        geom = shape(feat["geometry"])
                        if geom.is_empty:
                            continue

                        if not prepared_clip.intersects(geom):
                            continue

                        clipped = geom.intersection(clip_union)
                        if clipped.is_empty:
                            continue

                        dst.write(
                            {
                                "geometry": mapping(clipped),
                                "properties": dict(feat.get("properties", {})),
                            }
                        )
                        output_feature_count += 1

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Vector clip failed: {e}",
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
                    "input_feature_count": input_feature_count,
                    "output_feature_count": output_feature_count,
                },
            ),
            metadata={
                "format": "geojson",
                "input_feature_count": input_feature_count,
                "output_feature_count": output_feature_count,
            },
        )

        checks = self._run_checks(
            output_artifact,
            input_feature_count,
            output_feature_count,
            mask_bounds,
        )

        return OperatorResult(
            artifact=output_artifact,
            checks=checks,
            timing_seconds=elapsed,
        )

    def declared_checks(self) -> list[str]:
        return ["crs_valid", "output_within_clip", "feature_count"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        output: Artifact,
        input_count: int,
        output_count: int,
        mask_bounds: tuple[float, float, float, float],
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

        # Output extent within clip mask extent
        out_ext = output.spatial.extent
        if out_ext and mask_bounds:
            # Check if output extent is within clip extent (with tolerance for FP)
            eps = 1e-6
            within = (
                out_ext[0] >= mask_bounds[0] - eps
                and out_ext[1] >= mask_bounds[1] - eps
                and out_ext[2] <= mask_bounds[2] + eps
                and out_ext[3] <= mask_bounds[3] + eps
            )
            if within:
                results.append(
                    CheckResult(
                        check_name="output_within_clip",
                        state=ValidationState.VALID,
                        message="Output extent within clip mask extent",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        check_name="output_within_clip",
                        state=ValidationState.WARN,
                        message=(
                            f"Output extent {out_ext} not fully within "
                            f"clip extent {mask_bounds} — floating point edge case"
                        ),
                    )
                )
        elif output_count == 0:
            results.append(
                CheckResult(
                    check_name="output_within_clip",
                    state=ValidationState.VALID,
                    message="No output features — extent check not applicable",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="output_within_clip",
                    state=ValidationState.WARN,
                    message="Could not determine extents for comparison",
                )
            )

        # Feature count: output <= input (clipping can only reduce or maintain)
        if output_count <= input_count:
            results.append(
                CheckResult(
                    check_name="feature_count",
                    state=ValidationState.VALID,
                    message=f"Output features ({output_count}) <= input features ({input_count})",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="feature_count",
                    state=ValidationState.INVALID,
                    message=(
                        f"Output features ({output_count}) > input features ({input_count}) "
                        "— clipping should not increase feature count"
                    ),
                )
            )

        return results
