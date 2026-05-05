"""Quarry CLI — minimal invocation surface over the substrate.

Lane: adapter
Exposes registry queries and flow execution as shell commands.
No workflow engine, no config files, no plugin system.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from quarry_connectors.router import build_default_router
from quarry_core.executor import RunRecord
from quarry_core.operator import OperatorParams, OperatorResult
from quarry_core.router import ConnectorRouter, NoConnectorError
from quarry_core.source_ref import RASTER_EXTENSIONS, VECTOR_EXTENSIONS, SourceRef, SourceRefKind

_LOCAL_SOURCE_KINDS = {
    SourceRefKind.LOCAL_PATH,
    SourceRefKind.LOCAL_RASTER,
    SourceRefKind.LOCAL_VECTOR,
}
_LOCAL_EXTENSIONS = RASTER_EXTENSIONS | VECTOR_EXTENSIONS


def _existing_local_source_ref(path_part: str, sep: str, suffix: str, label: str) -> SourceRef:
    resolved = Path(path_part).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} file not found: {resolved}")
    ref_raw = f"{resolved}{sep}{suffix}" if sep else str(resolved)
    return SourceRef.local(ref_raw)


def _get_router() -> ConnectorRouter:
    """Create the default CLI router for common source references."""
    stac_api = os.environ.get("STAC_API_URL", "https://earth-search.aws.element84.com/v0")
    return build_default_router(stac_api_url=stac_api)


def _resolve_workspace(args) -> Path:
    return Path(args.workspace).resolve()


def _source_ref_from_cli(raw: str, label: str) -> SourceRef:
    stripped = raw.strip()
    path_part, sep, suffix = stripped.partition("::")
    path = Path(path_part).expanduser()
    extension = path.suffix.lower()
    explicit_local = path_part.startswith(("/", "./", "../", "~"))
    extension_local = extension in _LOCAL_EXTENSIONS

    if explicit_local or extension_local:
        return _existing_local_source_ref(path_part, sep, suffix, label)

    inferred = SourceRef.infer(stripped)
    if inferred.kind in _LOCAL_SOURCE_KINDS:
        return _existing_local_source_ref(path_part, sep, suffix, label)

    if inferred.kind == SourceRefKind.UNKNOWN and path.exists():
        return _existing_local_source_ref(path_part, sep, suffix, label)

    return inferred


def _materialize_cli_source(
    router: ConnectorRouter,
    raw: str,
    workspace: Path,
    *,
    label: str,
    quiet: bool = False,
):
    source_ref = _source_ref_from_cli(raw, label)
    try:
        match = router.select_one(source_ref)
    except NoConnectorError as e:
        raise ValueError(str(e)) from e

    if not quiet:
        print(f"Materializing {label.lower()}: {source_ref.raw}")
    return match.connector.materialize(source_ref.raw, workspace).artifact


def _handle_run_failure(run_record) -> int:
    """Render a failed single-step run consistently for CLI adapters."""
    if run_record.status.value != "completed":
        message = run_record.error or f"{run_record.operator_name} did not complete"
        print(f"FAILED: {message}", file=sys.stderr)
        return 1
    if run_record.output is None:
        print(f"FAILED: {run_record.operator_name} completed without output", file=sys.stderr)
        return 1
    return 0


def _handle_invalid_checks(checks, subject: str) -> int:
    """Render semantically invalid output consistently for CLI adapters."""
    invalid = [c for c in checks if c.state.value == "invalid"]
    if not invalid:
        return 0
    print(f"FAILED: {subject} produced {len(invalid)} invalid check(s)", file=sys.stderr)
    for c in invalid:
        print(f"  [{c.check_name}] {c.message}", file=sys.stderr)
    return 2


def _require_run_output(run_record: RunRecord) -> OperatorResult:
    if run_record.output is None:
        raise RuntimeError(f"{run_record.operator_name} completed without output")
    return run_record.output


def _json_default(obj):
    """Custom JSON encoder that converts Mapping types to dicts."""
    if isinstance(obj, MappingProxyType):
        return dict(obj)
    if isinstance(obj, Mapping) and not isinstance(obj, dict):
        return dict(obj)
    return str(obj)


def _emit_json(data) -> None:
    """Print one JSON object to stdout. Used when args.json_output is True."""
    print(json.dumps(data, default=_json_default, sort_keys=True))


# ---------------------------------------------------------------------------
# artifacts list
# ---------------------------------------------------------------------------


def cmd_artifacts_list(args) -> int:
    from quarry_core.artifact import ArtifactType
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    type_filter = ArtifactType(args.type) if args.type else None
    artifacts = registry.list_artifacts(artifact_type=type_filter, limit=args.limit)

    if args.json_output:
        data = [
            {
                "id": a.id,
                "name": a.name,
                "type": a.type.value,
                "uri": a.backing.uri if a.backing else None,
                "created_at": a.created_at,
            }
            for a in artifacts
        ]
        _emit_json(data)
        return 0

    if not artifacts:
        print("No artifacts found.")
        return 0

    # Header
    print(f"{'ID':<38} {'TYPE':<10} {'NAME':<30} {'CRS':<12} {'CREATED'}")
    print("-" * 110)
    for a in artifacts:
        created = a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "?"
        crs = a.spatial.crs or "-"
        print(f"{a.id:<38} {a.type.value:<10} {a.name:<30} {crs:<12} {created}")

    print(f"\n{len(artifacts)} artifact(s)")
    return 0


# ---------------------------------------------------------------------------
# artifacts show
# ---------------------------------------------------------------------------


def cmd_artifacts_show(args) -> int:
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    artifact = registry.get_artifact(args.artifact_id)

    if artifact is None:
        if args.json_output:
            _emit_json({"error": f"Artifact not found: {args.artifact_id}"})
        else:
            print(f"Artifact not found: {args.artifact_id}", file=sys.stderr)
        return 1

    if args.json_output:
        data = {
            "id": artifact.id,
            "name": artifact.name,
            "type": artifact.type.value,
            "uri": artifact.backing.uri if artifact.backing else None,
            "size_bytes": artifact.backing.size_bytes if artifact.backing else None,
            "content_hash": artifact.backing.content_hash if artifact.backing else None,
            "spatial": {
                "crs": artifact.spatial.crs,
                "extent": artifact.spatial.extent,
                "resolution": artifact.spatial.resolution,
                "band_count": artifact.spatial.band_count,
                "feature_count": artifact.spatial.feature_count,
            },
            "metadata": artifact.metadata,
            "lineage": {
                "parent_ids": artifact.parent_ids,
                "run_id": artifact.run_id,
            },
        }
        _emit_json(data)
        return 0

    print(f"ID:       {artifact.id}")
    print(f"Type:     {artifact.type.value}")
    print(f"Name:     {artifact.name}")
    print(f"Created:  {artifact.created_at}")

    if artifact.backing:
        print(f"Backing:  {artifact.backing.kind.value} @ {artifact.backing.uri}")
        if artifact.backing.size_bytes is not None:
            print(f"Size:     {artifact.backing.size_bytes:,} bytes")

    s = artifact.spatial
    if s.crs:
        print(f"CRS:      {s.crs}")
    if s.extent:
        print(f"Extent:   {s.extent}")
    if s.resolution:
        print(f"Res:      {s.resolution}")
    if s.band_count is not None:
        print(f"Bands:    {s.band_count}")
    if s.feature_count is not None:
        print(f"Features: {s.feature_count}")

    if artifact.checks:
        print(f"\nChecks ({len(artifact.checks)}):")
        for c in artifact.checks:
            print(f"  [{c.state.value:>7}] {c.check_name}: {c.message}")

    if artifact.metadata:
        print(f"\nMetadata: {artifact.metadata}")

    return 0


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def cmd_lineage(args) -> int:
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))

    # Verify the artifact exists
    artifact = registry.get_artifact(args.artifact_id)
    if artifact is None:
        if args.json_output:
            _emit_json({"error": f"Artifact not found: {args.artifact_id}"})
        else:
            print(f"Artifact not found: {args.artifact_id}", file=sys.stderr)
        return 1

    chain = registry.get_full_lineage(args.artifact_id)

    if args.json_output:
        data = [
            {
                "artifact_id": edge["artifact_id"],
                "name": edge["name"],
                "type": edge["type"],
                "operation": edge["operation"],
                "run_id": edge.get("run_id"),
            }
            for edge in chain
        ]
        _emit_json(data)
        return 0

    print(f"Lineage for: {artifact.name} ({artifact.id})")
    if not chain:
        print("  (no ancestors)")
        return 0

    print()
    for edge in chain:
        print(f"  {edge['name']} ({edge['type']})")
        print(f"    id:        {edge['artifact_id']}")
        print(f"    operation: {edge['operation']}")
        if edge.get("run_id"):
            print(f"    run:       {edge['run_id']}")
        print()

    print(f"{len(chain)} ancestor(s)")
    return 0


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


def cmd_runs_list(args) -> int:
    from quarry_core.executor import RunStatus
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    status_filter = RunStatus(args.status) if args.status else None
    runs = registry.list_runs(status=status_filter, limit=args.limit)

    if args.json_output:
        data = [
            {
                "id": r.id,
                "operator_name": r.operator_name,
                "status": r.status.value,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
            }
            for r in runs
        ]
        _emit_json(data)
        return 0

    if not runs:
        print("No runs found.")
        return 0

    print(f"{'ID':<38} {'OPERATOR':<25} {'STATUS':<12} {'SUBMITTED':<18} {'DURATION'}")
    print("-" * 110)
    for r in runs:
        submitted = r.submitted_at.strftime("%Y-%m-%d %H:%M") if r.submitted_at else "?"
        dur = f"{r.duration_seconds:.1f}s" if r.duration_seconds is not None else "-"
        print(f"{r.id:<38} {r.operator_name:<25} {r.status.value:<12} {submitted:<18} {dur}")

    print(f"\n{len(runs)} run(s)")
    return 0


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------


def cmd_runs_show(args) -> int:
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    run = registry.get_run(args.run_id)

    if run is None:
        if args.json_output:
            _emit_json({"error": f"Run not found: {args.run_id}"})
        else:
            print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 1

    if args.json_output:
        data = {
            "id": run.id,
            "operator_name": run.operator_name,
            "status": run.status.value,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "params": run.params,
            "input_ids": run.input_ids,
            "output_id": run.output.artifact.id if run.output else None,
            "error": run.error,
        }
        _emit_json(data)
        return 0

    print(f"ID:        {run.id}")
    print(f"Operator:  {run.operator_name}")
    print(f"Status:    {run.status.value}")
    print(f"Executor:  {run.executor_name or '-'}")
    print(f"Submitted: {run.submitted_at}")
    if run.started_at:
        print(f"Started:   {run.started_at}")
    if run.completed_at:
        print(f"Completed: {run.completed_at}")
    if run.duration_seconds is not None:
        print(f"Duration:  {run.duration_seconds:.2f}s")

    if run.input_ids:
        print(f"\nInputs ({len(run.input_ids)}):")
        for iid in run.input_ids:
            print(f"  {iid}")

    if run.params:
        print("\nParams:")
        for k, v in run.params.items():
            print(f"  {k}: {v}")

    if run.output:
        art = run.output.artifact
        print("\nOutput:")
        print(f"  {art.id}  {art.type.value}  {art.name}")

    if run.checks:
        print(f"\nChecks ({len(run.checks)}):")
        for c in run.checks:
            print(f"  [{c.state.value:>7}] {c.check_name}: {c.message}")

    if run.error:
        print(f"\nError: {run.error}")

    return 0


# ---------------------------------------------------------------------------
# checks show
# ---------------------------------------------------------------------------


def cmd_checks_show(args) -> int:
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    target_id = args.id

    # Try as artifact first, then as run
    artifact = registry.get_artifact(target_id)
    run = registry.get_run(target_id)

    if artifact is None and run is None:
        if args.json_output:
            _emit_json({"error": f"No artifact or run found for: {target_id}"})
        else:
            print(f"No artifact or run found for: {target_id}", file=sys.stderr)
        return 1

    checks = registry.get_checks(
        artifact_id=target_id if artifact else None,
        run_id=target_id if run else None,
    )

    if args.json_output:
        data = [
            {
                "check_name": c.check_name,
                "state": c.state.value,
                "message": c.message,
                "metadata": c.metadata,
            }
            for c in checks
        ]
        _emit_json(data)
        return 0

    if artifact is not None:
        label = f"artifact {artifact.name}"
    else:
        assert run is not None
        label = f"run {run.operator_name}"
    print(f"Checks for {label} ({target_id}):")

    if not checks:
        print("  (no checks)")
        return 0

    print()
    for c in checks:
        ts = c.timestamp.strftime("%Y-%m-%d %H:%M") if c.timestamp else "?"
        print(f"  [{c.state.value:>7}] {c.check_name}")
        print(f"           {c.message}")
        print(f"           {ts}")
        print()

    print(f"{len(checks)} check(s)")
    return 0


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------


def cmd_route(args) -> int:
    """Show inferred SourceRef and ranked connector matches for a source string."""
    from quarry_core.router import RegistrationView

    # Build router
    router = _get_router()

    # Resolve source via inference
    raw = args.source
    try:
        ref = SourceRef.infer(raw)
    except Exception as e:
        if args.json_output:
            _emit_json({"error": str(e)})
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Get matches
    matches = router.select(ref)

    # Build registration lookup for policy surface details
    reg_views = list(router.registrations)

    def find_view(connector_name: str, kind) -> RegistrationView | None:
        """Find the registration view that matches both connector and kind."""
        for view in reg_views:
            if view.connector_name == connector_name and kind in view.kinds:
                return view
        # Fallback: first view with matching connector name
        for view in reg_views:
            if view.connector_name == connector_name:
                return view
        return None

    if args.json_output:
        matches_data = []
        for match in matches:
            view = find_view(match.connector.name, ref.kind)
            match_dict = {
                "connector_name": match.connector.name,
                "reason": match.reason.value,
                "rank": match.rank,
                "kinds": [k.value for k in view.kinds] if view and view.kinds else [],
                "extensions": list(view.extensions) if view and view.extensions else [],
                "schemes": list(view.schemes) if view and view.schemes else [],
                "prefixes": list(view.prefixes) if view and view.prefixes else [],
            }
            matches_data.append(match_dict)
        data = {
            "source": {
                "raw": raw,
                "kind": ref.kind.value,
                "params": ref.params,
            },
            "matches": matches_data,
            "selected": matches[0].connector.name if matches else None,
        }
        _emit_json(data)
        return 0 if matches else 2

    # Text mode rendering below

    # Print Source section
    print("Source")
    print("------")
    print(f"  raw: {raw}")
    print(f"  kind: {ref.kind.value}")
    if ref.params:
        print(f"  params: {ref.params}")
    print()

    # Print Matches section
    print("Matches")
    print("-------")
    if not matches:
        print("  (none)")
    else:
        for match in matches:
            view = find_view(match.connector.name, ref.kind)
            if view is None:
                kinds_str = "-"
                ext_str = "-"
                scheme_str = "-"
                prefix_str = "-"
            else:
                kinds_str = ",".join(sorted(k.value for k in view.kinds)) if view.kinds else "-"
                ext_str = ",".join(sorted(view.extensions)) if view.extensions else "-"
                scheme_str = ",".join(sorted(view.schemes)) if view.schemes else "-"
                prefix_str = ",".join(sorted(view.prefixes)) if view.prefixes else "-"
            print(
                f"  [{match.rank}] {match.connector.name} ({match.reason.value}) — "
                f"kinds={kinds_str}, ext={ext_str}, scheme={scheme_str}, prefix={prefix_str}"
            )
    print()

    # Print Selected section
    print("Selected")
    print("--------")
    if matches:
        print(f"  {matches[0].connector.name}")
    else:
        print("  (no connector — NoConnectorError would be raised at materialize time)")

    return 0 if matches else 2


# ---------------------------------------------------------------------------
# run hydrology
# ---------------------------------------------------------------------------


def cmd_run_hydrology(args) -> int:
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.hydrology_flow import HydrologyFlow, HydrologyFlowParams
    from quarry_registry.registry import Registry

    workspace = _resolve_workspace(args)

    # Materialize DEM through router
    router = _get_router()
    try:
        dem_artifact = _materialize_cli_source(
            router, args.dem, workspace, label="DEM", quiet=args.json_output
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Set up executor + registry + flow
    executor = LocalExecutor()
    registry = Registry(workspace)
    flow = HydrologyFlow(executor=executor, registry=registry)

    hydro_dir = workspace / "hydrology"
    params = HydrologyFlowParams(
        workspace=hydro_dir,
        nodata=args.nodata,
        apply_gradient=not args.no_gradient,
        weight=args.weight,
    )

    if not args.json_output:
        print(f"Running hydrology flow → {hydro_dir}")
    flow_result = flow.run(dem_artifact, params)

    if not flow_result.success:
        if args.json_output:
            valid = sum(1 for c in flow_result.all_checks if c.state.value == "valid")
            invalid = sum(1 for c in flow_result.all_checks if c.state.value == "invalid")
            warning = sum(1 for c in flow_result.all_checks if c.state.value == "warning")
            data = {
                "operator_name": "hydrology",
                "status": "failed",
                "run_id": None,
                "output": None,
                "checks": {"valid": valid, "invalid": invalid, "warning": warning},
                "error": flow_result.error or f"Failed at step: {flow_result.failed_step}",
            }
            _emit_json(data)
        else:
            print(f"FAILED at step: {flow_result.failed_step}", file=sys.stderr)
            print(f"Error: {flow_result.error}", file=sys.stderr)
        return 1

    invalid_rc = _handle_invalid_checks(flow_result.all_checks, "hydrology")
    if invalid_rc and not args.json_output:
        return invalid_rc

    # Build JSON response or text report
    valid = sum(1 for c in flow_result.all_checks if c.state.value == "valid")
    invalid = sum(1 for c in flow_result.all_checks if c.state.value == "invalid")
    warning = sum(1 for c in flow_result.all_checks if c.state.value == "warning")

    if args.json_output:
        # Hydrology produces multiple artifacts - use the last run's output if available
        final_run = flow_result.runs[-1] if flow_result.runs else None
        output_data = None
        if final_run and final_run.output:
            art = final_run.output.artifact
            output_data = {
                "name": art.name,
                "uri": art.backing.uri if art.backing else None,
                "artifact_id": art.id,
            }
        data = {
            "operator_name": "hydrology",
            "status": "completed",
            "run_id": final_run.id if final_run else None,
            "output": output_data,
            "checks": {"valid": valid, "invalid": invalid, "warning": warning},
            "error": None,
        }
        _emit_json(data)
        return 0 if not invalid_rc else 2

    # Report results (text mode)
    print(f"\nCompleted ({len(flow_result.runs)} steps, {len(flow_result.all_checks)} checks)")
    for a in flow_result.artifacts:
        uri = a.backing.uri if a.backing else "?"
        print(f"  {a.name:<25} → {uri}")

    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# run zonal
# ---------------------------------------------------------------------------


def cmd_run_zonal(args) -> int:
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.zonal_stats import ZonalStatsOperator, ZonalStatsParams
    from quarry_registry.registry import Registry

    workspace = _resolve_workspace(args)

    # Materialize both inputs through router
    router = _get_router()
    try:
        raster_artifact = _materialize_cli_source(
            router, args.raster, workspace, label="Raster", quiet=args.json_output
        )
        zones_artifact = _materialize_cli_source(
            router, args.zones, workspace, label="Zones", quiet=args.json_output
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Set up executor + registry
    executor = LocalExecutor()
    registry = Registry(workspace)
    registry.save_artifact(raster_artifact)
    registry.save_artifact(zones_artifact)

    # Execute zonal stats
    output_dir = workspace / "zonal"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "zonal_stats.csv"

    params = ZonalStatsParams(
        output_path=str(output_path),
        band=args.band,
        zone_id_field=args.zone_id_field,
    )

    if not args.json_output:
        print(f"Running zonal stats → {output_path}")
    run_record = executor.submit(
        ZonalStatsOperator(),
        [raster_artifact, zones_artifact],
        params,
    )
    registry.save_run(run_record)

    failure_rc = _handle_run_failure(run_record)
    if failure_rc:
        if args.json_output:
            valid = sum(1 for c in run_record.checks if c.state.value == "valid")
            invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
            warning = sum(1 for c in run_record.checks if c.state.value == "warning")
            data = {
                "operator_name": run_record.operator_name,
                "status": "failed",
                "run_id": run_record.id,
                "output": None,
                "checks": {"valid": valid, "invalid": invalid, "warning": warning},
                "error": run_record.error or f"{run_record.operator_name} did not complete",
            }
            _emit_json(data)
        return 1

    invalid_rc = _handle_invalid_checks(run_record.checks, run_record.operator_name)
    if invalid_rc and not args.json_output:
        return invalid_rc

    # Report
    output = _require_run_output(run_record).artifact

    valid = sum(1 for c in run_record.checks if c.state.value == "valid")
    invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
    warning = sum(1 for c in run_record.checks if c.state.value == "warning")

    if args.json_output:
        data = {
            "operator_name": run_record.operator_name,
            "status": "completed",
            "run_id": run_record.id,
            "output": {
                "name": output.name,
                "uri": output.backing.uri if output.backing else None,
                "artifact_id": output.id,
            },
            "checks": {"valid": valid, "invalid": invalid, "warning": warning},
            "error": None,
        }
        _emit_json(data)
        return 0 if not invalid_rc else 2

    uri = output.backing.uri if output.backing else "?"
    print(f"\nCompleted (1 step, {len(run_record.checks)} checks)")
    print(f"  {output.name:<25} → {uri}")

    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# run sample
# ---------------------------------------------------------------------------


def cmd_run_sample(args) -> int:
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.sample_raster import SampleRasterOperator, SampleRasterParams
    from quarry_registry.registry import Registry

    workspace = _resolve_workspace(args)

    # Materialize both inputs through router
    router = _get_router()
    try:
        raster_artifact = _materialize_cli_source(
            router, args.raster, workspace, label="Raster", quiet=args.json_output
        )
        points_artifact = _materialize_cli_source(
            router, args.points, workspace, label="Points", quiet=args.json_output
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Set up executor + registry
    executor = LocalExecutor()
    registry = Registry(workspace)
    registry.save_artifact(raster_artifact)
    registry.save_artifact(points_artifact)

    # Parse bands
    bands: list[int] = []
    if args.bands:
        try:
            bands = [int(b.strip()) for b in args.bands.split(",")]
        except ValueError:
            print(
                f"Invalid --bands value: {args.bands!r} (expected comma-separated integers)",
                file=sys.stderr,
            )
            return 1

    # Output path
    output_dir = workspace / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else output_dir / "sample_raster.csv"

    params = SampleRasterParams(
        output_path=str(output_path),
        bands=bands,
        nodata_value=args.nodata,
    )

    if not args.json_output:
        print(f"Running sample raster → {output_path}")
    run_record = executor.submit(
        SampleRasterOperator(),
        [raster_artifact, points_artifact],
        params,
    )
    registry.save_run(run_record)

    failure_rc = _handle_run_failure(run_record)
    if failure_rc:
        if args.json_output:
            valid = sum(1 for c in run_record.checks if c.state.value == "valid")
            invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
            warning = sum(1 for c in run_record.checks if c.state.value == "warning")
            data = {
                "operator_name": run_record.operator_name,
                "status": "failed",
                "run_id": run_record.id,
                "output": None,
                "checks": {"valid": valid, "invalid": invalid, "warning": warning},
                "error": run_record.error or f"{run_record.operator_name} did not complete",
            }
            _emit_json(data)
        return 1

    invalid_rc = _handle_invalid_checks(run_record.checks, run_record.operator_name)
    if invalid_rc and not args.json_output:
        return invalid_rc

    # Report
    output = _require_run_output(run_record).artifact

    valid = sum(1 for c in run_record.checks if c.state.value == "valid")
    invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
    warning = sum(1 for c in run_record.checks if c.state.value == "warning")

    if args.json_output:
        data = {
            "operator_name": run_record.operator_name,
            "status": "completed",
            "run_id": run_record.id,
            "output": {
                "name": output.name,
                "uri": output.backing.uri if output.backing else None,
                "artifact_id": output.id,
            },
            "checks": {"valid": valid, "invalid": invalid, "warning": warning},
            "error": None,
        }
        _emit_json(data)
        return 0 if not invalid_rc else 2

    uri = output.backing.uri if output.backing else "?"
    print(f"\nCompleted (1 step, {len(run_record.checks)} checks)")
    print(f"  {output.name:<25} → {uri}")

    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# run rasterize
# ---------------------------------------------------------------------------


def cmd_run_rasterize(args) -> int:
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.rasterize_vector import (
        RasterizeVectorOperator,
        RasterizeVectorParams,
    )
    from quarry_registry.registry import Registry

    workspace = _resolve_workspace(args)

    # Materialize input through router
    router = _get_router()
    try:
        vector_artifact = _materialize_cli_source(
            router, args.vector, workspace, label="Vector", quiet=args.json_output
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Set up executor + registry
    executor = LocalExecutor()
    registry = Registry(workspace)
    registry.save_artifact(vector_artifact)

    # Output path
    output_dir = workspace / "rasterize"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else output_dir / "rasterized.tif"

    # Parse resolution
    try:
        parts = [float(x.strip()) for x in args.resolution.split(",")]
    except ValueError:
        print(
            f"Invalid --resolution value: {args.resolution!r} (expected x_res,y_res)",
            file=sys.stderr,
        )
        return 1
    if len(parts) == 1:
        resolution = (parts[0], parts[0])
    elif len(parts) == 2:
        resolution = (parts[0], parts[1])
    else:
        print(
            f"Invalid --resolution value: {args.resolution!r} (expected x_res or x_res,y_res)",
            file=sys.stderr,
        )
        return 1

    # Parse extent
    extent = None
    if args.extent:
        try:
            extent_parts = [float(x.strip()) for x in args.extent.split(",")]
        except ValueError:
            print(
                f"Invalid --extent value: {args.extent!r} (expected xmin,ymin,xmax,ymax)",
                file=sys.stderr,
            )
            return 1
        if len(extent_parts) != 4:
            print(
                f"Invalid --extent value: {args.extent!r} (expected xmin,ymin,xmax,ymax)",
                file=sys.stderr,
            )
            return 1
        extent = (extent_parts[0], extent_parts[1], extent_parts[2], extent_parts[3])

    params = RasterizeVectorParams(
        output_path=str(output_path),
        resolution=resolution,
        extent=extent,
        burn_value=args.burn_value,
        burn_attribute=args.burn_attribute,
        nodata=args.nodata,
        dtype=args.dtype,
    )

    if not args.json_output:
        print(f"Running rasterize → {output_path}")
    run_record = executor.submit(
        RasterizeVectorOperator(),
        [vector_artifact],
        params,
    )
    registry.save_run(run_record)

    failure_rc = _handle_run_failure(run_record)
    if failure_rc:
        if args.json_output:
            valid = sum(1 for c in run_record.checks if c.state.value == "valid")
            invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
            warning = sum(1 for c in run_record.checks if c.state.value == "warning")
            data = {
                "operator_name": run_record.operator_name,
                "status": "failed",
                "run_id": run_record.id,
                "output": None,
                "checks": {"valid": valid, "invalid": invalid, "warning": warning},
                "error": run_record.error or f"{run_record.operator_name} did not complete",
            }
            _emit_json(data)
        return 1

    invalid_rc = _handle_invalid_checks(run_record.checks, run_record.operator_name)
    if invalid_rc and not args.json_output:
        return invalid_rc

    # Report
    output = _require_run_output(run_record).artifact

    valid = sum(1 for c in run_record.checks if c.state.value == "valid")
    invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
    warning = sum(1 for c in run_record.checks if c.state.value == "warning")

    if args.json_output:
        data = {
            "operator_name": run_record.operator_name,
            "status": "completed",
            "run_id": run_record.id,
            "output": {
                "name": output.name,
                "uri": output.backing.uri if output.backing else None,
                "artifact_id": output.id,
            },
            "checks": {"valid": valid, "invalid": invalid, "warning": warning},
            "error": None,
        }
        _emit_json(data)
        return 0 if not invalid_rc else 2

    uri = output.backing.uri if output.backing else "?"
    print(f"\nCompleted (1 step, {len(run_record.checks)} checks)")
    print(f"  {output.name:<25} → {uri}")

    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# Param coercion for generic dispatch
# ---------------------------------------------------------------------------


def _coerce_value(raw: str, target_type: type) -> object:
    """Coerce a raw CLI string to the target dataclass field type."""
    import types
    import typing

    origin = typing.get_origin(target_type)
    args = typing.get_args(target_type)

    # Union / Optional (X | None)
    if origin is typing.Union or isinstance(target_type, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if type(None) in args or len(non_none) < len(args):
            if raw.lower() == "none":
                return None
            if len(non_none) == 1:
                return _coerce_value(raw, non_none[0])

    # Literal
    if origin is typing.Literal:
        if raw not in args:
            raise ValueError(f"{raw!r} not in allowed values: {args}")
        return raw

    # list[X]
    if origin is list:
        if not raw:
            return []
        elem_type = args[0] if args else str
        return [_coerce_value(item.strip(), elem_type) for item in raw.split(",")]

    # tuple[X, ...]
    if origin is tuple:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != len(args):
            raise ValueError(
                f"Expected {len(args)} comma-separated values for tuple, got {len(parts)}"
            )
        return tuple(_coerce_value(p, t) for p, t in zip(parts, args))

    # Scalars
    if target_type is bool:
        low = raw.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot convert {raw!r} to bool (expected true/false/1/0/yes/no)")

    if target_type is int:
        return int(raw)

    if target_type is float:
        return float(raw)

    if target_type is str:
        return raw

    raise ValueError(f"Unsupported param type: {target_type}")


def _build_params(
    params_cls: type[OperatorParams],
    raw_params: dict[str, str],
    output_path: str,
) -> OperatorParams:
    """Build a typed params dataclass from raw CLI key=value strings."""
    import dataclasses
    import typing

    hints = typing.get_type_hints(params_cls, include_extras=True)
    valid_fields = {f.name for f in dataclasses.fields(params_cls)}

    kwargs: dict[str, object] = {"output_path": output_path}

    for key, raw_value in raw_params.items():
        if key not in valid_fields:
            raise ValueError(
                f"Unknown parameter {key!r} for {params_cls.__name__}. "
                f"Available: {', '.join(sorted(valid_fields - {'output_path'}))}"
            )
        kwargs[key] = _coerce_value(raw_value, hints[key])

    params = params_cls(**kwargs)
    if not isinstance(params, OperatorParams):
        raise TypeError(f"{params_cls.__name__} must inherit OperatorParams")
    return params


# ---------------------------------------------------------------------------
# run <operator-name> (generic dispatch)
# ---------------------------------------------------------------------------


def cmd_run_generic(args) -> int:
    """Generic operator dispatch: materialize inputs, execute, persist, report."""
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.registry import get_operator, get_params_class
    from quarry_registry.registry import Registry

    operator_name: str = args.operator_name
    input_paths: list[str] = args.input
    output_path_raw: str | None = args.output
    raw_params: dict[str, str] = {}
    for item in args.params or []:
        if "=" not in item:
            print(f"Invalid param format: {item!r} (expected key=value)", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        raw_params[k] = v

    workspace = _resolve_workspace(args)

    # 1. Get operator and params class
    try:
        operator = get_operator(operator_name)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1

    params_cls = get_params_class(operator_name)
    spec = operator.spec

    # 2. Validate input count
    if len(input_paths) < spec.min_inputs:
        print(
            f"Operator {operator_name!r} requires at least {spec.min_inputs} input(s), "
            f"got {len(input_paths)}",
            file=sys.stderr,
        )
        return 1
    if spec.max_inputs >= 0 and len(input_paths) > spec.max_inputs:
        print(
            f"Operator {operator_name!r} accepts at most {spec.max_inputs} input(s), "
            f"got {len(input_paths)}",
            file=sys.stderr,
        )
        return 1

    # 3. Resolve output path
    output_dir = workspace / operator_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path_raw:
        output_path = str(Path(output_path_raw).resolve())
    else:
        ext_map = {"raster": ".tif", "vector": ".gpkg", "table": ".csv"}
        ext = ext_map.get(spec.output_type.value, ".out")
        output_path = str(output_dir / f"{operator_name}{ext}")

    # 4. Build typed params
    try:
        params = _build_params(params_cls, raw_params, output_path)
    except (ValueError, TypeError) as e:
        print(f"Parameter error: {e}", file=sys.stderr)
        return 1

    # 5. Materialize inputs through router
    router = _get_router()
    artifacts = []
    for input_path_str in input_paths:
        try:
            artifact = _materialize_cli_source(
                router, input_path_str, workspace, label="Input", quiet=args.json_output
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        artifacts.append(artifact)

    # 6. Set up executor + registry
    executor = LocalExecutor()
    registry = Registry(workspace)
    for art in artifacts:
        registry.save_artifact(art)

    # 7. Execute
    if not args.json_output:
        print(f"Running {operator_name} → {output_path}")
    run_record = executor.submit(operator, artifacts, params)
    registry.save_run(run_record)

    failure_rc = _handle_run_failure(run_record)
    if failure_rc:
        if args.json_output:
            valid = sum(1 for c in run_record.checks if c.state.value == "valid")
            invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
            warning = sum(1 for c in run_record.checks if c.state.value == "warning")
            data = {
                "operator_name": run_record.operator_name,
                "status": "failed",
                "run_id": run_record.id,
                "output": None,
                "checks": {"valid": valid, "invalid": invalid, "warning": warning},
                "error": run_record.error or f"{run_record.operator_name} did not complete",
            }
            _emit_json(data)
        return failure_rc

    invalid_rc = _handle_invalid_checks(run_record.checks, run_record.operator_name)
    if invalid_rc and not args.json_output:
        return invalid_rc

    # 9. Report
    output_artifact = _require_run_output(run_record).artifact

    valid = sum(1 for c in run_record.checks if c.state.value == "valid")
    invalid = sum(1 for c in run_record.checks if c.state.value == "invalid")
    warning = sum(1 for c in run_record.checks if c.state.value == "warning")

    if args.json_output:
        data = {
            "operator_name": run_record.operator_name,
            "status": "completed",
            "run_id": run_record.id,
            "output": {
                "name": output_artifact.name,
                "uri": output_artifact.backing.uri if output_artifact.backing else None,
                "artifact_id": output_artifact.id,
            },
            "checks": {"valid": valid, "invalid": invalid, "warning": warning},
            "error": None,
        }
        _emit_json(data)
        return 0 if not invalid_rc else 2

    uri = output_artifact.backing.uri if output_artifact.backing else "?"
    print(f"\nCompleted (1 step, {len(run_record.checks)} checks)")
    print(f"  {output_artifact.name:<25} → {uri}")
    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _add_artifacts_subparser(subparsers) -> None:
    """Add the 'artifacts' subparser tree (list, show)."""
    art_parser = subparsers.add_parser("artifacts", help="Query the artifact registry")
    art_sub = art_parser.add_subparsers(dest="artifacts_command")

    # artifacts list
    art_list = art_sub.add_parser("list", help="List artifacts")
    art_list.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    art_list.add_argument(
        "--type",
        choices=["raster", "vector", "table"],
        help="Filter by artifact type",
    )
    art_list.add_argument("--limit", type=int, default=100, help="Max results (default: 100)")
    art_list.set_defaults(func=cmd_artifacts_list)

    # artifacts show
    art_show = art_sub.add_parser("show", help="Show artifact details")
    art_show.add_argument("artifact_id", help="Artifact ID")
    art_show.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    art_show.set_defaults(func=cmd_artifacts_show)


def _add_lineage_subparser(subparsers) -> None:
    """Add the 'lineage' subparser."""
    lin_parser = subparsers.add_parser("lineage", help="Show artifact lineage")
    lin_parser.add_argument("artifact_id", help="Artifact ID")
    lin_parser.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    lin_parser.set_defaults(func=cmd_lineage)


def _add_runs_subparser(subparsers) -> None:
    """Add the 'runs' subparser tree (list, show)."""
    runs_parser = subparsers.add_parser("runs", help="Inspect run records")
    runs_sub = runs_parser.add_subparsers(dest="runs_command")

    # runs list
    runs_list = runs_sub.add_parser("list", help="List runs")
    runs_list.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    runs_list.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed", "cancelled"],
        help="Filter by status",
    )
    runs_list.add_argument("--limit", type=int, default=100, help="Max results (default: 100)")
    runs_list.set_defaults(func=cmd_runs_list)

    # runs show
    runs_show = runs_sub.add_parser("show", help="Show run details")
    runs_show.add_argument("run_id", help="Run ID")
    runs_show.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    runs_show.set_defaults(func=cmd_runs_show)


def _add_checks_subparser(subparsers) -> None:
    """Add the 'checks' subparser tree (show)."""
    checks_parser = subparsers.add_parser("checks", help="Inspect validation checks")
    checks_sub = checks_parser.add_subparsers(dest="checks_command")

    # checks show
    checks_show = checks_sub.add_parser("show", help="Show checks for an artifact or run")
    checks_show.add_argument("id", help="Artifact ID or Run ID")
    checks_show.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    checks_show.set_defaults(func=cmd_checks_show)


def _add_route_subparser(subparsers) -> None:
    """Add the 'route' subparser."""
    route_parser = subparsers.add_parser(
        "route", help="Show inferred SourceRef and ranked connector matches for a source string"
    )
    route_parser.add_argument("source", help="Source string (path, URI, STAC reference, etc.)")
    route_parser.set_defaults(func=cmd_route)


def _add_run_subparser(subparsers) -> None:
    """Add the 'run' subparser tree (hydrology, zonal, sample, rasterize, and generic dispatch)."""
    run_parser = subparsers.add_parser("run", help="Execute a flow")
    run_sub = run_parser.add_subparsers(dest="run_command")

    # run hydrology
    hydro = run_sub.add_parser(
        "hydrology", help="Run the hydrology flow (fill → D8 → accumulation)"
    )
    hydro.add_argument("--dem", required=True, help="Path to input DEM raster")
    hydro.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    hydro.add_argument("--nodata", type=float, default=None, help="Nodata value override")
    hydro.add_argument("--no-gradient", action="store_true", help="Disable flat-region gradient")
    hydro.add_argument("--weight", type=float, default=1.0, help="Flow accumulation weight")
    hydro.set_defaults(func=cmd_run_hydrology)

    # run zonal
    zonal = run_sub.add_parser("zonal", help="Run zonal statistics (raster + polygon zones → CSV)")
    zonal.add_argument("--raster", required=True, help="Path to input raster")
    zonal.add_argument(
        "--zones", required=True, help="Path to polygon zones (GeoPackage, shapefile)"
    )
    zonal.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    zonal.add_argument("--band", type=int, default=1, help="Raster band to analyze (default: 1)")
    zonal.add_argument("--zone-id-field", default=None, help="Feature property to use as zone ID")
    zonal.set_defaults(func=cmd_run_zonal)

    # run sample
    sample = run_sub.add_parser("sample", help="Sample raster values at point locations → CSV")
    sample.add_argument("--raster", required=True, help="Path to input raster")
    sample.add_argument(
        "--points", required=True, help="Path to point vector (GeoPackage, shapefile)"
    )
    sample.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    sample.add_argument("--bands", default=None, help="Comma-separated band indices (default: all)")
    sample.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: workspace/sample/sample_raster.csv)",
    )
    sample.add_argument("--nodata", type=float, default=None, help="Nodata value override")
    sample.set_defaults(func=cmd_run_sample)

    # run rasterize
    rasterize = run_sub.add_parser(
        "rasterize", help="Rasterize vector polygons → GeoTIFF (constant or attribute burn)"
    )
    rasterize.add_argument("--vector", required=True, help="Path to input polygon vector")
    rasterize.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    rasterize.add_argument(
        "--resolution",
        required=True,
        help="Pixel resolution: x_res or x_res,y_res (CRS units)",
    )
    rasterize.add_argument(
        "--burn-value", type=float, default=1.0, help="Constant burn value (default: 1.0)"
    )
    rasterize.add_argument(
        "--burn-attribute", default=None, help="Feature property for per-feature burn value"
    )
    rasterize.add_argument("--nodata", type=float, default=0.0, help="Nodata value (default: 0.0)")
    rasterize.add_argument("--dtype", default="float32", help="Output dtype (default: float32)")
    rasterize.add_argument(
        "--output",
        default=None,
        help="Output GeoTIFF path (default: workspace/rasterize/rasterized.tif)",
    )
    rasterize.add_argument(
        "--extent", default=None, help="Output extent: xmin,ymin,xmax,ymax (default: vector bounds)"
    )
    rasterize.set_defaults(func=cmd_run_rasterize)

    # --- run <operator-name> (generic dispatch) ---
    from quarry_operators.registry import OPERATOR_NAMES

    for op_name in OPERATOR_NAMES:
        if op_name in run_sub.choices:
            continue
        op_parser = run_sub.add_parser(op_name, help=f"Run the {op_name} operator")
        op_parser.add_argument(
            "--input",
            action="append",
            required=True,
            dest="input",
            help="Input file path (repeatable for multi-input operators)",
        )
        op_parser.add_argument(
            "--output",
            default=None,
            help=f"Output file path (default: workspace/{op_name}/{op_name}.<ext>)",
        )
        op_parser.add_argument(
            "-p",
            action="append",
            dest="params",
            metavar="KEY=VALUE",
            help="Operator parameter (repeatable). E.g. -p units=degrees",
        )
        op_parser.add_argument(
            "--workspace",
            default=".",
            help="Workspace directory (default: .)",
        )
        op_parser.set_defaults(func=cmd_run_generic, operator_name=op_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quarry",
        description="Quarry — geospatial execution substrate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of human text",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_artifacts_subparser(subparsers)
    _add_lineage_subparser(subparsers)
    _add_runs_subparser(subparsers)
    _add_checks_subparser(subparsers)
    _add_route_subparser(subparsers)
    _add_run_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
