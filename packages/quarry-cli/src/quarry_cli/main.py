"""Quarry CLI — minimal invocation surface over the substrate.

Lane: adapter
Exposes registry queries and flow execution as shell commands.
No workflow engine, no config files, no plugin system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_workspace(args) -> Path:
    return Path(args.workspace).resolve()


# ---------------------------------------------------------------------------
# artifacts list
# ---------------------------------------------------------------------------


def cmd_artifacts_list(args) -> int:
    from quarry_core.artifact import ArtifactType
    from quarry_registry.registry import Registry

    registry = Registry(_resolve_workspace(args))
    type_filter = ArtifactType(args.type) if args.type else None
    artifacts = registry.list_artifacts(artifact_type=type_filter, limit=args.limit)

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
        print(f"Artifact not found: {args.artifact_id}", file=sys.stderr)
        return 1

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
        print(f"Artifact not found: {args.artifact_id}", file=sys.stderr)
        return 1

    chain = registry.get_full_lineage(args.artifact_id)

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
# run hydrology
# ---------------------------------------------------------------------------


def cmd_run_hydrology(args) -> int:
    from quarry_connectors.local_file import LocalFileConnector
    from quarry_core.executors.local import LocalExecutor
    from quarry_operators.hydrology_flow import HydrologyFlow, HydrologyFlowParams
    from quarry_registry.registry import Registry

    dem_path = Path(args.dem).resolve()
    if not dem_path.exists():
        print(f"DEM file not found: {dem_path}", file=sys.stderr)
        return 1

    workspace = _resolve_workspace(args)

    # Materialize DEM through connector
    connector = LocalFileConnector()
    print(f"Materializing DEM: {dem_path}")
    result = connector.materialize(str(dem_path), workspace)
    dem_artifact = result.artifact

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

    print(f"Running hydrology flow → {hydro_dir}")
    flow_result = flow.run(dem_artifact, params)

    if not flow_result.success:
        print(f"FAILED at step: {flow_result.failed_step}", file=sys.stderr)
        print(f"Error: {flow_result.error}", file=sys.stderr)
        return 1

    # Report results
    print(f"\nCompleted ({len(flow_result.runs)} steps, {len(flow_result.all_checks)} checks)")
    for a in flow_result.artifacts:
        uri = a.backing.uri if a.backing else "?"
        print(f"  {a.name:<25} → {uri}")

    invalid = [c for c in flow_result.all_checks if c.state.value == "invalid"]
    if invalid:
        print(f"\nWARNING: {len(invalid)} invalid check(s):")
        for c in invalid:
            print(f"  [{c.check_name}] {c.message}")

    print(f"\nRegistry: {registry.db_path}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quarry",
        description="Quarry — geospatial execution substrate",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- artifacts ---
    art_parser = subparsers.add_parser("artifacts", help="Query the artifact registry")
    art_sub = art_parser.add_subparsers(dest="artifacts_command")

    # artifacts list
    art_list = art_sub.add_parser("list", help="List artifacts")
    art_list.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    art_list.add_argument(
        "--type",
        choices=["raster", "vector", "table", "temporal_stack", "tile_set", "model", "point_cloud"],
        help="Filter by artifact type",
    )
    art_list.add_argument("--limit", type=int, default=100, help="Max results (default: 100)")
    art_list.set_defaults(func=cmd_artifacts_list)

    # artifacts show
    art_show = art_sub.add_parser("show", help="Show artifact details")
    art_show.add_argument("artifact_id", help="Artifact ID")
    art_show.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    art_show.set_defaults(func=cmd_artifacts_show)

    # --- lineage ---
    lin_parser = subparsers.add_parser("lineage", help="Show artifact lineage")
    lin_parser.add_argument("artifact_id", help="Artifact ID")
    lin_parser.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    lin_parser.set_defaults(func=cmd_lineage)

    # --- run ---
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
