"""Canonical example: watershed analysis with Quarry.

Lane: example (exercises connector, operator, executor, flow, registry)

Demonstrates the full Quarry substrate:
  1. INGEST  — create synthetic DEM + zone polygons, materialize via LocalFileConnector
  2. PROCESS — fill depressions, compute D8 flow direction, compute flow accumulation
  3. ANALYZE — compute zonal statistics of flow accumulation per zone polygon
  4. EXPORT  — normalize flow accumulation raster to COG
  5. INSPECT — walk the registry to show artifacts, runs, and lineage

No external data required. Runs entirely with synthetic data in a temp directory.

Usage:
    uv run python examples/watershed_analysis.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import fiona
import numpy as np
import rasterio
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.executors.local import LocalExecutor
from quarry_operators.build_cog import BuildCOGOperator, BuildCOGParams
from quarry_operators.hydrology_flow import HydrologyFlow, HydrologyFlowParams
from quarry_operators.zonal_stats import ZonalStatsOperator, ZonalStatsParams
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping


def create_synthetic_dem(path: Path, *, rows: int = 320, cols: int = 320) -> Path:
    """Generate a synthetic DEM that slopes toward a central valley.

    The surface is a parabolic trough along the y-axis with some noise,
    ensuring water flows from edges toward the center and then downhill.
    """
    x = np.linspace(-1, 1, cols)
    y = np.linspace(1, 0, rows)  # north-to-south slope
    xx, yy = np.meshgrid(x, y)

    # Parabolic cross-section + downhill slope + noise
    elevation = (
        100.0 + 30.0 * xx**2 + 50.0 * yy + np.random.default_rng(42).normal(0, 0.5, (rows, cols))
    )
    elevation = elevation.astype(np.float32)

    transform = from_bounds(500000, 4500000, 500640, 4500640, cols, rows)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32610),
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)

    return path


def create_zone_polygons(path: Path) -> Path:
    """Generate two rectangular zone polygons splitting the DEM into north/south halves.

    Uses GeoPackage (not GeoJSON) to preserve the UTM CRS — GeoJSON spec
    mandates WGS84, which would cause a CRS mismatch with the DEM.
    """
    north = Polygon(
        [
            (500000, 4500320),
            (500640, 4500320),
            (500640, 4500640),
            (500000, 4500640),
        ]
    )
    south = Polygon(
        [
            (500000, 4500000),
            (500640, 4500000),
            (500640, 4500320),
            (500000, 4500320),
        ]
    )

    schema = {"geometry": "Polygon", "properties": {"zone_name": "str"}}
    crs = CRS.from_epsg(32610).to_dict()

    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(north), "properties": {"zone_name": "north"}})
        dst.write({"geometry": mapping(south), "properties": {"zone_name": "south"}})

    return path


def main() -> None:
    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="quarry_example_"))
    data_dir = workspace / "data"
    hydro_dir = workspace / "hydro"
    export_dir = workspace / "export"
    data_dir.mkdir()
    export_dir.mkdir()

    print(f"Workspace: {workspace}\n")

    # ── 1. INGEST ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("1. INGEST — create data and materialize through connectors")
    print("=" * 60)

    dem_path = create_synthetic_dem(data_dir / "dem.tif")
    zones_path = create_zone_polygons(data_dir / "zones.gpkg")

    connector = LocalFileConnector()
    dem_result = connector.materialize(str(dem_path), workspace)
    zones_result = connector.materialize(str(zones_path), workspace)

    dem_artifact = dem_result.artifact
    zones_artifact = zones_result.artifact

    print(f"  DEM artifact:   {dem_artifact.id[:12]}...  type={dem_artifact.type.value}")
    print(f"    CRS: {dem_artifact.spatial.crs}")
    print(f"    Extent: {dem_artifact.spatial.extent}")
    bands = dem_artifact.spatial.band_count
    res = dem_artifact.spatial.resolution
    print(f"    Bands: {bands}, Resolution: {res}")
    print(f"  Zones artifact: {zones_artifact.id[:12]}...  type={zones_artifact.type.value}")
    print(f"    Features: {zones_artifact.spatial.feature_count}")
    print()

    # ── 2. PROCESS ────────────────────────────────────────────────────────
    print("=" * 60)
    print("2. PROCESS — hydrology chain: fill → D8 → accumulation")
    print("=" * 60)

    executor = LocalExecutor()
    registry = Registry(workspace)

    # Save input artifacts to registry
    registry.save_artifact(dem_artifact)
    registry.save_artifact(zones_artifact)

    flow = HydrologyFlow(executor=executor, registry=registry)
    flow_params = HydrologyFlowParams(workspace=hydro_dir)
    flow_result = flow.run(dem_artifact, flow_params)

    if not flow_result.success:
        print(f"  FAILED at step: {flow_result.failed_step}")
        print(f"  Error: {flow_result.error}")
        sys.exit(1)

    print(f"  Filled DEM:        {flow_result.filled_dem.id[:12]}...")
    print(f"  Flow direction:    {flow_result.flow_direction.id[:12]}...")
    print(f"  Flow accumulation: {flow_result.flow_accumulation.id[:12]}...")
    print(f"  Runs: {len(flow_result.runs)}, Checks: {len(flow_result.all_checks)}")

    invalid_checks = [c for c in flow_result.all_checks if c.state.value == "invalid"]
    if invalid_checks:
        print(f"  WARNING: {len(invalid_checks)} invalid checks")
        for c in invalid_checks:
            print(f"    {c.check_name}: {c.message}")
    else:
        print("  All checks passed")
    print()

    # ── 3. ANALYZE ────────────────────────────────────────────────────────
    print("=" * 60)
    print("3. ANALYZE — zonal statistics of flow accumulation per zone")
    print("=" * 60)

    zonal_op = ZonalStatsOperator()
    zonal_params = ZonalStatsParams(
        output_path=str(workspace / "zonal_stats.csv"),
        band=1,
        zone_id_field="zone_name",
    )

    # Execute through the executor for proper RunRecord tracking
    zonal_record = executor.submit(
        zonal_op,
        [flow_result.flow_accumulation, zones_artifact],
        zonal_params,
    )
    if zonal_record.status.value != "completed" or zonal_record.output is None:
        print(f"  FAILED zonal_stats: {zonal_record.error or 'zonal_stats did not complete'}")
        sys.exit(1)
    registry.save_run(zonal_record)

    zonal_artifact = zonal_record.output.artifact
    print(f"  Zonal stats artifact: {zonal_artifact.id[:12]}...  type={zonal_artifact.type.value}")

    # Read and display the CSV
    with open(zonal_params.output_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Zones: {len(rows)}")
    for row in rows:
        name = row.get("zone_name", row.get("zone_id", "?"))
        mean, mx, cnt = float(row["mean"]), float(row["max"]), row["count"]
        print(f"    {name}: mean={mean:.1f}, max={mx:.1f}, count={cnt}")
    print()

    # ── 4. EXPORT ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("4. EXPORT — normalize flow accumulation to Cloud-Optimized GeoTIFF")
    print("=" * 60)

    cog_op = BuildCOGOperator()
    cog_params = BuildCOGParams(
        output_path=str(export_dir / "flow_accumulation.cog.tif"),
        blocksize=256,
        compress="deflate",
    )

    cog_record = executor.submit(
        cog_op,
        [flow_result.flow_accumulation],
        cog_params,
    )
    if cog_record.status.value != "completed" or cog_record.output is None:
        print(f"  FAILED build_cog: {cog_record.error or 'build_cog did not complete'}")
        sys.exit(1)
    registry.save_run(cog_record)

    cog_artifact = cog_record.output.artifact
    cog_checks = {c.check_name: c.state.value for c in cog_record.checks}
    print(
        f"  COG artifact: {cog_artifact.id[:12]}...  size={cog_artifact.backing.size_bytes:,} bytes"
    )
    print(f"  Checks: {cog_checks}")
    print()

    # ── 5. INSPECT ────────────────────────────────────────────────────────
    print("=" * 60)
    print("5. INSPECT — registry contents and lineage")
    print("=" * 60)

    stats = registry.stats()
    print(
        f"  Registry: {stats['artifacts']} artifacts, {stats['runs']} runs, "
        f"{stats['checks']} checks, {stats['lineage_edges']} lineage edges"
    )
    print(f"  Artifact types: {stats['artifact_types']}")
    print(f"  Run statuses: {stats['run_statuses']}")
    print()

    # Walk lineage from the COG back to the original DEM
    print("  Lineage chain (COG → DEM):")
    chain = registry.get_full_lineage(cog_artifact.id)
    for edge in chain:
        print(f"    {edge['name']} ({edge['type']}) --[{edge['operation']}]--> ...")

    # Walk lineage from zonal stats back to inputs
    print()
    print("  Lineage chain (zonal stats → inputs):")
    chain = registry.get_full_lineage(zonal_artifact.id)
    for edge in chain:
        print(f"    {edge['name']} ({edge['type']}) --[{edge['operation']}]--> ...")

    print()
    print("=" * 60)
    print("Done. All artifacts, runs, checks, and lineage persisted.")
    print(f"Registry: {registry.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
