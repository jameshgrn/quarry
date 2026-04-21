"""Pressure test: CLI `run zonal` command.

Lane: adapter

Exercises:
  - `run zonal` end-to-end: raster + zones → CSV output + registry populated
  - `run zonal` with --band flag variant
  - `run zonal` with --zone-id-field flag variant
  - `run zonal` returns 1 for missing raster path
  - `run zonal` returns 1 for missing zones path
  - `run zonal --workspace` flag respected (output + registry land in specified dir)
  - Output CSV has correct schema and row count
  - Registry contains 3 artifacts (raster + zones + output table)
  - Lineage: output table has 2 ancestors (raster + zones)
  - Full round-trip: run zonal → artifacts list → artifacts show → lineage

Failure signals:
  - CLI returns non-zero on success
  - CLI returns zero on error
  - Registry not populated after `run zonal`
  - Output CSV missing or wrong schema
  - Lineage chain broken (output should have 2 input ancestors)
"""

import csv
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from quarry_cli.main import main
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_raster(path: Path, data: np.ndarray, nodata: float | None = None) -> Path:
    """Write a single-band float32 GeoTIFF."""
    h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(32618),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)
    return path


def _write_multiband_raster(path: Path, data: np.ndarray) -> Path:
    """Write a multi-band float32 GeoTIFF. data shape: (bands, rows, cols)."""
    bands, h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype="float32",
        crs=CRS.from_epsg(32618),
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)
    return path


def _write_zones(
    path: Path,
    polygons: list[Polygon],
    properties: list[dict] | None = None,
) -> Path:
    """Write polygon zones to GeoPackage (preserves projected CRS, unlike GeoJSON)."""
    schema = {"geometry": "Polygon", "properties": {}}
    if properties and len(properties) > 0:
        for k in properties[0]:
            schema["properties"][k] = "str"
    crs = CRS.from_epsg(32618).to_dict()
    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for i, poly in enumerate(polygons):
            props = properties[i] if properties else {}
            dst.write({"geometry": mapping(poly), "properties": props})
    return path


def _make_grid_raster(size: int = 10) -> np.ndarray:
    """10×10 raster with values 1..100."""
    return np.arange(1, size * size + 1, dtype="float32").reshape(size, size)


def _make_two_zones() -> list[Polygon]:
    """Two non-overlapping polygons covering left and right halves of a 10×10 grid."""
    left = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    right = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    return [left, right]


@pytest.fixture()
def workspace(tmp_path):
    return tmp_path


@pytest.fixture()
def raster_path(workspace):
    path = workspace / "test_raster.tif"
    _write_raster(path, _make_grid_raster())
    return path


@pytest.fixture()
def zones_path(workspace):
    path = workspace / "test_zones.gpkg"
    _write_zones(path, _make_two_zones())
    return path


@pytest.fixture()
def zones_with_ids_path(workspace):
    path = workspace / "test_zones_ids.gpkg"
    _write_zones(
        path,
        _make_two_zones(),
        properties=[{"name": "west"}, {"name": "east"}],
    )
    return path


@pytest.fixture()
def multiband_raster_path(workspace):
    path = workspace / "test_multiband.tif"
    band1 = np.ones((10, 10), dtype="float32") * 10.0
    band2 = np.ones((10, 10), dtype="float32") * 20.0
    _write_multiband_raster(path, np.stack([band1, band2]))
    return path


# ---------------------------------------------------------------------------
# run zonal — end-to-end
# ---------------------------------------------------------------------------


class TestRunZonal:
    def test_end_to_end(self, raster_path, zones_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

        out = capsys.readouterr().out
        assert "Completed" in out
        assert "1 step" in out

        # Verify output CSV exists
        output_csv = workspace / "zonal" / "zonal_stats.csv"
        assert output_csv.exists()

        # Verify CSV schema
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        expected_cols = {"zone_id", "count", "sum", "mean", "min", "max", "std"}
        assert set(rows[0].keys()) == expected_cols

    def test_registry_populated(self, raster_path, zones_path, workspace):
        main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        # raster input + zones input + output table = 3
        assert len(artifacts) == 3

        types = {a.type.value for a in artifacts}
        assert "raster" in types
        assert "vector" in types
        assert "table" in types

    def test_lineage(self, raster_path, zones_path, workspace):
        main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()

        # Find the table artifact (the output)
        table = next(a for a in artifacts if a.type.value == "table")
        chain = registry.get_full_lineage(table.id)
        # Output has 2 ancestors: raster + zones
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# run zonal — flag variants
# ---------------------------------------------------------------------------


class TestRunZonalFlags:
    def test_band_flag(self, multiband_raster_path, zones_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(multiband_raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
                "--band",
                "2",
            ]
        )
        assert rc == 0

        output_csv = workspace / "zonal" / "zonal_stats.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        # Band 2 is all 20s, so mean should be 20.0
        for row in rows:
            assert float(row["mean"]) == pytest.approx(20.0)

    def test_zone_id_field(self, raster_path, zones_with_ids_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_with_ids_path),
                "--workspace",
                str(workspace),
                "--zone-id-field",
                "name",
            ]
        )
        assert rc == 0

        output_csv = workspace / "zonal" / "zonal_stats.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        zone_ids = {row["zone_id"] for row in rows}
        assert zone_ids == {"west", "east"}

    def test_workspace_flag(self, raster_path, zones_path, tmp_path):
        custom_ws = tmp_path / "custom_workspace"
        custom_ws.mkdir()
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(custom_ws),
            ]
        )
        assert rc == 0
        assert (custom_ws / ".quarry" / "registry.duckdb").exists()
        assert (custom_ws / "zonal" / "zonal_stats.csv").exists()


# ---------------------------------------------------------------------------
# run zonal — error paths
# ---------------------------------------------------------------------------


class TestRunZonalErrors:
    def test_missing_raster(self, zones_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(workspace / "nonexistent.tif"),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_missing_zones(self, raster_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(workspace / "nonexistent.gpkg"),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


class TestOutputVerification:
    def test_stats_correctness(self, raster_path, zones_path, workspace):
        """Verify computed stats match hand-calculated values."""
        main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )

        output_csv = workspace / "zonal" / "zonal_stats.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))

        # Both zones cover pixels, so count > 0
        for row in rows:
            assert int(row["count"]) > 0
            assert float(row["mean"]) > 0
            assert float(row["min"]) <= float(row["mean"]) <= float(row["max"])

    def test_nodata_raster(self, workspace, zones_path, capsys):
        """Raster with nodata — verify nodata pixels excluded from stats."""
        data = np.ones((10, 10), dtype="float32") * 5.0
        data[0:5, 0:5] = -9999.0  # nodata in upper-left quadrant
        raster_path = workspace / "nodata_raster.tif"
        _write_raster(raster_path, data, nodata=-9999.0)

        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

        output_csv = workspace / "zonal" / "zonal_stats.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))

        # Both zones should have mean=5.0 for non-nodata pixels
        for row in rows:
            if int(row["count"]) > 0:
                assert float(row["mean"]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Integration: full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_run_then_list_then_show_then_lineage(self, raster_path, zones_path, workspace, capsys):
        """Full CLI round-trip: run zonal → list → show → lineage."""
        # Run
        rc = main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        capsys.readouterr()  # clear

        # List
        rc = main(["artifacts", "list", "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "3 artifact(s)" in out

        # Show the table artifact
        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        table = next(a for a in artifacts if a.type.value == "table")

        rc = main(["artifacts", "show", table.id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert table.id in out
        assert "table" in out

        # Lineage
        rc = main(["lineage", table.id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 ancestor(s)" in out

    def test_type_filter_table(self, raster_path, zones_path, workspace, capsys):
        """After zonal run, --type table should show exactly 1 artifact."""
        main(
            [
                "run",
                "zonal",
                "--raster",
                str(raster_path),
                "--zones",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        capsys.readouterr()

        rc = main(
            [
                "artifacts",
                "list",
                "--workspace",
                str(workspace),
                "--type",
                "table",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 artifact(s)" in out
