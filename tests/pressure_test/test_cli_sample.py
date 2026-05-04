"""Pressure test: CLI `run sample` command.

Lane: adapter

Exercises:
  - `run sample` end-to-end: raster + points → CSV output + registry populated
  - `run sample` with --bands flag (single band, multiple bands)
  - `run sample` with --output flag (custom output path)
  - `run sample` with --nodata flag
  - `run sample` returns 1 for missing raster path
  - `run sample` returns 1 for missing points path
  - `run sample` returns 1 for invalid --bands value
  - `run sample --workspace` flag respected
  - Output CSV has correct schema and row count
  - Registry contains 3 artifacts (raster + points + output table)
  - Lineage: output table has 2 ancestors (raster + points)
  - Full round-trip: run sample → artifacts list → artifacts show → lineage

Failure signals:
  - CLI returns non-zero on success
  - CLI returns zero on error
  - Registry not populated after `run sample`
  - Output CSV missing or wrong schema
  - Lineage chain broken (output should have 2 input ancestors)
"""

import csv
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from adapter_helpers import make_invalid_completed_run
from quarry_cli.main import main
from quarry_core.artifact import ArtifactType
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, mapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_raster(
    path: Path, data: np.ndarray, nodata: float | None = None, bands: int = 1
) -> Path:
    """Write a float32 GeoTIFF."""
    if data.ndim == 2:
        h, w = data.shape
        count = bands
    else:
        count, h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": "float32",
        "crs": CRS.from_epsg(32618),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            for b in range(count):
                dst.write(data[b], b + 1)
    return path


def _write_points(path: Path, coords: list[tuple[float, float]]) -> Path:
    """Write point geometries to GeoPackage."""
    schema = {"geometry": "Point", "properties": {"id": "int"}}
    crs = CRS.from_epsg(32618).to_dict()
    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for i, (x, y) in enumerate(coords):
            dst.write({"geometry": mapping(Point(x, y)), "properties": {"id": i}})
    return path


@pytest.fixture()
def workspace(tmp_path):
    return tmp_path


@pytest.fixture()
def raster_path(workspace):
    """10×10 raster with values 0..99."""
    path = workspace / "test_raster.tif"
    data = np.arange(100, dtype="float32").reshape(10, 10)
    _write_raster(path, data)
    return path


@pytest.fixture()
def multiband_raster_path(workspace):
    """10×10 raster with 3 bands: 10s, 20s, 30s."""
    path = workspace / "test_multiband.tif"
    band1 = np.ones((10, 10), dtype="float32") * 10.0
    band2 = np.ones((10, 10), dtype="float32") * 20.0
    band3 = np.ones((10, 10), dtype="float32") * 30.0
    _write_raster(path, np.stack([band1, band2, band3]))
    return path


@pytest.fixture()
def nodata_raster_path(workspace):
    """10×10 raster with nodata at specific pixels."""
    path = workspace / "test_nodata.tif"
    data = np.ones((10, 10), dtype="float32") * 5.0
    data[5, 5] = -9999.0
    _write_raster(path, data, nodata=-9999.0)
    return path


@pytest.fixture()
def points_path(workspace):
    """3 points inside the 10×10 raster extent."""
    path = workspace / "test_points.gpkg"
    _write_points(path, [(1.5, 1.5), (5.5, 5.5), (9.5, 9.5)])
    return path


# ---------------------------------------------------------------------------
# run sample — end-to-end
# ---------------------------------------------------------------------------


class TestRunSample:
    def test_end_to_end(self, raster_path, points_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

        out = capsys.readouterr().out
        assert "Completed" in out
        assert "1 step" in out

        # Verify output CSV exists
        output_csv = workspace / "sample" / "sample_raster.csv"
        assert output_csv.exists()

        # Verify CSV schema: point_id + band_1 (single-band raster, all bands)
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert "point_id" in rows[0]
        assert "band_1" in rows[0]

    def test_registry_populated(self, raster_path, points_path, workspace):
        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        # raster input + points input + output table = 3
        assert len(artifacts) == 3

        types = {a.type.value for a in artifacts}
        assert "raster" in types
        assert "vector" in types
        assert "table" in types

    def test_lineage(self, raster_path, points_path, workspace):
        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()

        table = next(a for a in artifacts if a.type.value == "table")
        chain = registry.get_full_lineage(table.id)
        # Output has 2 ancestors: raster + points
        assert len(chain) == 2

    def test_run_persisted(self, raster_path, points_path, workspace):
        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )

        registry = Registry(workspace)
        runs = registry.list_runs()
        assert len(runs) == 1
        assert runs[0].operator_name == "sample_raster"
        assert runs[0].status.value == "completed"


# ---------------------------------------------------------------------------
# run sample — flag variants
# ---------------------------------------------------------------------------


class TestRunSampleFlags:
    def test_bands_single(self, multiband_raster_path, points_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(multiband_raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
                "--bands",
                "2",
            ]
        )
        assert rc == 0

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        # Only band_2 column
        assert "band_2" in rows[0]
        assert "band_1" not in rows[0]
        assert "band_3" not in rows[0]
        for row in rows:
            assert float(row["band_2"]) == pytest.approx(20.0)

    def test_bands_multiple(self, multiband_raster_path, points_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(multiband_raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
                "--bands",
                "1,3",
            ]
        )
        assert rc == 0

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert "band_1" in rows[0]
        assert "band_3" in rows[0]
        assert "band_2" not in rows[0]
        for row in rows:
            assert float(row["band_1"]) == pytest.approx(10.0)
            assert float(row["band_3"]) == pytest.approx(30.0)

    def test_bands_all_default(self, multiband_raster_path, points_path, workspace):
        """No --bands flag → sample all bands."""
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(multiband_raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert "band_1" in rows[0]
        assert "band_2" in rows[0]
        assert "band_3" in rows[0]

    def test_output_flag(self, raster_path, points_path, workspace, capsys):
        custom_output = workspace / "my_output" / "result.csv"
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
                "--output",
                str(custom_output),
            ]
        )
        assert rc == 0
        assert custom_output.exists()

    def test_nodata_flag(self, nodata_raster_path, workspace):
        """Point at nodata pixel gets NaN."""
        # Write a point at (5.5, 4.5) which lands on pixel [5,5] = nodata
        pts = workspace / "nodata_pts.gpkg"
        _write_points(pts, [(5.5, 4.5)])

        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(nodata_raster_path),
                "--points",
                str(pts),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        # Nodata pixel should produce NaN
        val = rows[0]["band_1"]
        assert val == "" or val.lower() == "nan"

    def test_workspace_flag(self, raster_path, points_path, tmp_path):
        custom_ws = tmp_path / "custom_workspace"
        custom_ws.mkdir()
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(custom_ws),
            ]
        )
        assert rc == 0
        assert (custom_ws / ".quarry" / "registry.duckdb").exists()
        assert (custom_ws / "sample" / "sample_raster.csv").exists()


# ---------------------------------------------------------------------------
# run sample — error paths
# ---------------------------------------------------------------------------


class TestRunSampleErrors:
    def test_missing_raster(self, points_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(workspace / "nonexistent.tif"),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_missing_points(self, raster_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(workspace / "nonexistent.gpkg"),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_invalid_bands(self, raster_path, points_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
                "--bands",
                "abc",
            ]
        )
        assert rc == 1
        assert "Invalid --bands" in capsys.readouterr().err

    def test_operator_failure_returns_1(self, raster_path, workspace, capsys):
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(raster_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "FAILED:" in capsys.readouterr().err

        registry = Registry(workspace)
        runs = registry.list_runs()
        assert len(runs) == 1
        assert runs[0].operator_name == "sample_raster"
        assert runs[0].status.value == "failed"

    def test_invalid_checks_return_2(
        self,
        raster_path,
        points_path,
        workspace,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(
            "quarry_core.executors.local.LocalExecutor.submit",
            lambda _self, _operator, _inputs, _params: make_invalid_completed_run(
                workspace,
                operator_name="sample_raster",
                artifact_type=ArtifactType.TABLE,
                output_name="sample/invalid.csv",
            ),
        )

        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 2
        assert "FAILED:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


class TestOutputVerification:
    def test_sampled_values_correct(self, workspace):
        """Verify sampled values match expected raster cell values."""
        # Uniform raster: all 42.0
        raster = workspace / "uniform.tif"
        data = np.ones((10, 10), dtype="float32") * 42.0
        _write_raster(raster, data)

        pts = workspace / "verify_pts.gpkg"
        _write_points(pts, [(3.5, 3.5), (7.5, 7.5)])

        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster),
                "--points",
                str(pts),
                "--workspace",
                str(workspace),
            ]
        )

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for row in rows:
            assert float(row["band_1"]) == pytest.approx(42.0)

    def test_point_outside_extent_gets_nan(self, raster_path, workspace):
        """Point outside raster extent → NaN."""
        pts = workspace / "outside_pts.gpkg"
        _write_points(pts, [(100.0, 100.0)])

        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(pts),
                "--workspace",
                str(workspace),
            ]
        )

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        val = rows[0]["band_1"]
        assert val == "" or val.lower() == "nan"

    def test_row_count_matches_input_points(self, raster_path, workspace):
        """Output row count always equals input point count."""
        pts = workspace / "many_pts.gpkg"
        coords = [(float(i) + 0.5, float(j) + 0.5) for i in range(5) for j in range(5)]
        _write_points(pts, coords)

        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(pts),
                "--workspace",
                str(workspace),
            ]
        )

        output_csv = workspace / "sample" / "sample_raster.csv"
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 25


# ---------------------------------------------------------------------------
# Integration: full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_run_then_list_then_show_then_lineage(
        self, raster_path, points_path, workspace, capsys
    ):
        """Full CLI round-trip: run sample → list → show → lineage."""
        rc = main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        capsys.readouterr()

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

    def test_runs_show_after_sample(self, raster_path, points_path, workspace, capsys):
        """run sample → runs show displays params as key-value lines."""
        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
                "--bands",
                "1",
            ]
        )
        capsys.readouterr()

        registry = Registry(workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        rc = main(["runs", "show", run_id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Params:" in out
        assert "output_path:" in out
        assert "bands:" in out

    def test_checks_show_after_sample(self, raster_path, points_path, workspace, capsys):
        """run sample → checks show lists validation checks."""
        main(
            [
                "run",
                "sample",
                "--raster",
                str(raster_path),
                "--points",
                str(points_path),
                "--workspace",
                str(workspace),
            ]
        )
        capsys.readouterr()

        registry = Registry(workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        rc = main(["checks", "show", run_id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Checks for run" in out
