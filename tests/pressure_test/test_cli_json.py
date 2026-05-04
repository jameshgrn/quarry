"""Pressure test: CLI JSON mode.

Lane: adapter

Exercises:
  - `--json` flag produces parseable JSON output on stdout
  - `--json` route command emits correct source/matches/selected structure
  - `--json` route command with no match returns exit code 2 and empty matches
  - `--json` artifacts list command returns empty list for fresh workspace
  - `--json` run zonal command suppresses progress prints and emits single JSON
  - Default text mode (no --json) remains unchanged

Failure signals:
  - JSON output is not valid parseable JSON
  - JSON output contains more than one line
  - Missing expected keys in JSON response
  - Text mode output contains JSON braces
  - Progress prints appear in JSON mode output
"""

import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
from quarry_cli.main import main
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Helpers (inlined to avoid dependency on other test files)
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


def _write_polygon_zones(path: Path, polygons: list[Polygon]) -> Path:
    """Write polygon zones to GeoPackage."""
    schema = {"geometry": "Polygon", "properties": {}}
    crs = CRS.from_epsg(32618).to_dict()
    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for poly in polygons:
            dst.write({"geometry": mapping(poly), "properties": {}})
    return path


def _make_grid_raster(size: int = 10) -> np.ndarray:
    """10×10 raster with values 1..100."""
    return np.arange(1, size * size + 1, dtype="float32").reshape(size, size)


def _make_two_zones() -> list[Polygon]:
    """Two non-overlapping polygons covering left and right halves of a 10×10 grid."""
    left = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    right = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    return [left, right]


# ---------------------------------------------------------------------------
# Route command JSON tests
# ---------------------------------------------------------------------------


class TestRouteJson:
    def test_route_json_emits_parseable_object(self, capsys, tmp_path):
        # Create a dummy raster file to route
        raster_path = tmp_path / "x.tif"
        _write_raster(raster_path, _make_grid_raster())

        rc = main(["--json", "route", str(raster_path)])
        captured = capsys.readouterr()
        assert rc == 0

        # Should be valid JSON
        data = json.loads(captured.out)

        # Required keys
        assert "source" in data
        assert "matches" in data
        assert "selected" in data

        # Source structure
        assert "raw" in data["source"]
        assert "kind" in data["source"]
        assert data["source"]["kind"] == "local_raster"

        # Should select COG connector for a TIFF file
        assert data["selected"] == "cog"

    def test_route_json_no_match_emits_empty_matches(self, capsys):
        # A nonsense string without recognized scheme or extension won't match any connector
        rc = main(["--json", "route", "completely_nonsense_string"])
        captured = capsys.readouterr()
        assert rc == 2

        data = json.loads(captured.out)
        assert "source" in data
        assert "matches" in data
        assert "selected" in data

        assert data["matches"] == []
        assert data["selected"] is None


# ---------------------------------------------------------------------------
# Artifacts command JSON tests
# ---------------------------------------------------------------------------


class TestArtifactsJson:
    def test_artifacts_list_json_empty_workspace(self, tmp_path, capsys):
        # Fresh workspace with no artifacts
        workspace = tmp_path / "fresh_ws"
        workspace.mkdir()

        rc = main(["--json", "artifacts", "list", "--workspace", str(workspace)])
        captured = capsys.readouterr()
        assert rc == 0

        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# Run command JSON tests
# ---------------------------------------------------------------------------


class TestRunJson:
    def test_run_json_suppresses_progress_lines(self, tmp_path, capsys):
        workspace = tmp_path / "ws"
        workspace.mkdir()

        # Create test raster
        raster_path = workspace / "test_raster.tif"
        _write_raster(raster_path, _make_grid_raster())

        # Create test zones
        zones_path = workspace / "test_zones.gpkg"
        _write_polygon_zones(zones_path, _make_two_zones())

        rc = main(
            [
                "--json",
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
        captured = capsys.readouterr()

        # Should succeed
        assert rc == 0

        # Should be single-line JSON output
        lines = captured.out.strip().split("\n")
        assert len(lines) == 1

        # Should be valid JSON
        data = json.loads(captured.out)

        # Verify JSON structure
        assert data["operator_name"] == "zonal_stats"
        assert data["status"] == "completed"
        assert "run_id" in data
        assert "output" in data
        assert data["output"] is not None
        assert "name" in data["output"]
        assert "uri" in data["output"]
        assert "artifact_id" in data["output"]
        assert "checks" in data
        assert "valid" in data["checks"]
        assert "invalid" in data["checks"]

        # Progress prints should be suppressed
        assert "Materializing" not in captured.out
        assert "Completed" not in captured.out
        assert "Registry:" not in captured.out


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestDefaultTextMode:
    def test_default_text_mode_unchanged(self, capsys, tmp_path):
        # Create a dummy raster file to route
        raster_path = tmp_path / "x.tif"
        _write_raster(raster_path, _make_grid_raster())

        rc = main(["route", str(raster_path)])
        captured = capsys.readouterr()
        assert rc == 0

        # Text mode should have the header "Source"
        assert "Source" in captured.out

        # Text mode should NOT contain JSON braces
        assert not captured.out.strip().startswith("{")

        # Should have the expected text sections
        assert "Matches" in captured.out
        assert "Selected" in captured.out
