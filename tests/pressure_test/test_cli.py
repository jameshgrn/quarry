"""Pressure test: CLI invocation surface.

Lane: adapter

Exercises:
  - Parser construction and help output
  - `artifacts list` against empty and populated registry
  - `artifacts show` for existing and missing artifact
  - `lineage` for artifact with and without ancestors
  - `run hydrology` end-to-end: DEM in, artifacts + registry out
  - `run hydrology` with missing DEM (error path)
  - Subcommand dispatch: missing subcommand prints help
  - Type filter on artifact list
  - All commands respect --workspace

Failure signals:
  - CLI returns non-zero on success
  - CLI returns zero on error
  - Registry not populated after `run hydrology`
  - Lineage chain broken after flow execution
  - Output not written to workspace
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from quarry_cli.main import build_parser, main
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dem(path: Path, data: np.ndarray, nodata: float = -9999.0) -> Path:
    """Write a single-band float32 DEM to a GeoTIFF."""
    h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32618),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _bowl_dem(size: int = 20) -> np.ndarray:
    """Create a bowl-shaped DEM (center is lowest)."""
    y, x = np.mgrid[:size, :size]
    center = size / 2
    return ((x - center) ** 2 + (y - center) ** 2).astype("float32")


@pytest.fixture()
def workspace(tmp_path):
    return tmp_path


@pytest.fixture()
def dem_path(workspace):
    path = workspace / "test_dem.tif"
    _make_dem(path, _bowl_dem())
    return path


@pytest.fixture()
def populated_workspace(dem_path, workspace):
    """Run hydrology flow to populate registry, return workspace."""
    rc = main(["run", "hydrology", "--dem", str(dem_path), "--workspace", str(workspace)])
    assert rc == 0
    return workspace


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_build_parser(self):
        parser = build_parser()
        assert parser.prog == "quarry"

    def test_no_command_returns_zero(self):
        assert main([]) == 0

    def test_artifacts_no_subcommand(self):
        # argparse prints help and exits for missing subcommand
        assert main(["artifacts"]) == 0

    def test_run_no_subcommand(self):
        assert main(["run"]) == 0


# ---------------------------------------------------------------------------
# artifacts list
# ---------------------------------------------------------------------------


class TestArtifactsList:
    def test_empty_registry(self, workspace, capsys):
        rc = main(["artifacts", "list", "--workspace", str(workspace)])
        assert rc == 0
        assert "No artifacts found" in capsys.readouterr().out

    def test_populated_registry(self, populated_workspace, capsys):
        rc = main(["artifacts", "list", "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        # Hydrology flow produces: input DEM + filled + D8 + accumulation = 4 artifacts
        assert "4 artifact(s)" in out

    def test_type_filter(self, populated_workspace, capsys):
        rc = main(
            [
                "artifacts",
                "list",
                "--workspace",
                str(populated_workspace),
                "--type",
                "raster",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "artifact(s)" in out

    def test_type_filter_no_match(self, populated_workspace, capsys):
        rc = main(
            [
                "artifacts",
                "list",
                "--workspace",
                str(populated_workspace),
                "--type",
                "vector",
            ]
        )
        assert rc == 0
        assert "No artifacts found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# artifacts show
# ---------------------------------------------------------------------------


class TestArtifactsShow:
    def test_show_existing(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        assert len(artifacts) > 0

        rc = main(
            [
                "artifacts",
                "show",
                artifacts[0].id,
                "--workspace",
                str(populated_workspace),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert artifacts[0].id in out
        assert artifacts[0].type.value in out

    def test_show_missing(self, workspace, capsys):
        rc = main(["artifacts", "show", "nonexistent-id", "--workspace", str(workspace)])
        assert rc == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_lineage_leaf_artifact(self, populated_workspace, capsys):
        """Flow accumulation should have 3 ancestors (DEM → filled → D8 → accum)."""
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        # Find flow_accumulation artifact (most recent, most ancestors)
        accum = None
        for a in artifacts:
            chain = registry.get_full_lineage(a.id)
            if len(chain) >= 3:
                accum = a
                break
        assert accum is not None, "No artifact with 3+ ancestors found"

        rc = main(["lineage", accum.id, "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ancestor(s)" in out
        assert accum.name in out

    def test_lineage_root_artifact(self, populated_workspace, capsys):
        """Input DEM should have no ancestors."""
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        # Find the artifact with zero ancestors (the input DEM)
        root = None
        for a in artifacts:
            chain = registry.get_full_lineage(a.id)
            if len(chain) == 0:
                root = a
                break
        assert root is not None, "No root artifact found"

        rc = main(["lineage", root.id, "--workspace", str(populated_workspace)])
        assert rc == 0
        assert "no ancestors" in capsys.readouterr().out

    def test_lineage_missing_artifact(self, workspace, capsys):
        rc = main(["lineage", "nonexistent-id", "--workspace", str(workspace)])
        assert rc == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run hydrology
# ---------------------------------------------------------------------------


class TestRunHydrology:
    def test_end_to_end(self, dem_path, workspace, capsys):
        rc = main(["run", "hydrology", "--dem", str(dem_path), "--workspace", str(workspace)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Completed" in out
        assert "3 steps" in out

        # Verify output files exist
        hydro_dir = workspace / "hydrology"
        assert (hydro_dir / "filled_dem.tif").exists()
        assert (hydro_dir / "flow_direction.tif").exists()
        assert (hydro_dir / "flow_accumulation.tif").exists()

        # Verify registry populated
        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        assert len(artifacts) == 4  # input + 3 outputs

    def test_missing_dem(self, workspace, capsys):
        rc = main(
            [
                "run",
                "hydrology",
                "--dem",
                str(workspace / "nonexistent.tif"),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_custom_weight(self, dem_path, workspace, capsys):
        rc = main(
            [
                "run",
                "hydrology",
                "--dem",
                str(dem_path),
                "--workspace",
                str(workspace),
                "--weight",
                "2.0",
            ]
        )
        assert rc == 0
        assert "Completed" in capsys.readouterr().out

    def test_no_gradient(self, dem_path, workspace, capsys):
        rc = main(
            [
                "run",
                "hydrology",
                "--dem",
                str(dem_path),
                "--workspace",
                str(workspace),
                "--no-gradient",
            ]
        )
        assert rc == 0
        assert "Completed" in capsys.readouterr().out

    def test_workspace_flag_respected(self, dem_path, tmp_path):
        custom_ws = tmp_path / "custom_workspace"
        custom_ws.mkdir()
        rc = main(
            [
                "run",
                "hydrology",
                "--dem",
                str(dem_path),
                "--workspace",
                str(custom_ws),
            ]
        )
        assert rc == 0
        assert (custom_ws / ".quarry" / "registry.duckdb").exists()
        assert (custom_ws / "hydrology" / "filled_dem.tif").exists()


# ---------------------------------------------------------------------------
# Integration: full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_run_then_list_then_lineage(self, dem_path, workspace, capsys):
        """Full CLI round-trip: run → list → show → lineage."""
        # Run
        rc = main(["run", "hydrology", "--dem", str(dem_path), "--workspace", str(workspace)])
        assert rc == 0
        capsys.readouterr()  # clear

        # List
        rc = main(["artifacts", "list", "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "4 artifact(s)" in out

        # Get an artifact ID from registry to use in show + lineage
        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        last = artifacts[0]  # most recent

        # Show
        rc = main(["artifacts", "show", last.id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert last.id in out

        # Lineage
        rc = main(["lineage", last.id, "--workspace", str(workspace)])
        assert rc == 0
