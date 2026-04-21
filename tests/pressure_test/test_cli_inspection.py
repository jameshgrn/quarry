"""Pressure test: CLI run/check inspection commands.

Lane: adapter

Exercises:
  - `runs list` against empty and populated registry
  - `runs list --status` filtering
  - `runs show <run-id>` for existing and missing run
  - `checks show <id>` for artifact, run, and missing IDs
  - Subcommand dispatch: missing subcommand prints help
  - Output formatting for all commands

Failure signals:
  - CLI returns non-zero on success
  - CLI returns zero on error
  - Run records not found after flow execution
  - Check records not found for artifacts/runs
  - Formatting breaks (missing columns, wrong alignment)
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from quarry_cli.main import main
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
    """Run hydrology flow to populate registry with runs + checks."""
    rc = main(["run", "hydrology", "--dem", str(dem_path), "--workspace", str(workspace)])
    assert rc == 0
    return workspace


# ---------------------------------------------------------------------------
# runs: subcommand dispatch
# ---------------------------------------------------------------------------


class TestRunsDispatch:
    def test_runs_no_subcommand(self):
        assert main(["runs"]) == 0

    def test_checks_no_subcommand(self):
        assert main(["checks"]) == 0


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


class TestRunsList:
    def test_empty_registry(self, workspace, capsys):
        rc = main(["runs", "list", "--workspace", str(workspace)])
        assert rc == 0
        assert "No runs found" in capsys.readouterr().out

    def test_populated_registry(self, populated_workspace, capsys):
        rc = main(["runs", "list", "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OPERATOR" in out
        assert "STATUS" in out
        assert "completed" in out
        # Hydrology flow produces 3 runs (fill, d8, accumulation)
        assert "3 run(s)" in out

    def test_status_filter_completed(self, populated_workspace, capsys):
        rc = main(
            ["runs", "list", "--status", "completed", "--workspace", str(populated_workspace)]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "completed" in out
        assert "3 run(s)" in out

    def test_status_filter_no_match(self, populated_workspace, capsys):
        rc = main(["runs", "list", "--status", "failed", "--workspace", str(populated_workspace)])
        assert rc == 0
        assert "No runs found" in capsys.readouterr().out

    def test_limit(self, populated_workspace, capsys):
        rc = main(["runs", "list", "--limit", "1", "--workspace", str(populated_workspace)])
        assert rc == 0
        assert "1 run(s)" in capsys.readouterr().out

    def test_output_columns(self, populated_workspace, capsys):
        """Table header has all expected columns."""
        main(["runs", "list", "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        for col in ("ID", "OPERATOR", "STATUS", "SUBMITTED", "DURATION"):
            assert col in out


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------


class TestRunsShow:
    def test_show_existing_run(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        assert len(runs) > 0
        run_id = runs[0].id

        rc = main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert run_id in out
        assert "Operator:" in out
        assert "Status:" in out
        assert "completed" in out

    def test_show_displays_inputs(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        assert "Inputs" in out

    def test_show_displays_output(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        assert "Output:" in out

    def test_show_displays_checks(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        assert "Checks" in out

    def test_show_displays_timing(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        assert "Duration:" in out

    def test_show_missing_run(self, workspace, capsys):
        rc = main(["runs", "show", "nonexistent-id", "--workspace", str(workspace)])
        assert rc == 1
        assert "Run not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# checks show
# ---------------------------------------------------------------------------


class TestChecksShow:
    def test_checks_for_artifact(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        # Find an artifact that has checks
        checked = [a for a in artifacts if a.checks]
        assert len(checked) > 0, "Expected at least one artifact with checks"
        aid = checked[0].id

        rc = main(["checks", "show", aid, "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Checks for artifact" in out
        assert "check(s)" in out

    def test_checks_for_run(self, populated_workspace, capsys):
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        # Find a run that has checks
        checked = [r for r in runs if r.checks]
        assert len(checked) > 0, "Expected at least one run with checks"
        rid = checked[0].id

        rc = main(["checks", "show", rid, "--workspace", str(populated_workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Checks for run" in out

    def test_checks_missing_id(self, workspace, capsys):
        rc = main(["checks", "show", "nonexistent-id", "--workspace", str(workspace)])
        assert rc == 1
        assert "No artifact or run found" in capsys.readouterr().err

    def test_checks_formatting(self, populated_workspace, capsys):
        """Each check shows state, name, message, and timestamp."""
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        checked = [a for a in artifacts if a.checks]
        assert len(checked) > 0
        aid = checked[0].id

        main(["checks", "show", aid, "--workspace", str(populated_workspace)])
        out = capsys.readouterr().out
        # Should contain validation state markers
        assert any(state in out for state in ("valid", "invalid", "warn", "unchecked"))

    def test_checks_empty_for_artifact(self, populated_workspace, capsys):
        """Artifact with no checks still returns 0."""
        registry = Registry(populated_workspace)
        artifacts = registry.list_artifacts()
        # Source DEM artifact typically has no checks
        unchecked = [a for a in artifacts if not a.checks]
        if not unchecked:
            pytest.skip("All artifacts have checks in this workspace")
        aid = unchecked[0].id

        rc = main(["checks", "show", aid, "--workspace", str(populated_workspace)])
        assert rc == 0
        assert "(no checks)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Round-trip: run → inspect
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_run_then_list_then_show(self, populated_workspace, capsys):
        """Full cycle: run hydrology → runs list → runs show → checks show."""
        # List runs
        rc = main(["runs", "list", "--workspace", str(populated_workspace)])
        assert rc == 0
        capsys.readouterr()

        # Pick first run and show it
        registry = Registry(populated_workspace)
        runs = registry.list_runs()
        run_id = runs[0].id

        rc = main(["runs", "show", run_id, "--workspace", str(populated_workspace)])
        assert rc == 0
        capsys.readouterr()

        # Show checks for same run
        rc = main(["checks", "show", run_id, "--workspace", str(populated_workspace)])
        assert rc == 0
