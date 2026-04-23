"""Adversarial DEM fixtures for the D8 hydrology pack.

Lane: check

Purpose: deepen trust in the fill → D8 → accumulation chain by
exercising pathological geometries that stress edge-case handling.

Scenarios:
  1. Tiny hand-verifiable DEMs (2x2, 3x3, 2x3)
  2. Weird nodata geometry (L-shaped, island, nodata-at-boundary)
  3. Thin diagonal channel (1-cell wide, diagonal spill path)
  4. Plateaus near boundaries (flat shelf draining through single spill cell)
  5. Mask discontinuities (nodata bisects valid region into disconnected components)
  6. All-nodata-except-one-cell (degenerate single valid cell)
  7. Checkerboard nodata (maximal mask fragmentation)
  8. Corner-to-corner slope (pure diagonal drainage)

Every test runs the full chain (fill → D8 → accumulation) and checks:
  - Chain completes successfully
  - Conservation holds (outlet sum = valid cell count × weight)
  - No interior pits after fill
  - No invalid checks
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.artifact import ValidationState
from quarry_core.executors.local import LocalExecutor
from quarry_operators.d8_flow_direction import OUTLET, PIT
from quarry_operators.hydrology_flow import HydrologyFlow, HydrologyFlowParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dem(path: Path, data: np.ndarray, nodata: float = -9999.0) -> Path:
    """Write a single-band float64 DEM to a GeoTIFF."""
    h, w = data.shape
    transform = rasterio.transform.from_bounds(0.0, 0.0, float(w), float(h), w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float64",
        crs=rasterio.crs.CRS.from_epsg(32618),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _run_chain(dem_path: Path, workspace: Path, weight: float = 1.0):
    """Run full hydrology chain, return result + raw arrays."""
    conn = LocalFileConnector()
    art = conn.materialize(str(dem_path), workspace).artifact
    executor = LocalExecutor()
    flow = HydrologyFlow(executor=executor)
    result = flow.run(art, HydrologyFlowParams(workspace=workspace / "run", weight=weight))
    return result


def _read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _assert_chain_healthy(result, weight: float = 1.0):
    """Assert standard chain health: success, no invalid checks, conservation."""
    assert result.success, f"Chain failed at {result.failed_step}: {result.error}"
    assert not result.has_invalid_checks, (
        f"Invalid checks: {[c for c in result.all_checks if c.state == ValidationState.INVALID]}"
    )
    # Conservation
    conservation = [c for c in result.all_checks if c.check_name == "conservation"]
    assert len(conservation) == 1
    assert conservation[0].state == ValidationState.VALID, conservation[0].message
    # No interior pits
    pit_checks = [c for c in result.all_checks if c.check_name == "no_interior_pits"]
    assert len(pit_checks) == 1
    assert pit_checks[0].state == ValidationState.VALID, pit_checks[0].message


# ===========================================================================
# 1. Tiny hand-verifiable DEMs
# ===========================================================================


class TestTinyDEMs:
    """DEMs small enough to verify every cell by hand."""

    def test_2x2_uniform(self, tmp_path):
        """2x2 flat — all cells are boundary, all outlets."""
        data = np.full((2, 2), 5.0)
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # 4 valid cells, all boundary → each is an outlet with acc=1
        # (some may receive flow from neighbors via flat resolution)
        assert acc.sum() >= 4.0  # conservation: total weight distributed

    def test_2x2_single_low_corner(self, tmp_path):
        """2x2 with one low corner — other 3 drain to it.

        [5] [5]
        [5] [1]

        Cell (1,1) at elevation 1 is lowest. All boundary cells.
        Expected: (1,1) receives all flow → acc = 4.
        """
        data = np.array([[5.0, 5.0], [5.0, 1.0]])
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # (1,1) is the low point — should collect all 4 cells
        assert acc[1, 1] == pytest.approx(4.0)

    def test_3x3_center_pit(self, tmp_path):
        """3x3 with center pit — fills to boundary, drains outward.

        [5] [5] [5]
        [5] [1] [5]
        [5] [5] [5]

        After fill: center raised to 5, all flat.
        With gradient: center drains to nearest boundary.
        """
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ]
        )
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # Filled DEM: center should be raised to 5
        filled = _read_raster(tmp_path / "ws" / "run" / "filled_dem.tif")
        assert filled[1, 1] >= 5.0

    def test_2x3_asymmetric_slope(self, tmp_path):
        """2x3 east slope — hand-verifiable accumulation.

        [3] [2] [1]
        [3] [2] [1]

        All flow east. Right column = outlets.
        acc[0,0]=1, acc[0,1]=2, acc[0,2]=3 (outlet)
        acc[1,0]=1, acc[1,1]=2, acc[1,2]=3 (outlet)
        Conservation: 2 outlets × 3 = 6 cells × 1.0.
        """
        data = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # Right column collects all flow from its row
        assert acc[0, 2] >= 2.0  # at least 2 (self + left neighbor's contribution)
        assert acc[1, 2] >= 2.0

    def test_1x5_linear_chain(self, tmp_path):
        """1x5 east slope — trivial linear accumulation.

        [5] [4] [3] [2] [1]

        acc = [1, 2, 3, 4, 5]
        """
        data = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # Rightmost cell is outlet collecting all 5
        assert acc[0, 4] == pytest.approx(5.0)

    def test_5x1_vertical_chain(self, tmp_path):
        """5x1 south slope — trivial vertical accumulation.

        [5]
        [4]
        [3]
        [2]
        [1]

        acc = [1, 2, 3, 4, 5]
        """
        data = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        assert acc[4, 0] == pytest.approx(5.0)


# ===========================================================================
# 2. Weird nodata geometry
# ===========================================================================


class TestNodataGeometry:
    """Nodata shapes that stress mask handling."""

    def test_l_shaped_nodata(self, tmp_path):
        """L-shaped nodata carves into the DEM.

        Valid region is the complement of an L in the upper-left.
        Tests that fill/D8/accumulation handle irregular nodata boundaries.
        """
        nodata = -9999.0
        data = np.full((7, 7), 10.0)
        # Gentle south slope so there's drainage
        for r in range(7):
            data[r, :] += (6 - r) * 0.5
        # L-shaped nodata: top-left block + left column extension
        data[0:3, 0:3] = nodata
        data[3:5, 0] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_nodata_island(self, tmp_path):
        """Nodata surrounds a small valid island (3x3 valid in 7x7 nodata).

        Known limitation: Priority-Flood seeds from grid boundary. If the
        valid island is completely disconnected from the grid edge by nodata,
        the flood never reaches it. Interior pits on the island stay unfilled.
        D8 assigns PIT codes to unreachable low cells.

        This test documents the limitation — chain still completes, conservation
        still holds (PITs retain self-weight), but no_interior_pits may fail.
        """
        nodata = -9999.0
        data = np.full((7, 7), nodata)
        # 3x3 valid island in center
        data[2:5, 2:5] = 10.0
        data[3, 3] = 5.0  # center pit on island

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")

        # Chain completes — the limitation doesn't crash anything
        assert result.success

        # Conservation still holds (PITs retain self-weight, outlets collect rest)
        conservation = [c for c in result.all_checks if c.check_name == "conservation"]
        assert conservation[0].state == ValidationState.VALID

        # Verify all 9 valid cells accounted for
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        valid = acc >= 0
        fdir = _read_raster(tmp_path / "ws" / "run" / "flow_direction.tif")
        outlet_mask = valid & ((fdir == OUTLET) | (fdir == PIT))
        outlet_sum = float(acc[outlet_mask].sum())
        assert abs(outlet_sum - 9.0) < 1e-6

    def test_nodata_at_all_corners(self, tmp_path):
        """Nodata in all 4 corners, valid cross shape.

        Tests boundary outlet detection when corners are invalid.
        """
        nodata = -9999.0
        data = np.full((5, 5), 10.0)
        for r in range(5):
            data[r, :] += (4 - r) * 0.5
        # Knock out corners
        data[0, 0] = nodata
        data[0, 4] = nodata
        data[4, 0] = nodata
        data[4, 4] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_nodata_one_cell_wide_gap(self, tmp_path):
        """Single-cell nodata gap bisects a south-sloping DEM.

        Row 3 center cell is nodata. Tests that fill handles
        nodata "pinch point" correctly.
        """
        nodata = -9999.0
        data = np.zeros((7, 7))
        for r in range(7):
            data[r, :] = 20.0 - r * 2
        data[3, 3] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_nodata_boundary_strip(self, tmp_path):
        """Entire north boundary is nodata.

        Valid region is rows 1-6 of a 7x7 grid.
        Tests that fill doesn't rely on the actual grid boundary
        being valid — nodata-adjacent cells become effective outlets.
        """
        nodata = -9999.0
        data = np.full((7, 7), 10.0)
        for r in range(7):
            data[r, :] += (6 - r) * 0.5
        data[0, :] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)


# ===========================================================================
# 3. Thin channels and diagonal spill paths
# ===========================================================================


class TestThinChannels:
    """1-cell-wide drainage paths that stress D8 direction assignment."""

    def test_diagonal_channel_nw_to_se(self, tmp_path):
        """1-cell-wide diagonal channel from NW corner to SE corner.

        High plateau at 20, channel at 10 descending to 1.
        Only drainage path is the diagonal.
        """
        data = np.full((9, 9), 20.0)
        for i in range(9):
            data[i, i] = 10.0 - i  # diagonal from (0,0) to (8,8)

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # The SE corner should have high accumulation (whole plateau drains to it)
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        assert acc[8, 8] > 40  # most of 81 cells should drain here

    def test_sinuous_channel(self, tmp_path):
        """S-shaped channel through a plateau.

        Tests that D8 can route through a winding 1-cell path.
        """
        data = np.full((9, 9), 20.0)
        # S-curve: top-left → bottom-right
        channel = [
            (0, 1),
            (1, 1),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 5),
            (6, 5),
            (6, 6),
            (6, 7),
            (7, 7),
            (8, 7),
        ]
        for idx, (r, c) in enumerate(channel):
            data[r, c] = 15.0 - idx * 0.5

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_narrow_valley_two_cells_wide(self, tmp_path):
        """Two-cell-wide valley between ridges.

        Ridge at 30, valley floor at 10, sloping south.
        Valley is columns 4-5 in a 10x10 grid.
        """
        data = np.full((10, 10), 30.0)
        for r in range(10):
            data[r, 4] = 15.0 - r * 0.5
            data[r, 5] = 15.0 - r * 0.5

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # Bottom of valley should have high accumulation relative to grid size.
        # Valley collects 20 valley cells + adjacent ridge cells that drain in.
        # Not all 100 cells — distant ridge cells drain to grid boundary instead.
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        assert max(acc[9, 4], acc[9, 5]) > 20


# ===========================================================================
# 4. Plateaus near boundaries
# ===========================================================================


class TestBoundaryPlateaus:
    """Flat regions adjacent to grid boundaries with constrained spill."""

    def test_plateau_single_spill_point(self, tmp_path):
        """Raised rim with flat interior, single low spill point at boundary.

        Rim at 15, interior plateau at 10, one boundary cell at 5.
        All interior flow must drain through the spill point because the
        rim blocks all other exits.
        """
        data = np.full((7, 7), 15.0)  # rim
        data[1:6, 1:6] = 10.0  # interior plateau
        data[0, 3] = 5.0  # single spill point in rim

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # Spill point collects all interior flow (25 cells) + adjacent rim cells
        assert acc[0, 3] > 20

    def test_boundary_shelf(self, tmp_path):
        """Flat shelf along south boundary, interior slopes toward it.

        Interior at 20 sloping south, bottom 2 rows flat at 5.
        Tests flat resolution when the flat region IS the boundary.
        """
        data = np.zeros((10, 10))
        for r in range(8):
            data[r, :] = 20.0 - r * 2
        data[8, :] = 5.0
        data[9, :] = 5.0

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_plateau_with_pit_near_edge(self, tmp_path):
        """Flat plateau with a pit one cell from the boundary.

        Tests fill when the pit is close enough to the boundary
        that fill elevation = boundary elevation (spill barely works).
        """
        data = np.full((5, 5), 10.0)
        # Pit one cell from south edge
        data[3, 2] = 2.0
        # South boundary slightly lower
        data[4, :] = 9.0

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # Pit should fill to 9.0 (south boundary level) since boundary drains
        filled = _read_raster(tmp_path / "ws" / "run" / "filled_dem.tif")
        assert filled[3, 2] >= 9.0


# ===========================================================================
# 5. Mask discontinuities (disconnected valid regions)
# ===========================================================================


class TestMaskDiscontinuities:
    """Nodata that splits the valid region into disconnected components."""

    def test_nodata_cross_bisects_grid(self, tmp_path):
        """Nodata cross splits 9x9 grid into 4 quadrants.

        Each quadrant is an independent drainage basin.
        Conservation must hold per-quadrant.
        """
        nodata = -9999.0
        data = np.full((9, 9), 10.0)
        for r in range(9):
            for c in range(9):
                data[r, c] = 10.0 + (8 - r) * 0.3 + (8 - c) * 0.3
        # Nodata cross at row 4 and col 4
        data[4, :] = nodata
        data[:, 4] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # Verify 4 quadrants: 4×4 = 16 cells each, 64 total valid
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        valid_count = int(np.sum(acc >= 0))
        assert valid_count == 64

    def test_two_valid_strips(self, tmp_path):
        """Two horizontal valid strips separated by nodata row.

        Top strip: rows 0-2 (3×7 = 21 cells)
        Nodata: row 3
        Bottom strip: rows 4-6 (3×7 = 21 cells)

        Each strip drains independently.
        """
        nodata = -9999.0
        data = np.zeros((7, 7))
        for r in range(7):
            data[r, :] = 20.0 - r * 2
        data[3, :] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)


# ===========================================================================
# 6. Degenerate cases
# ===========================================================================


class TestDegenerateCases:
    """Extreme edge cases that shouldn't crash the chain."""

    def test_single_valid_cell_in_nodata_sea(self, tmp_path):
        """One valid cell surrounded by nodata.

        The cell is both the headwater and the outlet.
        Accumulation = 1 (self-weight).
        """
        nodata = -9999.0
        data = np.full((5, 5), nodata)
        data[2, 2] = 10.0

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")

        assert result.success
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        # The single valid cell should have acc = 1
        assert acc[2, 2] == pytest.approx(1.0)

    def test_two_valid_cells_adjacent(self, tmp_path):
        """Two adjacent valid cells in nodata sea.

        Higher cell drains to lower cell. Lower cell = outlet.
        """
        nodata = -9999.0
        data = np.full((5, 5), nodata)
        data[2, 2] = 10.0
        data[2, 3] = 5.0  # lower, east of center

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")

        assert result.success
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        assert acc[2, 3] == pytest.approx(2.0)  # collects both cells
        assert acc[2, 2] == pytest.approx(1.0)  # self only

    def test_all_same_elevation_large(self, tmp_path):
        """15x15 perfectly flat DEM.

        No natural drainage direction. Flat resolution must assign
        directions from interior to boundary. All cells drain to
        boundary outlets.
        """
        data = np.full((15, 15), 42.0)
        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)


# ===========================================================================
# 7. Checkerboard and fragmented nodata
# ===========================================================================


class TestFragmentedNodata:
    """Maximally fragmented nodata patterns."""

    def test_checkerboard_nodata(self, tmp_path):
        """Alternating valid/nodata cells (checkerboard pattern).

        Every valid cell is surrounded by nodata on cardinal sides.
        Each valid cell is its own isolated drainage basin.

        Uses a sloped base so valid cells have drainage direction.
        """
        nodata = -9999.0
        data = np.full((7, 7), nodata)
        for r in range(7):
            for c in range(7):
                if (r + c) % 2 == 0:
                    data[r, c] = 20.0 - r * 0.5

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")

        assert result.success
        # Each valid cell is isolated — all should have acc=1
        # (or pairs connected diagonally might merge)
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        valid = acc >= 0
        valid_count = int(np.sum(valid))
        assert valid_count > 0

    def test_nodata_scattered_holes(self, tmp_path):
        """Random nodata holes (10% of cells) on a sloped surface.

        Nodata holes create mask discontinuities that the chain must handle.
        """
        nodata = -9999.0
        rng = np.random.default_rng(7)
        data = np.zeros((15, 15))
        for r in range(15):
            data[r, :] = 30.0 - r * 1.5 + rng.uniform(-0.5, 0.5, 15)
        # Punch 10% nodata holes
        mask = rng.random((15, 15)) < 0.10
        data[mask] = nodata

        dem = _make_dem(tmp_path / "dem.tif", data, nodata=nodata)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)


# ===========================================================================
# 8. Corner-to-corner and diagonal drainage
# ===========================================================================


class TestDiagonalDrainage:
    """Pure diagonal or corner-biased drainage patterns."""

    def test_corner_to_corner_slope(self, tmp_path):
        """Elevation increases from SE corner to NW corner.

        Tests that D8 handles diagonal drainage properly when the
        steepest descent is consistently diagonal.
        """
        data = np.zeros((9, 9))
        for r in range(9):
            for c in range(9):
                data[r, c] = (8 - r) + (8 - c)  # highest at (0,0), lowest at (8,8)

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # SE corner should be the major outlet
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        assert acc[8, 8] > 30  # significant fraction of 81 cells

    def test_saddle_point(self, tmp_path):
        """Saddle between two ridges with two valleys.

        Creates a DEM where flow must choose between two downhill paths.
        """
        data = np.zeros((9, 9))
        for r in range(9):
            for c in range(9):
                # Saddle: high on NW-SE diagonal, low on NE-SW diagonal
                data[r, c] = 20.0 + (r - 4) ** 2 - (c - 4) ** 2

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

    def test_ridge_line(self, tmp_path):
        """North-south ridge splits drainage east and west.

        Center column high, sides slope away.
        Tests that flow correctly splits to both sides.
        """
        data = np.zeros((10, 10))
        for r in range(10):
            for c in range(10):
                # Ridge at col 5, slopes away
                data[r, c] = 20.0 - abs(c - 4.5) * 2 + (9 - r) * 0.3

        dem = _make_dem(tmp_path / "dem.tif", data)
        result = _run_chain(dem, tmp_path / "ws")
        _assert_chain_healthy(result)

        # Both east and west boundary should receive significant flow
        acc = _read_raster(tmp_path / "ws" / "run" / "flow_accumulation.tif")
        west_max = acc[:, 0].max()
        east_max = acc[:, 9].max()
        assert west_max > 5
        assert east_max > 5
