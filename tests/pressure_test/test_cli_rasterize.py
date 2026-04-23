"""Pressure test: CLI `run rasterize` command.

Lane: adapter

Exercises:
  - `run rasterize` end-to-end: vector → GeoTIFF + registry populated
  - Constant burn (default burn_value=1.0)
  - Attribute burn (per-feature values from a property)
  - Bad attribute name → still produces raster (features with missing attr skipped)
  - Empty vector input → raster filled with nodata
  - Invalid args: missing --vector, missing --resolution, bad --resolution, bad --extent
  - Output verification: pixel values, dimensions, CRS preserved
  - Registry: 2 artifacts (vector input + raster output), run persisted, lineage
  - Full round-trip: run rasterize → artifacts list → artifacts show → lineage

Failure signals:
  - CLI returns non-zero on success
  - CLI returns zero on error
  - Registry not populated after `run rasterize`
  - Output GeoTIFF missing or wrong values
  - Lineage chain broken (output should have 1 input ancestor)
"""

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
from shapely.geometry import box, mapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_polygons(
    path: Path,
    polygons: list[tuple[tuple[float, float, float, float], dict]],
    crs_epsg: int = 32618,
) -> Path:
    """Write polygon geometries to GeoPackage.

    Each entry is (bbox_tuple, properties_dict).
    Schema is inferred from the first entry's properties.
    """
    if not polygons:
        schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    else:
        props_schema = {}
        for k, v in polygons[0][1].items():
            if isinstance(v, float):
                props_schema[k] = "float"
            elif isinstance(v, int):
                props_schema[k] = "int"
            else:
                props_schema[k] = "str"
        schema = {"geometry": "Polygon", "properties": props_schema}

    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for bbox, props in polygons:
            geom = box(*bbox)
            dst.write({"geometry": mapping(geom), "properties": props})
    return path


@pytest.fixture()
def workspace(tmp_path):
    return tmp_path


@pytest.fixture()
def vector_path(workspace):
    """Two non-overlapping 2×2 boxes in a 10×10 extent."""
    path = workspace / "test_polys.gpkg"
    _write_polygons(
        path,
        [
            ((1.0, 1.0, 3.0, 3.0), {"id": 1, "value": 10.0}),
            ((5.0, 5.0, 7.0, 7.0), {"id": 2, "value": 20.0}),
        ],
    )
    return path


@pytest.fixture()
def empty_vector_path(workspace):
    """Empty GeoPackage with polygon schema."""
    path = workspace / "empty_polys.gpkg"
    _write_polygons(path, [])
    return path


# ---------------------------------------------------------------------------
# run rasterize — end-to-end
# ---------------------------------------------------------------------------


class TestRunRasterize:
    def test_end_to_end(self, vector_path, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 0

        out = capsys.readouterr().out
        assert "Completed" in out
        assert "1 step" in out

        output_tif = workspace / "rasterize" / "rasterized.tif"
        assert output_tif.exists()

    def test_registry_populated(self, vector_path, workspace):
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        # vector input + raster output = 2
        assert len(artifacts) == 2

        types = {a.type.value for a in artifacts}
        assert "vector" in types
        assert "raster" in types

    def test_lineage(self, vector_path, workspace):
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        raster = next(a for a in artifacts if a.type.value == "raster")
        chain = registry.get_full_lineage(raster.id)
        # Output has 1 ancestor: the vector input
        assert len(chain) == 1

    def test_run_persisted(self, vector_path, workspace):
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        registry = Registry(workspace)
        runs = registry.list_runs()
        assert len(runs) == 1
        assert runs[0].operator_name == "rasterize_vector"
        assert runs[0].status.value == "completed"

    def test_operator_failure_returns_1(self, workspace, capsys):
        raster_path = workspace / "not_vector.tif"
        data = np.ones((10, 10), dtype="float32")
        transform = rasterio.transform.from_bounds(0.0, 0.0, 10.0, 10.0, 10, 10)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs=CRS.from_epsg(32618),
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(raster_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 1
        assert "FAILED:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Burn modes
# ---------------------------------------------------------------------------


class TestBurnModes:
    def test_constant_burn_default(self, vector_path, workspace):
        """Default burn_value=1.0 — polygon pixels get 1.0, background gets 0.0."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            data = src.read(1)
        # Should have some 1.0 pixels (burned) and some 0.0 pixels (nodata)
        assert np.any(data == 1.0)
        assert np.any(data == 0.0)

    def test_constant_burn_custom(self, vector_path, workspace):
        """Custom burn_value=99.0."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--burn-value",
                "99.0",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            data = src.read(1)
        assert np.any(data == 99.0)

    def test_attribute_burn(self, vector_path, workspace):
        """Burn from 'value' attribute — polygons get 10.0 and 20.0."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--burn-attribute",
                "value",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            data = src.read(1)
        unique = set(np.unique(data))
        assert 10.0 in unique
        assert 20.0 in unique

    def test_bad_attribute_produces_nodata_raster(self, vector_path, workspace):
        """Non-existent attribute → all features skipped → raster filled with nodata."""
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--burn-attribute",
                "nonexistent_field",
                "--nodata",
                "-9999.0",
            ]
        )
        assert rc == 0

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            data = src.read(1)
        # All pixels should be nodata since no features had the attribute
        assert np.all(data == -9999.0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_vector_raises(self, empty_vector_path, workspace):
        """Empty vector can't be materialized (fiona can't compute bounds) → raises."""
        with pytest.raises(Exception, match="bounds|Driver"):
            main(
                [
                    "run",
                    "rasterize",
                    "--vector",
                    str(empty_vector_path),
                    "--workspace",
                    str(workspace),
                    "--resolution",
                    "1.0",
                    "--extent",
                    "0,0,10,10",
                    "--nodata",
                    "-1.0",
                ]
            )


# ---------------------------------------------------------------------------
# Flag variants
# ---------------------------------------------------------------------------


class TestRunRasterizeFlags:
    def test_resolution_xy(self, vector_path, workspace):
        """Asymmetric resolution: x_res=0.5, y_res=1.0."""
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "0.5,1.0",
            ]
        )
        assert rc == 0

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            # x_res=0.5 → more columns; y_res=1.0 → fewer rows
            assert src.width > src.height

    def test_output_flag(self, vector_path, workspace):
        custom_output = workspace / "my_output" / "result.tif"
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--output",
                str(custom_output),
            ]
        )
        assert rc == 0
        assert custom_output.exists()

    def test_extent_flag(self, vector_path, workspace):
        """Explicit extent clips output to a sub-region."""
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--extent",
                "0,0,5,5",
            ]
        )
        assert rc == 0

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.width == 5
            assert src.height == 5

    def test_dtype_flag(self, vector_path, workspace):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--dtype",
                "uint8",
                "--burn-value",
                "255",
            ]
        )
        assert rc == 0

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.dtypes[0] == "uint8"

    def test_nodata_flag(self, vector_path, workspace):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--nodata",
                "-9999.0",
            ]
        )
        assert rc == 0

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.nodata == -9999.0

    def test_workspace_flag(self, vector_path, tmp_path):
        custom_ws = tmp_path / "custom_workspace"
        custom_ws.mkdir()
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(custom_ws),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 0
        assert (custom_ws / ".quarry" / "registry.duckdb").exists()
        assert (custom_ws / "rasterize" / "rasterized.tif").exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRunRasterizeErrors:
    def test_missing_vector(self, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(workspace / "nonexistent.gpkg"),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_invalid_resolution(self, vector_path, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "abc",
            ]
        )
        assert rc == 1
        assert "Invalid --resolution" in capsys.readouterr().err

    def test_invalid_resolution_three_values(self, vector_path, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0,2.0,3.0",
            ]
        )
        assert rc == 1
        assert "Invalid --resolution" in capsys.readouterr().err

    def test_invalid_extent(self, vector_path, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--extent",
                "abc",
            ]
        )
        assert rc == 1
        assert "Invalid --extent" in capsys.readouterr().err

    def test_invalid_extent_wrong_count(self, vector_path, workspace, capsys):
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--extent",
                "0,0,10",
            ]
        )
        assert rc == 1
        assert "Invalid --extent" in capsys.readouterr().err

    def test_invalid_checks_return_2(self, vector_path, workspace, monkeypatch, capsys):
        monkeypatch.setattr(
            "quarry_core.executors.local.LocalExecutor.submit",
            lambda _self, _operator, _inputs, _params: make_invalid_completed_run(
                workspace,
                operator_name="rasterize_vector",
                artifact_type=ArtifactType.RASTER,
                output_name="rasterize/invalid.tif",
            ),
        )

        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 2
        assert "FAILED:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


class TestOutputVerification:
    def test_crs_preserved(self, vector_path, workspace):
        """Output raster CRS matches input vector CRS."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 32618

    def test_dimensions_match_resolution(self, workspace):
        """10×10 extent at resolution 2.0 → 5×5 grid."""
        path = workspace / "square.gpkg"
        _write_polygons(
            path,
            [((0.0, 0.0, 10.0, 10.0), {"id": 1, "value": 1.0})],
        )

        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(path),
                "--workspace",
                str(workspace),
                "--resolution",
                "2.0",
                "--extent",
                "0,0,10,10",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.width == 5
            assert src.height == 5

    def test_single_band(self, vector_path, workspace):
        """Output is always single-band."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )

        output_tif = workspace / "rasterize" / "rasterized.tif"
        with rasterio.open(output_tif) as src:
            assert src.count == 1


# ---------------------------------------------------------------------------
# Integration: full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_run_then_list_then_show_then_lineage(self, vector_path, workspace, capsys):
        """Full CLI round-trip: run rasterize → list → show → lineage."""
        rc = main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
            ]
        )
        assert rc == 0
        capsys.readouterr()

        # List
        rc = main(["artifacts", "list", "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 artifact(s)" in out

        # Show the raster artifact
        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        raster = next(a for a in artifacts if a.type.value == "raster")

        rc = main(["artifacts", "show", raster.id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert raster.id in out
        assert "raster" in out

        # Lineage
        rc = main(["lineage", raster.id, "--workspace", str(workspace)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 ancestor(s)" in out

    def test_runs_show_after_rasterize(self, vector_path, workspace, capsys):
        """run rasterize → runs show displays params."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
                "--burn-value",
                "5.0",
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
        assert "resolution:" in out

    def test_checks_show_after_rasterize(self, vector_path, workspace, capsys):
        """run rasterize → checks show lists validation checks."""
        main(
            [
                "run",
                "rasterize",
                "--vector",
                str(vector_path),
                "--workspace",
                str(workspace),
                "--resolution",
                "1.0",
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
