"""Pressure test: generic operator dispatch via CLI.

Lane: adapter

Exercises:
  - Operator registry: all 12 names registered, name↔class match, lazy import
  - Param coercion: str, int, float, bool, Optional, list, tuple, Literal
  - Generic dispatch end-to-end: slope, zonal_stats, rasterize_vector, reproject,
    spatial_join, build_cog, clip_raster
  - Error paths: missing input, too few/many inputs, bad param format, unknown operator
  - Registry integration: artifacts + run persisted, lineage correct

Failure signals:
  - Unknown operator accepted silently
  - Param coercion produces wrong type
  - Generic dispatch returns non-zero on valid input
  - Registry not populated after generic dispatch
  - Existing named flows (hydrology, zonal, sample, rasterize) broken
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import fiona
import numpy as np
import pytest
import rasterio
from quarry_cli.main import _build_params, _coerce_value, main
from quarry_core.artifact import ArtifactType
from quarry_operators.registry import OPERATOR_NAMES, get_operator, get_params_class
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, mapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_raster(
    path: Path,
    data: np.ndarray,
    nodata: float | None = None,
    crs_epsg: int = 32618,
) -> Path:
    """Write a single-band float32 GeoTIFF."""
    h, w = data.shape
    transform = from_bounds(0.0, 0.0, float(w), float(h), w, h)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)
    return path


def _write_vector(
    path: Path,
    geometries: list,
    geometry_type: str = "Polygon",
    properties: list[dict] | None = None,
    crs_epsg: int = 32618,
) -> Path:
    """Write geometries to GeoPackage."""
    schema = {"geometry": geometry_type, "properties": {}}
    if properties and len(properties) > 0:
        for k, v in properties[0].items():
            schema["properties"][k] = "float" if isinstance(v, (int, float)) else "str"
    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for i, geom in enumerate(geometries):
            props = properties[i] if properties else {}
            dst.write({"geometry": mapping(geom), "properties": props})
    return path


def _bowl_dem(size: int = 20) -> np.ndarray:
    """Bowl-shaped DEM: center is lowest, edges are highest."""
    y, x = np.mgrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    return ((x - cx) ** 2 + (y - cy) ** 2).astype("float32")


@pytest.fixture()
def workspace(tmp_path):
    return tmp_path


@pytest.fixture()
def dem_path(workspace):
    path = workspace / "dem.tif"
    _write_raster(path, _bowl_dem())
    return path


@pytest.fixture()
def raster_path(workspace):
    path = workspace / "raster.tif"
    data = np.arange(1, 101, dtype="float32").reshape(10, 10)
    _write_raster(path, data)
    return path


@pytest.fixture()
def zones_path(workspace):
    path = workspace / "zones.gpkg"
    left = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    right = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    _write_vector(path, [left, right])
    return path


@pytest.fixture()
def points_path(workspace):
    path = workspace / "points.gpkg"
    pts = [Point(2.5, 2.5), Point(7.5, 7.5)]
    _write_vector(path, pts, geometry_type="Point")
    return path


@pytest.fixture()
def polygons_path(workspace):
    """Two overlapping polygons with numeric attribute for rasterization."""
    path = workspace / "polygons.gpkg"
    p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    p2 = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])
    _write_vector(path, [p1, p2], properties=[{"val": 10.0}, {"val": 20.0}])
    return path


# ===========================================================================
# A. Operator Registry Tests
# ===========================================================================


class TestOperatorRegistry:
    def test_all_twelve_names_registered(self):
        assert len(OPERATOR_NAMES) == 12

    def test_operator_names_match_classes(self):
        for name in OPERATOR_NAMES:
            op = get_operator(name)
            assert op.name == name, f"{name}: class.name={op.name!r}"

    def test_params_class_exists_for_each(self):
        for name in OPERATOR_NAMES:
            cls = get_params_class(name)
            assert cls is not None

    def test_unknown_operator_raises(self):
        with pytest.raises(KeyError, match="Unknown operator"):
            get_operator("bogus")

    def test_parser_has_all_operator_subcommands(self):
        from quarry_cli.main import build_parser

        parser = build_parser()
        # Dig into run sub-subcommands
        run_action = None
        for action in parser._subparsers._actions:
            if hasattr(action, "_parser_class"):
                for name, sub in action.choices.items():
                    if name == "run":
                        run_action = sub
                        break
        assert run_action is not None
        run_choices = set()
        for action in run_action._subparsers._actions:
            if hasattr(action, "choices") and action.choices:
                run_choices.update(action.choices.keys())
        for name in OPERATOR_NAMES:
            assert name in run_choices, f"Operator {name!r} missing from 'run' subparsers"


# ===========================================================================
# B. Param Coercion Tests
# ===========================================================================


class TestCoerceValue:
    def test_str(self):
        assert _coerce_value("hello", str) == "hello"

    def test_int(self):
        assert _coerce_value("42", int) == 42

    def test_float(self):
        assert _coerce_value("3.14", float) == pytest.approx(3.14)

    def test_bool_true(self):
        for val in ("true", "True", "1", "yes"):
            assert _coerce_value(val, bool) is True

    def test_bool_false(self):
        for val in ("false", "False", "0", "no"):
            assert _coerce_value(val, bool) is False

    def test_bool_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _coerce_value("maybe", bool)

    def test_float_or_none_value(self):
        result = _coerce_value("3.14", float | None)
        assert result == pytest.approx(3.14)

    def test_float_or_none_none(self):
        result = _coerce_value("none", float | None)
        assert result is None

    def test_str_or_none_value(self):
        result = _coerce_value("hello", str | None)
        assert result == "hello"

    def test_str_or_none_none(self):
        result = _coerce_value("None", str | None)
        assert result is None

    def test_list_int(self):
        assert _coerce_value("1,2,3", list[int]) == [1, 2, 3]

    def test_list_int_empty(self):
        assert _coerce_value("", list[int]) == []

    def test_tuple_two_floats(self):
        result = _coerce_value("1.0,2.0", tuple[float, float])
        assert result == pytest.approx((1.0, 2.0))

    def test_tuple_four_floats(self):
        result = _coerce_value("1,2,3,4", tuple[float, float, float, float])
        assert result == pytest.approx((1.0, 2.0, 3.0, 4.0))

    def test_tuple_wrong_count_raises(self):
        with pytest.raises(ValueError, match="Expected 2"):
            _coerce_value("1,2,3", tuple[float, float])

    def test_literal_valid(self):
        result = _coerce_value("degrees", Literal["degrees", "percent"])
        assert result == "degrees"

    def test_literal_invalid_raises(self):
        with pytest.raises(ValueError, match="not in allowed values"):
            _coerce_value("bogus", Literal["degrees", "percent"])


class TestBuildParams:
    def test_slope_params(self):
        from quarry_operators.slope import SlopeParams

        result = _build_params(SlopeParams, {"units": "percent"}, "/tmp/out.tif")
        assert result.output_path == "/tmp/out.tif"
        assert result.units == "percent"

    def test_unknown_key_raises(self):
        from quarry_operators.slope import SlopeParams

        with pytest.raises(ValueError, match="Unknown parameter"):
            _build_params(SlopeParams, {"bogus_key": "val"}, "/tmp/out.tif")

    def test_rasterize_resolution(self):
        from quarry_operators.rasterize_vector import RasterizeVectorParams

        result = _build_params(RasterizeVectorParams, {"resolution": "1.0,2.0"}, "/tmp/out.tif")
        assert result.resolution == pytest.approx((1.0, 2.0))

    def test_fill_depressions_bool(self):
        from quarry_operators.fill_depressions import FillDepressionsParams

        result = _build_params(FillDepressionsParams, {"apply_gradient": "false"}, "/tmp/out.tif")
        assert result.apply_gradient is False


# ===========================================================================
# C. End-to-End Execution Tests
# ===========================================================================


class TestGenericDispatchE2E:
    def test_slope(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
                "-p",
                "units=degrees",
            ]
        )
        assert rc == 0

    def test_aspect(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "aspect",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

    def test_build_cog(self, workspace):
        # Need 512×512 so GDAL generates overviews (is_cog requires tiling + overviews)
        big_dem = workspace / "big_dem.tif"
        _write_raster(big_dem, np.ones((512, 512), dtype="float32"))
        rc = main(
            [
                "run",
                "build_cog",
                "--input",
                str(big_dem),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

    def test_fill_depressions(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "fill_depressions",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

    def test_zonal_stats(self, raster_path, zones_path, workspace):
        rc = main(
            [
                "run",
                "zonal_stats",
                "--input",
                str(raster_path),
                "--input",
                str(zones_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

    def test_rasterize_vector(self, polygons_path, workspace):
        rc = main(
            [
                "run",
                "rasterize_vector",
                "--input",
                str(polygons_path),
                "--workspace",
                str(workspace),
                "-p",
                "resolution=1.0,1.0",
            ]
        )
        assert rc == 0

    def test_reproject(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "reproject",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
                "-p",
                "target_crs=EPSG:4326",
            ]
        )
        assert rc == 0

    def test_spatial_join(self, workspace):
        left_path = workspace / "left.gpkg"
        right_path = workspace / "right.gpkg"
        p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        p2 = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])
        _write_vector(left_path, [p1])
        _write_vector(right_path, [p2], properties=[{"name": "zone_a"}])
        rc = main(
            [
                "run",
                "spatial_join",
                "--input",
                str(left_path),
                "--input",
                str(right_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0

    def test_clip_raster_with_bounds(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "clip_raster",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
                "-p",
                "bounds=2,2,10,10",
            ]
        )
        assert rc == 0

    def test_custom_output_path(self, dem_path, workspace):
        custom_out = workspace / "custom_slope.tif"
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--output",
                str(custom_out),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        assert custom_out.exists()


# ===========================================================================
# D. Error Path Tests
# ===========================================================================


class TestGenericDispatchErrors:
    def test_missing_input_file(self, workspace):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(workspace / "nonexistent.tif"),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1

    def test_too_few_inputs(self, raster_path, workspace, capsys):
        rc = main(
            [
                "run",
                "zonal_stats",
                "--input",
                str(raster_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "at least 2" in capsys.readouterr().err

    def test_too_many_inputs(self, dem_path, workspace, capsys):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 1
        assert "at most 1" in capsys.readouterr().err

    def test_bad_param_format(self, dem_path, workspace, capsys):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
                "-p",
                "noequals",
            ]
        )
        assert rc == 1
        assert "key=value" in capsys.readouterr().err

    def test_unknown_param_key(self, dem_path, workspace, capsys):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
                "-p",
                "bogus=val",
            ]
        )
        assert rc == 1
        assert "Unknown parameter" in capsys.readouterr().err


# ===========================================================================
# E. Registry Integration Tests
# ===========================================================================


class TestGenericDispatchRegistry:
    def test_artifacts_persisted(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        registry = Registry(workspace)
        artifacts = registry.list_artifacts()
        assert len(artifacts) == 2  # input + output
        types = {a.type for a in artifacts}
        assert ArtifactType.RASTER in types

    def test_run_persisted(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        registry = Registry(workspace)
        runs = registry.list_runs()
        assert len(runs) == 1
        assert runs[0].operator_name == "slope"
        assert runs[0].status.value == "completed"

    def test_lineage_correct(self, dem_path, workspace):
        rc = main(
            [
                "run",
                "slope",
                "--input",
                str(dem_path),
                "--workspace",
                str(workspace),
            ]
        )
        assert rc == 0
        registry = Registry(workspace)
        runs = registry.list_runs()
        assert len(runs) == 1
        output_artifact = runs[0].output.artifact
        chain = registry.get_full_lineage(output_artifact.id)
        assert len(chain) >= 1  # at least the input DEM ancestor
