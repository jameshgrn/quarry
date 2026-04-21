"""Tests for file inspection."""

import tempfile

import fiona
import fiona.crs
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from georuntime.core.inspect import inspect_file


class TestInspectRaster:
    """Test raster file inspection."""

    def test_inspect_raster(self, tmp_path):
        """Should extract raster metadata."""
        raster_path = tmp_path / "test.tif"

        # Create test raster
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=rasterio.Affine.scale(0.001, 0.001),
        ) as dst:
            data = np.zeros((100, 100), dtype=np.uint8)
            dst.write_band(1, data)
            dst.write_band(2, data)
            dst.write_band(3, data)

        artifact = inspect_file(str(raster_path))

        assert artifact["artifact_type"] == "raster"
        assert artifact["name"] == "test"
        assert artifact["crs"] == "EPSG:4326"
        assert artifact["band_count"] == 3
        assert artifact["driver"] == "GTiff"
        assert artifact["feature_count"] is None


class TestInspectVector:
    """Test vector file inspection."""

    def test_inspect_geojson(self, tmp_path):
        """Should extract GeoJSON metadata."""
        geojson_path = tmp_path / "points.geojson"

        schema = {"geometry": "Point", "properties": {"name": "str"}}
        with fiona.open(
            geojson_path,
            "w",
            driver="GeoJSON",
            crs=fiona.crs.CRS.from_epsg(4326),
            schema=schema,
        ) as dst:
            dst.write(
                {"geometry": {"type": "Point", "coordinates": (0, 0)}, "properties": {"name": "a"}}
            )
            dst.write(
                {"geometry": {"type": "Point", "coordinates": (1, 1)}, "properties": {"name": "b"}}
            )

        artifact = inspect_file(str(geojson_path))

        assert artifact["artifact_type"] == "vector"
        assert artifact["name"] == "points"
        assert artifact["crs"] == "EPSG:4326"
        assert artifact["feature_count"] == 2
        assert artifact["driver"] == "GeoJSON"
        assert artifact["band_count"] is None


class TestInspectErrors:
    """Test error handling."""

    def test_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                inspect_file(f"{tmp}/missing.tif")

    def test_unsupported_extension(self, tmp_path):
        """Should raise ValueError for unsupported extensions."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a geospatial file")

        with pytest.raises(ValueError) as exc_info:
            inspect_file(str(txt_path))

        assert "unsupported" in str(exc_info.value).lower()


class TestInspectFixtures:
    """Test inspect on static fixture files."""

    def test_inspect_raster_fixture(self):
        """Should inspect the sample TIFF fixture."""
        from pathlib import Path

        fixture_path = Path(__file__).parent / "fixtures" / "sample.tif"

        result = inspect_file(str(fixture_path))

        assert result["artifact_type"] == "raster"
        assert result["name"] == "sample"
        assert result["crs"] == "EPSG:32618"
        assert result["band_count"] == 1
        assert result["driver"] == "GTiff"

    def test_inspect_vector_fixture(self):
        """Should inspect the sample GeoJSON fixture."""
        from pathlib import Path

        fixture_path = Path(__file__).parent / "fixtures" / "test.geojson"

        result = inspect_file(str(fixture_path))

        assert result["artifact_type"] == "vector"
        assert result["name"] == "test"
        assert result["crs"] == "EPSG:4326"
        assert result["driver"] == "GeoJSON"
        assert result["metadata"]["geometry"] == "Point"
