"""Tests for preview generation."""

from pathlib import Path

from PIL import Image

from georuntime.core.preview import preview_artifact


class TestPreviewFixtures:
    """Test preview on static fixture files."""

    def test_preview_raster_fixture(self, tmp_path):
        """Should generate PNG preview from raster."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample.tif"
        output_path = tmp_path / "raster_preview.png"

        result = preview_artifact(str(fixture_path), str(output_path))

        assert result["name"] == "raster_preview"
        assert result["driver"] == "PNG"
        assert result["width"] > 0
        assert result["height"] > 0
        assert output_path.exists()

        # Verify it's a valid PNG
        with Image.open(output_path) as img:
            assert img.format == "PNG"

    def test_preview_vector_fixture(self, tmp_path):
        """Should generate PNG preview from vector."""
        fixture_path = Path(__file__).parent / "fixtures" / "test.geojson"
        output_path = tmp_path / "vector_preview.png"

        result = preview_artifact(str(fixture_path), str(output_path))

        assert result["name"] == "vector_preview"
        assert result["driver"] == "PNG"
        assert result["width"] > 0
        assert result["height"] > 0
        assert output_path.exists()

        # Verify it's a valid PNG
        with Image.open(output_path) as img:
            assert img.format == "PNG"
