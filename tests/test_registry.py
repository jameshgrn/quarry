"""Tests for the artifact registry."""

import pytest

from georuntime.registry import VALID_ARTIFACT_TYPES, Registry


@pytest.fixture
def temp_registry(tmp_path):
    """Create a registry in a temp workspace."""
    return Registry(str(tmp_path))


class TestRegistry:
    """Test cases for Registry class."""

    def test_register_returns_uuid(self, temp_registry):
        """Register should return a UUID string."""
        artifact_id = temp_registry.register(
            {
                "name": "test",
                "artifact_type": "raster",
                "path": "/tmp/test.tif",
            }
        )
        assert len(artifact_id) == 36  # UUID format
        assert artifact_id.count("-") == 4

    def test_register_validates_artifact_type(self, temp_registry):
        """Register should reject invalid artifact types."""
        with pytest.raises(ValueError) as exc_info:
            temp_registry.register(
                {
                    "name": "test",
                    "artifact_type": "invalid",
                    "path": "/tmp/test.tif",
                }
            )
        assert "invalid" in str(exc_info.value).lower()
        assert "raster" in str(exc_info.value).lower()

    def test_get_returns_artifact(self, temp_registry):
        """Get should return the registered artifact."""
        aid = temp_registry.register(
            {
                "name": "my_raster",
                "artifact_type": "vector",
                "path": "/data/roads.shp",
                "crs": "EPSG:4326",
                "feature_count": 100,
            }
        )

        artifact = temp_registry.get(aid)
        assert artifact is not None
        assert artifact["name"] == "my_raster"
        assert artifact["artifact_type"] == "vector"
        assert artifact["crs"] == "EPSG:4326"
        assert artifact["feature_count"] == 100

    def test_get_returns_none_for_missing(self, temp_registry):
        """Get should return None for non-existent artifact."""
        artifact = temp_registry.get("00000000-0000-0000-0000-000000000000")
        assert artifact is None

    def test_list_returns_all(self, temp_registry):
        """List should return all artifacts."""
        temp_registry.register({"name": "a", "artifact_type": "raster", "path": "/a.tif"})
        temp_registry.register({"name": "b", "artifact_type": "vector", "path": "/b.shp"})

        artifacts = temp_registry.list()
        assert len(artifacts) == 2

    def test_list_filters_by_type(self, temp_registry):
        """List should filter by artifact type."""
        temp_registry.register({"name": "r1", "artifact_type": "raster", "path": "/r1.tif"})
        temp_registry.register({"name": "v1", "artifact_type": "vector", "path": "/v1.shp"})
        temp_registry.register({"name": "r2", "artifact_type": "raster", "path": "/r2.tif"})

        rasters = temp_registry.list("raster")
        vectors = temp_registry.list("vector")

        assert len(rasters) == 2
        assert len(vectors) == 1
        assert all(a["artifact_type"] == "raster" for a in rasters)

    def test_exists_checks_by_path(self, temp_registry):
        """Exists should check by resolved path."""
        temp_registry.register(
            {
                "name": "test",
                "artifact_type": "raster",
                "path": "/tmp/test.tif",
            }
        )

        assert temp_registry.exists("/tmp/test.tif")
        assert not temp_registry.exists("/tmp/other.tif")

    def test_json_fields_parsed(self, temp_registry):
        """JSON fields should be parsed back to objects."""
        aid = temp_registry.register(
            {
                "name": "test",
                "artifact_type": "raster",
                "path": "/tmp/test.tif",
                "extent": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "metadata": {"key": "value"},
            }
        )

        artifact = temp_registry.get(aid)
        assert artifact["extent"] == {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        assert artifact["metadata"] == {"key": "value"}


class TestValidArtifactTypes:
    """Test cases for artifact type validation."""

    def test_valid_types(self):
        """Should have expected valid types."""
        assert VALID_ARTIFACT_TYPES == {"vector", "raster", "table", "preview", "summary"}
