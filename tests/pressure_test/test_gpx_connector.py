"""
Pressure test: GPXConnector.

Lane: connector

Validates GPX file materialization:
- source_ref parsing (local path with optional ::layer suffix)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- layer selection: default picks most populated, explicit :: selection
- discover: layers in file, files in directory
- metadata: per-layer info, CRS
- error handling: nonexistent files, bad layer names, non-gpx files
"""

from __future__ import annotations

import pytest
from quarry_connectors.gpx import GPXConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_gpx_file(path, waypoints=None, tracks=None, routes=None):
    """Create a GPX file by writing XML directly.

    Fiona's GPX driver is read-only, so we must write XML directly.

    Args:
        path: Output file path
        waypoints: List of dicts with lat, lon, name, ele (optional)
        tracks: List of dicts with name and segments (list of point lists)
        routes: List of dicts with name and points (list of point dicts)
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="test"',
        '     xmlns="http://www.topografix.com/GPX/1/1">',
    ]

    # Add waypoints
    if waypoints:
        for wpt in waypoints:
            lat = wpt.get("lat", 0.0)
            lon = wpt.get("lon", 0.0)
            name = wpt.get("name", "")
            ele = wpt.get("ele")
            lines.append(f'  <wpt lat="{lat}" lon="{lon}">')
            if name:
                lines.append(f"    <name>{name}</name>")
            if ele is not None:
                lines.append(f"    <ele>{ele}</ele>")
            lines.append("  </wpt>")

    # Add tracks
    if tracks:
        for trk in tracks:
            name = trk.get("name", "")
            lines.append("  <trk>")
            if name:
                lines.append(f"    <name>{name}</name>")
            for segment in trk.get("segments", []):
                lines.append("    <trkseg>")
                for pt in segment:
                    lat = pt.get("lat", 0.0)
                    lon = pt.get("lon", 0.0)
                    ele = pt.get("ele")
                    ele_str = f"<ele>{ele}</ele>" if ele is not None else ""
                    lines.append(f'      <trkpt lat="{lat}" lon="{lon}">{ele_str}</trkpt>')
                lines.append("    </trkseg>")
            lines.append("  </trk>")

    # Add routes
    if routes:
        for rte in routes:
            name = rte.get("name", "")
            lines.append("  <rte>")
            if name:
                lines.append(f"    <name>{name}</name>")
            for pt in rte.get("points", []):
                lat = pt.get("lat", 0.0)
                lon = pt.get("lon", 0.0)
                ele = pt.get("ele")
                ele_str = f"<ele>{ele}</ele>" if ele is not None else ""
                lines.append(f'    <rtept lat="{lat}" lon="{lon}">{ele_str}</rtept>')
            lines.append("  </rte>")

    lines.append("</gpx>")

    path.write_text("\n".join(lines))


@pytest.fixture()
def gpx_file_all_layers(tmp_path):
    """Create a GPX file with waypoints, tracks, and routes."""
    path = tmp_path / "test_all.gpx"

    waypoints = [
        {"lat": 47.5, "lon": -122.3, "name": "WP1", "ele": 100.0},
        {"lat": 47.6, "lon": -122.2, "name": "WP2", "ele": 150.0},
    ]

    tracks = [
        {
            "name": "Track1",
            "segments": [
                [
                    {"lat": 47.5, "lon": -122.3, "ele": 100.0},
                    {"lat": 47.6, "lon": -122.2, "ele": 150.0},
                    {"lat": 47.7, "lon": -122.1, "ele": 200.0},
                ]
            ],
        }
    ]

    routes = [
        {
            "name": "Route1",
            "points": [
                {"lat": 47.5, "lon": -122.3, "ele": 100.0},
                {"lat": 47.8, "lon": -122.0, "ele": 250.0},
            ],
        }
    ]

    _create_gpx_file(path, waypoints=waypoints, tracks=tracks, routes=routes)
    return path


@pytest.fixture()
def gpx_file_waypoints_only(tmp_path):
    """Create a GPX file with only waypoints."""
    path = tmp_path / "test_waypoints.gpx"

    waypoints = [
        {"lat": 40.7, "lon": -74.0, "name": "NYC", "ele": 10.0},
        {"lat": 34.0, "lon": -118.2, "name": "LA", "ele": 50.0},
        {"lat": 41.8, "lon": -87.6, "name": "Chicago", "ele": 180.0},
    ]

    _create_gpx_file(path, waypoints=waypoints)
    return path


@pytest.fixture()
def gpx_file_tracks_only(tmp_path):
    """Create a GPX file with only tracks."""
    path = tmp_path / "test_tracks.gpx"

    tracks = [
        {
            "name": "Hike1",
            "segments": [
                [
                    {"lat": 45.5, "lon": -121.0, "ele": 500.0},
                    {"lat": 45.6, "lon": -121.1, "ele": 600.0},
                ]
            ],
        },
        {
            "name": "Hike2",
            "segments": [
                [
                    {"lat": 45.7, "lon": -121.2, "ele": 700.0},
                    {"lat": 45.8, "lon": -121.3, "ele": 800.0},
                    {"lat": 45.9, "lon": -121.4, "ele": 900.0},
                ]
            ],
        },
    ]

    _create_gpx_file(path, tracks=tracks)
    return path


@pytest.fixture()
def directory_with_gpx(tmp_path, gpx_file_all_layers, gpx_file_waypoints_only):
    """Create a directory with multiple GPX files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestGPXEagerLocal:
    """Validate eager materialization of local GPX files."""

    def test_eager_produces_vector(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_eager_produces_local_file_backing(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpx")

    def test_eager_wrapped_local_strategy(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        # track_points has 3 features (one per track point), which is most
        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(-122.3, abs=0.1)
        assert ymin == pytest.approx(47.5, abs=0.1)

    def test_eager_crs(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_eager_content_hash_present(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.lineage.params["source"] == "gpx"
        assert result.artifact.lineage.params["path"] == str(gpx_file_all_layers)
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata_schema(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert "schema" in result.artifact.metadata
        assert "selected_layer" in result.artifact.metadata
        assert "available_layers" in result.artifact.metadata


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestGPXLazyLocal:
    """Validate lazy (metadata-only) materialization of local GPX files."""

    def test_lazy_backing_kind(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        # track_points has 3 features (one per track point), which is most
        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert len(extent) == 4

    def test_lazy_backing_uri(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        assert "::" in result.artifact.backing.uri
        # track_points has most features (3) so it's selected by default
        assert "track_points" in result.artifact.backing.uri

    def test_lazy_lineage(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Layer Selection
# ---------------------------------------------------------------------------


class TestGPXLayerSelection:
    """Validate layer selection behavior."""

    def test_default_layer_picks_most_populated(self, gpx_file_all_layers, tmp_path):
        """Default layer should be track_points (3 points) since it has most features."""
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        # track_points has 3 features (one per track point), which is most
        assert result.artifact.metadata["selected_layer"] == "track_points"
        assert result.artifact.spatial.feature_count == 3

    def test_default_layer_waypoints_when_most(self, gpx_file_waypoints_only, tmp_path):
        """When waypoints is the only layer with features, select it."""
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_waypoints_only), tmp_path)

        assert result.artifact.metadata["selected_layer"] == "waypoints"
        assert result.artifact.spatial.feature_count == 3

    def test_explicit_layer_selection(self, gpx_file_all_layers, tmp_path):
        """Explicit ::waypoints should select waypoints layer."""
        connector = GPXConnector()
        source_ref = f"{gpx_file_all_layers}::waypoints"
        result = connector.materialize(source_ref, tmp_path)

        assert result.artifact.metadata["selected_layer"] == "waypoints"
        assert result.artifact.spatial.feature_count == 2

    def test_explicit_tracks_layer(self, gpx_file_all_layers, tmp_path):
        """Explicit ::tracks should select tracks layer."""
        connector = GPXConnector()
        source_ref = f"{gpx_file_all_layers}::tracks"
        result = connector.materialize(source_ref, tmp_path)

        assert result.artifact.metadata["selected_layer"] == "tracks"

    def test_explicit_routes_layer(self, gpx_file_all_layers, tmp_path):
        """Explicit ::routes should select routes layer."""
        connector = GPXConnector()
        source_ref = f"{gpx_file_all_layers}::routes"
        result = connector.materialize(source_ref, tmp_path)

        assert result.artifact.metadata["selected_layer"] == "routes"
        # routes layer has 1 feature (the route itself), route_points has 2
        assert result.artifact.spatial.feature_count == 1

    def test_available_layers_in_metadata(self, gpx_file_all_layers, tmp_path):
        """Metadata should include available_layers list."""
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        available = result.artifact.metadata["available_layers"]
        assert "waypoints" in available
        assert "tracks" in available
        assert "routes" in available

    def test_layer_counts_in_metadata(self, gpx_file_all_layers, tmp_path):
        """Metadata should include per-layer feature counts."""
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        counts = result.artifact.metadata["layer_counts"]
        # waypoints: 2 waypoints
        # tracks: 1 track (the track itself)
        # track_points: 3 track points (individual points in the track)
        # routes: 1 route (the route itself)
        # route_points: 2 route points (individual points in the route)
        assert counts["waypoints"] == 2
        assert counts["tracks"] == 1
        assert counts["track_points"] == 3
        assert counts["routes"] == 1
        assert counts["route_points"] == 2


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestGPXDiscover:
    """Validate discovery functionality."""

    def test_discover_layers_in_file(self, gpx_file_all_layers):
        """Discover on a GPX file should list its layers."""
        connector = GPXConnector()
        entries = connector.discover(str(gpx_file_all_layers))

        names = {e.name for e in entries}
        assert "waypoints" in names
        assert "tracks" in names
        assert "routes" in names

    def test_discover_layer_source_refs(self, gpx_file_all_layers):
        """Layer entries should have ::layer suffix in source_ref."""
        connector = GPXConnector()
        entries = connector.discover(str(gpx_file_all_layers))

        for entry in entries:
            assert "::" in str(entry.source_ref)

    def test_discover_layer_feature_counts(self, gpx_file_all_layers):
        """Layer entries should include feature counts."""
        connector = GPXConnector()
        entries = connector.discover(str(gpx_file_all_layers))

        for entry in entries:
            assert "feature_count" in entry.metadata

    def test_discover_files_in_directory(self, directory_with_gpx):
        """Discover on a directory should list GPX files."""
        connector = GPXConnector()
        entries = connector.discover(str(directory_with_gpx))

        names = {e.name for e in entries}
        assert "test_all" in names
        assert "test_waypoints" in names

    def test_discover_file_metadata_includes_layers(self, directory_with_gpx):
        """File entries should include available_layers in metadata."""
        connector = GPXConnector()
        entries = connector.discover(str(directory_with_gpx))

        for entry in entries:
            assert "available_layers" in entry.metadata
            assert "layer_counts" in entry.metadata

    def test_discover_empty_directory(self, tmp_path):
        """Discover on empty directory should return empty list."""
        connector = GPXConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_with_dict_query(self, directory_with_gpx):
        """Discover should accept dict with path key."""
        connector = GPXConnector()
        entries = connector.discover({"path": str(directory_with_gpx)})

        assert len(entries) >= 2


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestGPXMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_available_layers(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(str(gpx_file_all_layers))

        assert "available_layers" in meta
        assert "waypoints" in meta["available_layers"]

    def test_metadata_returns_layer_counts(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(str(gpx_file_all_layers))

        assert "layer_counts" in meta
        assert meta["layer_counts"]["waypoints"] == 2

    def test_metadata_returns_crs(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(str(gpx_file_all_layers))

        assert meta["crs"] == "EPSG:4326"

    def test_metadata_returns_driver(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(str(gpx_file_all_layers))

        assert meta["driver"] == "GPX"

    def test_metadata_per_layer_info(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(str(gpx_file_all_layers))

        assert "layers" in meta
        assert "waypoints" in meta["layers"]
        assert "tracks" in meta["layers"]

    def test_metadata_with_explicit_layer(self, gpx_file_all_layers):
        connector = GPXConnector()
        meta = connector.metadata(f"{gpx_file_all_layers}::waypoints")

        assert meta["selected_layer"] == "waypoints"
        assert meta["feature_count"] == 2


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestGPXErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = GPXConnector()
        nonexistent = tmp_path / "does_not_exist.gpx"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_bad_layer_name_raises(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        bad_ref = f"{gpx_file_all_layers}::nonexistent_layer"

        with pytest.raises(MaterializeError):
            connector.materialize(bad_ref, tmp_path)

    def test_non_gpx_file_raises(self, tmp_path):
        connector = GPXConnector()
        bad_file = tmp_path / "not_gpx.txt"
        bad_file.write_text("this is not a gpx file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = GPXConnector()
        nonexistent = tmp_path / "does_not_exist.gpx"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_nonexistent_path_raises(self, tmp_path):
        connector = GPXConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = GPXConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestGPXSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        ref = SourceRef.local(str(gpx_file_all_layers))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, gpx_file_all_layers, tmp_path):
        connector = GPXConnector()
        result = connector.materialize(str(gpx_file_all_layers), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_sourceref_with_layer(self, gpx_file_all_layers, tmp_path):
        """SourceRef with ::layer in raw should work."""
        connector = GPXConnector()
        ref = SourceRef.local(f"{gpx_file_all_layers}::waypoints")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.metadata["selected_layer"] == "waypoints"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestGPXCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = GPXConnector()
        assert connector.name == "gpx"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = GPXConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
