"""TopoJSONConnector — materializes TopoJSON files into canonical artifacts.

Lane: connector

TopoJSON is a topology-preserving extension of GeoJSON. Arcs are shared between
geometries, reducing file size. This connector parses TopoJSON and converts to
standard GeoJSON geometries.

Key features:
- Local file materialization (wrap in place or lazy handle)
- Arc decoding: delta-encoded coordinates with optional transform
- Object selection: default (first) object or explicit via ::object_name
- Supports .topojson and .json extensions (detected by content inspection)
- Uses stdlib json + shapely only — no new dependencies

Design decisions:
- source_ref: "path/to/file.topojson" or "path/to/file.topojson::object_name"
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- CRS is always assumed EPSG:4326 (same as GeoJSON convention)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import shapely.geometry
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    Lineage,
    SpatialDescriptor,
    content_hash,
)
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# Extensions for TopoJSON files
TOPOJSON_EXTENSIONS = {".topojson", ".json"}


def _is_topojson_file(path: Path) -> bool:
    """Check if file has a TopoJSON extension (case-insensitive)."""
    return path.suffix.lower() in TOPOJSON_EXTENSIONS


def _parse_source_ref(source_ref: SourceRef | str) -> tuple[str, str | None]:
    """Parse source_ref into (file_path, object_name).

    Returns:
        Tuple of (file_path, optional_object_name)
        Object name is extracted from "path::object_name" syntax.
    """
    if isinstance(source_ref, str):
        raw = source_ref.strip()
    else:
        raw = source_ref.raw.strip()

    if "::" in raw:
        file_path, object_name = raw.split("::", 1)
        return (file_path, object_name)

    return (raw, None)


def _decode_arc(
    arc_coords: list[list[float]], transform: dict[str, Any] | None = None
) -> list[list[float]]:
    """Decode delta-encoded arc coordinates.

    Args:
        arc_coords: List of [dx, dy] delta coordinates
        transform: Optional transform dict with "scale" and "translate"

    Returns:
        List of decoded [x, y] coordinates
    """
    coords: list[list[float]] = []
    x, y = 0.0, 0.0
    for dx, dy in arc_coords:
        x += dx
        y += dy
        if transform:
            scale = transform["scale"]
            translate = transform["translate"]
            coords.append(
                [
                    x * scale[0] + translate[0],
                    y * scale[1] + translate[1],
                ]
            )
        else:
            coords.append([x, y])
    return coords


def _decode_arc_index(
    arc_index: int, arcs: list[list[list[float]]], transform: dict[str, Any] | None = None
) -> list[list[float]]:
    """Decode a single arc by index.

    Positive index: use arc directly.
    Negative index (~i, bitwise NOT): use arc reversed.
    """
    if arc_index >= 0:
        return _decode_arc(arcs[arc_index], transform)
    else:
        # Negative index means reversed arc
        idx = ~arc_index  # bitwise NOT
        decoded = _decode_arc(arcs[idx], transform)
        return list(reversed(decoded))


def _decode_arcs(
    arc_indices: list[int], arcs: list[list[list[float]]], transform: dict[str, Any] | None = None
) -> list[list[float]]:
    """Decode and concatenate multiple arcs into a coordinate sequence.

    Args:
        arc_indices: List of arc indices (positive or negative)
        arcs: The arcs array from the TopoJSON
        transform: Optional transform dict

    Returns:
        Concatenated list of coordinates
    """
    coords: list[list[float]] = []
    for idx in arc_indices:
        arc_coords = _decode_arc_index(idx, arcs, transform)
        # Avoid duplicating the last point of previous arc with first of next
        if coords and arc_coords and coords[-1] == arc_coords[0]:
            coords.extend(arc_coords[1:])
        else:
            coords.extend(arc_coords)
    return coords


def _geometry_to_geojson(
    geom: dict[str, Any], arcs: list[list[list[float]]], transform: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Convert a TopoJSON geometry to GeoJSON format.

    Args:
        geom: TopoJSON geometry dict
        arcs: The arcs array from the TopoJSON
        transform: Optional transform dict

    Returns:
        GeoJSON geometry dict or None if empty
    """
    geom_type = geom.get("type")

    if geom_type == "Point":
        coords = geom.get("coordinates")
        if coords:
            return {"type": "Point", "coordinates": coords}
        return None

    if geom_type == "MultiPoint":
        coords = geom.get("coordinates")
        if coords:
            return {"type": "MultiPoint", "coordinates": coords}
        return None

    if geom_type == "LineString":
        arc_indices = geom.get("arcs", [])
        if not arc_indices:
            return None
        coords = _decode_arcs(arc_indices, arcs, transform)
        if len(coords) < 2:
            return None
        return {"type": "LineString", "coordinates": coords}

    if geom_type == "MultiLineString":
        arc_sets = geom.get("arcs", [])
        if not arc_sets:
            return None
        lines: list[list[list[float]]] = []
        for arc_indices in arc_sets:
            coords = _decode_arcs(arc_indices, arcs, transform)
            if len(coords) >= 2:
                lines.append(coords)
        if not lines:
            return None
        return {"type": "MultiLineString", "coordinates": lines}

    if geom_type == "Polygon":
        rings = geom.get("arcs", [])
        if not rings:
            return None
        geojson_rings: list[list[list[float]]] = []
        for ring_indices in rings:
            coords = _decode_arcs(ring_indices, arcs, transform)
            # Close the ring if needed
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) >= 4:  # Need at least 4 points for a valid ring
                geojson_rings.append(coords)
        if not geojson_rings:
            return None
        return {"type": "Polygon", "coordinates": geojson_rings}

    if geom_type == "MultiPolygon":
        polygon_sets = geom.get("arcs", [])
        if not polygon_sets:
            return None
        polygons: list[list[list[list[float]]]] = []
        for rings in polygon_sets:
            geojson_rings: list[list[list[float]]] = []
            for ring_indices in rings:
                coords = _decode_arcs(ring_indices, arcs, transform)
                # Close the ring if needed
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                if len(coords) >= 4:
                    geojson_rings.append(coords)
            if geojson_rings:
                polygons.append(geojson_rings)
        if not polygons:
            return None
        return {"type": "MultiPolygon", "coordinates": polygons}

    if geom_type == "GeometryCollection":
        geometries = geom.get("geometries", [])
        if not geometries:
            return None
        geojson_geoms: list[dict[str, Any]] = []
        for g in geometries:
            geojson_g = _geometry_to_geojson(g, arcs, transform)
            if geojson_g:
                geojson_geoms.append(geojson_g)
        if not geojson_geoms:
            return None
        return {"type": "GeometryCollection", "geometries": geojson_geoms}

    return None


def _read_topojson_metadata(path: Path) -> dict[str, Any]:
    """Read metadata from a TopoJSON file.

    Args:
        path: Path to the TopoJSON file

    Returns:
        Dict with objects, arcs, bbox, transform, feature_counts, geometry_types
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate TopoJSON structure
    if data.get("type") != "Topology":
        raise ValueError("Not a valid TopoJSON file: 'type' is not 'Topology'")

    if "objects" not in data:
        raise ValueError("Not a valid TopoJSON file: missing 'objects' key")

    objects = data["objects"]
    arcs = data.get("arcs", [])
    bbox = data.get("bbox")
    transform = data.get("transform")

    # Analyze each object
    object_info: dict[str, dict[str, Any]] = {}
    for obj_name, obj_data in objects.items():
        obj_type = obj_data.get("type")
        geometries: list[dict[str, Any]] = []

        if obj_type == "GeometryCollection":
            geometries = obj_data.get("geometries", [])
        elif obj_type in (
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
        ):
            # Single geometry wrapped as a list
            geometries = [obj_data]

        # Count features and collect geometry types
        feature_count = len(geometries)
        geom_types: set[str] = set()
        for g in geometries:
            gtype = g.get("type")
            if gtype:
                geom_types.add(gtype)

        object_info[obj_name] = {
            "type": obj_type,
            "feature_count": feature_count,
            "geometry_types": sorted(geom_types),
        }

    return {
        "objects": object_info,
        "object_names": list(objects.keys()),
        "arc_count": len(arcs),
        "bbox": bbox,
        "transform": transform,
        "has_transform": transform is not None,
    }


def _compute_extent_from_arcs(
    arcs: list[list[list[float]]], transform: dict[str, Any] | None = None
) -> tuple[float, float, float, float] | None:
    """Compute extent from decoded arcs.

    Args:
        arcs: The arcs array from the TopoJSON
        transform: Optional transform dict

    Returns:
        (xmin, ymin, xmax, ymax) or None if no arcs
    """
    if not arcs:
        return None

    all_coords: list[list[float]] = []
    for arc in arcs:
        decoded = _decode_arc(arc, transform)
        all_coords.extend(decoded)

    if not all_coords:
        return None

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _get_object_extent(
    obj_data: dict[str, Any],
    arcs: list[list[list[float]]],
    transform: dict[str, Any] | None = None,
) -> tuple[float, float, float, float] | None:
    """Compute extent for a specific object by decoding its geometries.

    Args:
        obj_data: The object data from TopoJSON
        arcs: The arcs array
        transform: Optional transform dict

    Returns:
        (xmin, ymin, xmax, ymax) or None
    """
    obj_type = obj_data.get("type")
    geometries: list[dict[str, Any]] = []

    if obj_type == "GeometryCollection":
        geometries = obj_data.get("geometries", [])
    elif obj_type in (
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    ):
        geometries = [obj_data]

    if not geometries:
        return None

    # Decode all geometries and compute bounds
    all_coords: list[list[float]] = []
    for geom in geometries:
        geojson_geom = _geometry_to_geojson(geom, arcs, transform)
        if geojson_geom:
            try:
                shapely_geom = shapely.geometry.shape(geojson_geom)
                if not shapely_geom.is_empty:
                    minx, miny, maxx, maxy = shapely_geom.bounds
                    all_coords.extend([[minx, miny], [maxx, maxy]])
            except Exception:
                # Skip geometries that can't be converted
                pass

    if not all_coords:
        return None

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return (min(xs), min(ys), max(xs), max(ys))


class TopoJSONConnector:
    """Materializes TopoJSON files into canonical Quarry artifacts.

    Supports local files with .topojson or .json extensions.
    Uses stdlib json + shapely for parsing and metadata extraction.
    """

    @property
    def name(self) -> str:
        return "topojson"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.MATERIALIZE_LAZY
            | ConnectorCapability.METADATA_ONLY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize a TopoJSON file into a canonical artifact.

        source_ref: local path to .topojson/.json file, optionally with ::object_name
        workspace: where to put materialized data if needed (unused for local)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        file_path, object_name = _parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        # Read and parse TopoJSON
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise MaterializeError(source_ref, f"Invalid JSON: {e}") from e
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read file: {e}") from e

        # Validate TopoJSON structure
        if data.get("type") != "Topology":
            raise MaterializeError(
                source_ref,
                f"Not a valid TopoJSON file: 'type' is '{data.get('type')}', expected 'Topology'",
            )

        if "objects" not in data:
            raise MaterializeError(source_ref, "Not a valid TopoJSON file: missing 'objects' key")

        objects = data["objects"]
        arcs = data.get("arcs", [])
        bbox = data.get("bbox")
        transform = data.get("transform")

        # Select object
        if object_name:
            if object_name not in objects:
                available = ", ".join(objects.keys())
                raise MaterializeError(
                    source_ref, f"Object '{object_name}' not found. Available: {available}"
                )
            selected_obj = objects[object_name]
            selected_name = object_name
        else:
            # Default to first object
            if not objects:
                raise MaterializeError(source_ref, "No objects found in TopoJSON file")
            selected_name = list(objects.keys())[0]
            selected_obj = objects[selected_name]

        # Get object type and geometries
        obj_type = selected_obj.get("type")
        geometries: list[dict[str, Any]] = []

        if obj_type == "GeometryCollection":
            geometries = selected_obj.get("geometries", [])
        elif obj_type in (
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
        ):
            geometries = [selected_obj]

        feature_count = len(geometries)

        # Collect geometry types
        geom_types: set[str] = set()
        for g in geometries:
            gtype = g.get("type")
            if gtype:
                geom_types.add(gtype)

        # Determine extent
        extent = None
        if bbox:
            extent = (bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            extent = _get_object_extent(selected_obj, arcs, transform)

        # Build spatial descriptor
        spatial = SpatialDescriptor(
            crs="EPSG:4326",  # TopoJSON assumes WGS84 like GeoJSON
            extent=extent,
            feature_count=feature_count,
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "topojson",
            "path": str(path),
            "object_name": selected_name,
            "lazy": lazy,
            "has_transform": transform is not None,
        }

        # Build artifact metadata
        artifact_meta: dict[str, Any] = {
            "object_name": selected_name,
            "object_type": obj_type,
            "geometry_types": sorted(geom_types),
            "feature_count": feature_count,
            "arc_count": len(arcs),
            "bbox": bbox,
            "has_transform": transform is not None,
            "assumed_crs": True,  # We assume EPSG:4326
            "all_object_names": list(objects.keys()),
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=selected_name,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"TopoJSON metadata only — {feature_count} features in '{selected_name}'",
            )

        # Eager mode: wrap in place, LOCAL_FILE backing
        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=selected_name,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=f"Local TopoJSON wrapped ({path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Discover TopoJSON files or objects.

        Two modes:
        - If query points to a .topojson FILE → list named objects within it
        - If query points to a DIRECTORY → list .topojson files

        query: file path or directory path as string, or dict with "path" key
        """
        if isinstance(query, dict):
            path_str = query.get("path")
        elif isinstance(query, str):
            path_str = query
        else:
            raise MaterializeError("discover", "No path specified")

        if not path_str:
            raise MaterializeError("discover", "No path specified")

        path = Path(path_str)

        if not path.exists():
            raise MaterializeError("discover", f"Path not found: {path_str}")

        # Mode 1: File - list objects within
        if path.is_file():
            if not _is_topojson_file(path):
                raise MaterializeError(
                    "discover", f"Not a TopoJSON file (expected .topojson or .json): {path_str}"
                )

            try:
                meta = _read_topojson_metadata(path)
            except Exception as e:
                raise MaterializeError("discover", f"Failed to read TopoJSON: {e}") from e

            entries = []
            for obj_name in meta["object_names"]:
                obj_info = meta["objects"][obj_name]
                entries.append(
                    CatalogEntry(
                        source_ref=f"{path}::{obj_name}",
                        name=obj_name,
                        spatial_hint={
                            "crs": "EPSG:4326",
                            "feature_count": obj_info["feature_count"],
                        },
                        metadata={
                            "object_type": obj_info["type"],
                            "geometry_types": obj_info["geometry_types"],
                        },
                    )
                )
            return entries

        # Mode 2: Directory - list .topojson files
        if path.is_dir():
            seen: set[str] = set()
            entries = []
            for ext in TOPOJSON_EXTENSIONS:
                for pattern in (f"*{ext}", f"*{ext.upper()}"):
                    for file_path in path.glob(pattern):
                        resolved = str(file_path.resolve())
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        entries.append(
                            CatalogEntry(
                                source_ref=str(file_path),
                                name=file_path.stem,
                                metadata={
                                    "extension": file_path.suffix,
                                },
                            )
                        )
            return entries

        # Should not reach here
        raise MaterializeError("discover", f"Invalid path: {path_str}")

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get TopoJSON metadata without materializing data."""
        file_path, object_name = _parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        # Read and parse TopoJSON
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise MaterializeError(source_ref, f"Invalid JSON: {e}") from e
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read file: {e}") from e

        # Validate TopoJSON structure
        if data.get("type") != "Topology":
            raise MaterializeError(
                source_ref,
                f"Not a valid TopoJSON file: 'type' is '{data.get('type')}', expected 'Topology'",
            )

        if "objects" not in data:
            raise MaterializeError(source_ref, "Not a valid TopoJSON file: missing 'objects' key")

        objects = data["objects"]
        arcs = data.get("arcs", [])
        bbox = data.get("bbox")
        transform = data.get("transform")

        # Build object info
        object_info = {}
        for obj_name, obj_data in objects.items():
            obj_type = obj_data.get("type")
            geometries: list[dict[str, Any]] = []

            if obj_type == "GeometryCollection":
                geometries = obj_data.get("geometries", [])
            elif obj_type in (
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ):
                geometries = [obj_data]

            geom_types: set[str] = set()
            for g in geometries:
                gtype = g.get("type")
                if gtype:
                    geom_types.add(gtype)

            object_info[obj_name] = {
                "type": obj_type,
                "feature_count": len(geometries),
                "geometry_types": sorted(geom_types),
            }

        # Determine extent
        extent = None
        if bbox:
            extent = (bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            extent = _compute_extent_from_arcs(arcs, transform)

        return {
            "object_names": list(objects.keys()),
            "objects": object_info,
            "arc_count": len(arcs),
            "bbox": bbox,
            "extent": extent,
            "has_transform": transform is not None,
            "crs": "EPSG:4326",
            "assumed_crs": True,
        }
