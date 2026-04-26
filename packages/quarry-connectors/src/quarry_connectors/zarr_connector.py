"""ZarrConnector — materializes Zarr stores into canonical raster artifacts.

Lane: connector

Zarr is a chunked, compressed N-dimensional array storage format used for large
raster/climate/model data. This connector reads Zarr stores (directories or .zarr
files) and materializes them as raster artifacts.

Design decisions:
- source_ref format: "path/to/store.zarr" (auto-select variable) or
  "path/to/store.zarr::variable_name" (specific array)
- Variable auto-selection: first array in group, or root if it's an Array
- Lazy: metadata-only, LAZY_HANDLE backing
- Eager: wrap in place, LOCAL_FILE backing (Zarr stores are self-contained)
- Discover: list arrays within a store, or list .zarr stores in a directory
- CRS extraction: checks array.attrs for "crs", "crs_wkt", CF conventions
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    Lineage,
    SpatialDescriptor,
)
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# Optional zarr dependency
try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    zarr = None


def _is_zarr_store(path: Path) -> bool:
    """Check if path is a Zarr store.

    - .zarr directory suffix
    - .zip file (could be zipped zarr)
    - Directory containing .zarray or .zgroup marker files
    """
    if path.suffix == ".zarr" and path.is_dir():
        return True
    if path.suffix == ".zip" and path.is_file():
        return True
    if path.is_dir():
        if (path / ".zarray").exists() or (path / ".zgroup").exists():
            return True
    return False


def _extract_crs(array) -> str | None:
    """Extract CRS from Zarr array attributes.

    Checks multiple conventions:
    - Direct "crs" or "crs_wkt" attribute
    - CF convention: "grid_mapping" attr pointing to another variable
    - xarray convention: "spatial_ref" coordinate
    """
    attrs = array.attrs

    # Direct CRS attributes
    if "crs_wkt" in attrs:
        return str(attrs["crs_wkt"])
    if "crs" in attrs:
        return str(attrs["crs"])
    if "spatial_ref" in attrs:
        return str(attrs["spatial_ref"])

    # CF convention: grid_mapping points to another variable
    grid_mapping = attrs.get("grid_mapping")
    if grid_mapping and isinstance(grid_mapping, str):
        # Try to look up the grid_mapping variable in parent group
        try:
            parent = array.store
            if hasattr(parent, "__getitem__"):
                # Try to get grid mapping attrs from store
                gm_path = f"{grid_mapping}/.zattrs"
                if gm_path in parent:
                    import json

                    gm_attrs = json.loads(parent[gm_path])
                    if "crs_wkt" in gm_attrs:
                        return str(gm_attrs["crs_wkt"])
                    if "spatial_ref" in gm_attrs:
                        return str(gm_attrs["spatial_ref"])
        except Exception:
            pass

    return None


def _extract_extent(array, attrs: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Extract spatial extent from Zarr array attributes.

    Checks CF convention geospatial bounds attributes.
    """
    # CF convention geospatial bounds
    lat_min = attrs.get("geospatial_lat_min")
    lat_max = attrs.get("geospatial_lat_max")
    lon_min = attrs.get("geospatial_lon_min")
    lon_max = attrs.get("geospatial_lon_max")

    if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
        try:
            return (float(lon_min), float(lat_min), float(lon_max), float(lat_max))
        except (ValueError, TypeError):
            pass

    # Try x/y bounds
    x_min = attrs.get("geospatial_x_min")
    x_max = attrs.get("geospatial_x_max")
    y_min = attrs.get("geospatial_y_min")
    y_max = attrs.get("geospatial_y_max")

    if all(v is not None for v in [x_min, x_max, y_min, y_max]):
        try:
            return (float(x_min), float(y_min), float(x_max), float(y_max))
        except (ValueError, TypeError):
            pass

    return None


def _get_band_count(array) -> int:
    """Infer band count from array shape and dimensions.

    For 2D arrays: band_count=1
    For 3D arrays: infer from shape (typically bands is first or last dim)
    """
    shape = array.shape
    if len(shape) == 2:
        return 1
    if len(shape) == 3:
        # Common conventions: (bands, y, x) or (time, y, x) or (y, x, bands)
        # Assume smallest dimension is bands, or first if equal
        return min(shape)
    if len(shape) > 3:
        # Multi-dimensional: use last non-spatial dimension
        return shape[0] if len(shape) > 2 else 1
    return 1


def _get_array_metadata(array) -> dict[str, Any]:
    """Extract metadata from a Zarr array."""
    attrs = dict(array.attrs)

    # Handle compressor for both zarr v2 and v3
    compressor = None
    if hasattr(array, "compressors"):
        # zarr v3
        try:
            compressors = array.compressors
            compressor = str(compressors) if compressors else None
        except Exception:
            pass
    elif hasattr(array, "compressor"):
        # zarr v2 - but accessing may raise for v3 format
        try:
            comp = array.compressor
            compressor = str(comp) if comp else None
        except (TypeError, AttributeError):
            pass

    meta = {
        "shape": list(array.shape),
        "chunks": list(array.chunks) if hasattr(array, "chunks") else None,
        "dtype": str(array.dtype),
        "compressor": compressor,
        "fill_value": array.fill_value if hasattr(array, "fill_value") else None,
        "attrs": attrs,
    }

    # Extract CRS
    crs = _extract_crs(array)
    if crs:
        meta["crs"] = crs

    # Extract extent
    extent = _extract_extent(array, attrs)
    if extent:
        meta["extent"] = extent

    return meta


class ZarrConnector:
    """Materializes Zarr stores into canonical Quarry raster artifacts.

    Supports variable selection via :: separator in source_ref.
    Uses zarr library for reading store structure and array metadata.
    """

    @property
    def name(self) -> str:
        return "zarr"

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
        """Materialize a Zarr array into a canonical artifact.

        source_ref formats:
            "path/to/store.zarr"                    — auto-select first array
            "path/to/store.zarr::variable_name"     — specific array
        """
        if not HAS_ZARR:
            raise MaterializeError(
                source_ref,
                "zarr package not installed. Install with: pip install zarr",
            )

        path, variable = self._parse_source_ref(source_ref)
        path_obj = Path(path)

        if not path_obj.exists():
            raise MaterializeError(source_ref, f"Path not found: {path}")

        if not _is_zarr_store(path_obj):
            raise MaterializeError(source_ref, f"Not a Zarr store: {path}")

        # Open the store
        try:
            store = zarr.open(str(path_obj), mode="r")
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open Zarr store: {e}") from e

        # Get the array (root or from group)
        array, actual_var = self._get_array(store, variable, source_ref)

        # Extract metadata
        meta = _get_array_metadata(array)
        shape = meta["shape"]
        chunks = meta["chunks"]
        dtype = meta["dtype"]
        compressor = meta["compressor"]
        crs = meta.get("crs")
        extent = meta.get("extent")
        band_count = _get_band_count(array)

        # Build spatial descriptor
        spatial = SpatialDescriptor(
            crs=crs,
            extent=extent,
            resolution=None,  # Zarr doesn't store resolution directly
            band_count=band_count,
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "zarr",
            "path": str(path_obj),
            "variable": actual_var,
            "lazy": lazy,
            "shape": shape,
            "chunks": chunks,
            "dtype": dtype,
        }

        # Build artifact metadata
        artifact_meta = {
            "shape": shape,
            "chunks": chunks,
            "dtype": dtype,
            "compressor": compressor,
            "variable": actual_var,
            "attrs": meta["attrs"],
        }
        if crs:
            artifact_meta["crs"] = crs

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=actual_var or "array",
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"{path_obj}::{actual_var}" if actual_var else str(path_obj),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"Zarr array '{actual_var}' — metadata only",
            )

        # Eager mode: wrap in place with LOCAL_FILE backing
        # For Zarr, we keep the store as-is (it's self-contained)
        artifact = Artifact(
            type=ArtifactType.RASTER,
            name=actual_var or "array",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path_obj),
                size_bytes=self._estimate_store_size(path_obj),
                content_hash=None,  # Zarr stores are directories, hash would be expensive
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=f"Zarr store wrapped in place ({path_obj.name})",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Discover Zarr arrays or stores.

        Two modes:
        - If query points to a .zarr store: list arrays within it
        - If query points to a directory: list .zarr stores in it
        """
        if not HAS_ZARR:
            raise MaterializeError(
                query,
                "zarr package not installed. Install with: pip install zarr",
            )

        if isinstance(query, dict):
            path_str = query.get("path")
        elif isinstance(query, str):
            path_str = query
        else:
            raise MaterializeError("discover", "Query must specify 'path'")

        if not path_str:
            raise MaterializeError("discover", "No path specified")

        path = Path(path_str)

        if not path.exists():
            raise MaterializeError("discover", f"Path not found: {path}")

        # If it's a Zarr store, list arrays within
        if _is_zarr_store(path):
            return self._discover_arrays(path)

        # If it's a directory, list Zarr stores in it
        if path.is_dir():
            return self._discover_stores(path)

        raise MaterializeError("discover", f"Not a Zarr store or directory: {path}")

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata about a Zarr array without materializing."""
        if not HAS_ZARR:
            raise MaterializeError(
                source_ref,
                "zarr package not installed. Install with: pip install zarr",
            )

        path, variable = self._parse_source_ref(source_ref)
        path_obj = Path(path)

        if not path_obj.exists():
            raise MaterializeError(source_ref, f"Path not found: {path}")

        if not _is_zarr_store(path_obj):
            raise MaterializeError(source_ref, f"Not a Zarr store: {path}")

        try:
            store = zarr.open(str(path_obj), mode="r")
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open Zarr store: {e}") from e

        array, actual_var = self._get_array(store, variable, source_ref)
        meta = _get_array_metadata(array)
        band_count = _get_band_count(array)

        return {
            "path": str(path_obj),
            "variable": actual_var,
            "shape": meta["shape"],
            "chunks": meta["chunks"],
            "dtype": meta["dtype"],
            "compressor": meta["compressor"],
            "crs": meta.get("crs"),
            "extent": meta.get("extent"),
            "band_count": band_count,
            "attrs": meta["attrs"],
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None]:
        """Parse source_ref into (path, variable).

        Returns:
            (store_path, variable_name_or_None)
        """
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            params = source_ref.params or {}
            path = params.get("path", source_ref.raw)
            variable = params.get("variable")
            return path, variable

        raw = source_ref.strip()

        # Parse :: separator
        if "::" in raw:
            path_part, var_part = raw.split("::", 1)
            return path_part.strip(), var_part.strip()

        return raw, None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_array(
        self, store, variable: str | None, source_ref: SourceRef | str
    ) -> tuple[Any, str | None]:
        """Get array from store, handling Group vs Array and variable selection."""
        # If store is already an Array (root is array, not group)
        if hasattr(store, "shape") and hasattr(store, "dtype"):
            # It's an array
            if variable is not None:
                # User asked for a specific variable but root is array
                raise MaterializeError(
                    source_ref,
                    f"Store root is an array, cannot select variable '{variable}'",
                )
            return store, None

        # It's a Group
        if variable is not None:
            # Select specific variable
            if variable not in store:
                available = list(store.keys())
                raise MaterializeError(
                    source_ref,
                    f"Variable '{variable}' not found. Available: {available}",
                )
            return store[variable], variable

        # Auto-select: first array in group
        arrays = [k for k in store.keys() if hasattr(store[k], "shape")]
        if not arrays:
            raise MaterializeError(source_ref, "No arrays found in Zarr store")

        first_var = arrays[0]
        return store[first_var], first_var

    def _discover_arrays(self, path: Path) -> list[CatalogEntry]:
        """List arrays within a Zarr store."""
        try:
            store = zarr.open(str(path), mode="r")
        except Exception as e:
            raise MaterializeError(path, f"Failed to open Zarr store: {e}") from e

        entries = []

        # If root is an array
        if hasattr(store, "shape") and hasattr(store, "dtype"):
            meta = _get_array_metadata(store)
            entries.append(
                CatalogEntry(
                    source_ref=f"{path}",
                    name="array",
                    metadata={
                        "shape": meta["shape"],
                        "chunks": meta["chunks"],
                        "dtype": meta["dtype"],
                        "crs": meta.get("crs"),
                    },
                )
            )
            return entries

        # Root is a group: list all arrays
        for name in store.keys():
            arr = store[name]
            if not hasattr(arr, "shape"):
                continue  # Skip sub-groups for now

            meta = _get_array_metadata(arr)
            entries.append(
                CatalogEntry(
                    source_ref=f"{path}::{name}",
                    name=name,
                    metadata={
                        "shape": meta["shape"],
                        "chunks": meta["chunks"],
                        "dtype": meta["dtype"],
                        "crs": meta.get("crs"),
                    },
                )
            )

        return entries

    def _discover_stores(self, path: Path) -> list[CatalogEntry]:
        """List Zarr stores in a directory."""
        entries = []
        seen: set[str] = set()

        # Look for .zarr directories
        for pattern in ("*.zarr", "*.ZARR"):
            for store_path in path.glob(pattern):
                resolved = str(store_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                entries.append(
                    CatalogEntry(
                        source_ref=str(store_path),
                        name=store_path.stem,
                        metadata={
                            "type": "zarr_store",
                            "path": str(store_path),
                        },
                    )
                )

        # Look for directories with .zarray or .zgroup markers
        for subpath in path.iterdir():
            if subpath.is_dir() and _is_zarr_store(subpath):
                resolved = str(subpath.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                entries.append(
                    CatalogEntry(
                        source_ref=str(subpath),
                        name=subpath.name,
                        metadata={
                            "type": "zarr_store",
                            "path": str(subpath),
                        },
                    )
                )

        return entries

    def _estimate_store_size(self, path: Path) -> int | None:
        """Estimate total size of Zarr store (directory)."""
        try:
            total = 0
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
            return total
        except Exception:
            return None
