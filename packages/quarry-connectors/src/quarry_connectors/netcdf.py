"""NetCDFConnector — materializes NetCDF/HDF5 scientific raster data into canonical artifacts.

Lane: connector

Handles NetCDF (.nc) and HDF5 (.h5, .hdf5) scientific data files via GDAL's
NetCDF and HDF5 drivers through rasterio. Supports variable selection via
:: separator or SourceRef params.

Design decisions:
- source_ref format: "path.nc" (auto-select variable) or "path.nc::variable_name"
- Variable auto-selection: prefer first variable with spatial dimensions (lat/lon or y/x)
- Lazy: metadata-only via rasterio, LAZY_HANDLE backing with NETCDF/HDF5 URI
- Eager: copy to GeoTIFF in workspace via rasterio.shutil.copy()
- Discover: list variables using netCDF4 (if available) or rasterio subdatasets
- No xarray dependency — rasterio provides sufficient access
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import rasterio
import rasterio.shutil
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

# Optional netCDF4 for rich variable discovery
try:
    import netCDF4  # noqa: N813

    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


def _detect_format(path: str) -> str:
    """Detect file format from extension."""
    p = path.lower()
    if p.endswith(".nc"):
        return "netcdf"
    if p.endswith(".h5") or p.endswith(".hdf5") or p.endswith(".hdf"):
        return "hdf5"
    return "unknown"


def _build_subdataset_uri(path: str, variable: str, fmt: str) -> str:
    """Build GDAL subdataset URI for opening a specific variable."""
    if fmt == "netcdf":
        return f'NETCDF:"{path}":{variable}'
    if fmt == "hdf5":
        return f'HDF5:"{path}"://{variable}'
    return path


def _parse_subdataset(subdataset: str) -> tuple[str, str]:
    """Parse a GDAL subdataset string into (path, variable).

    Handles formats like:
    - NETCDF:"path.nc":variable
    - HDF5:"file.h5"://dataset_name
    """
    # Find the quoted path
    if '"' not in subdataset:
        # Fallback: try to parse without quotes
        parts = subdataset.split(":")
        if len(parts) >= 3:
            return parts[1], parts[2]
        return subdataset, ""

    # Extract quoted path
    first_quote = subdataset.find('"')
    second_quote = subdataset.find('"', first_quote + 1)
    if first_quote == -1 or second_quote == -1:
        return subdataset, ""

    path = subdataset[first_quote + 1 : second_quote]
    rest = subdataset[second_quote + 1 :]

    # Variable is after the second quote, usually prefixed with : or ://
    rest = rest.lstrip(":")
    rest = rest.lstrip("/")
    variable = rest

    return path, variable


class NetCDFConnector:
    """Materializes NetCDF/HDF5 scientific raster data into canonical Quarry artifacts.

    Supports variable selection via :: separator in source_ref or SourceRef params.
    Uses rasterio (GDAL) for reading; optional netCDF4 for rich variable discovery.
    """

    def __init__(self, default_variable: str | None = None):
        """Initialize NetCDF connector.

        Args:
            default_variable: Optional default variable name if not specified in source_ref.
        """
        self._default_variable = default_variable

    @property
    def name(self) -> str:
        return "netcdf"

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
        """Materialize a NetCDF/HDF5 variable into a canonical artifact.

        source_ref formats:
            "path/to/file.nc"                    — auto-select variable
            "path/to/file.nc::variable_name"     — specific variable
            "path/to/file.h5::dataset_name"      — HDF5 dataset
        """
        path, variable, fmt = self._parse_source_ref(source_ref)

        if not Path(path).exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        # Auto-select variable if not specified
        if variable is None:
            variable = self._auto_select_variable(path, fmt)
            if variable is None:
                raise MaterializeError(source_ref, "No variables found in file")

        # Build subdataset URI for opening
        subdataset_uri = _build_subdataset_uri(path, variable, fmt)

        # Inspect the selected variable
        try:
            spatial, band_count, dtype = self._inspect(subdataset_uri)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open variable '{variable}': {e}") from e

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "netcdf",
            "path": path,
            "variable": variable,
            "format": fmt,
            "lazy": lazy,
        }

        # Build artifact metadata
        artifact_meta = {
            "format": fmt,
            "variable": variable,
            "dtype": dtype,
            "band_count": band_count,
        }

        if lazy:
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=variable,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=subdataset_uri,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"NetCDF variable '{variable}' — metadata only",
            )

        # Eager: copy to GeoTIFF in workspace
        output_path = self._copy_to_geotiff(subdataset_uri, workspace, variable)
        lineage_params["output_format"] = "geotiff"

        artifact = Artifact(
            type=ArtifactType.RASTER,
            name=variable,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )
        size = output_path.stat().st_size
        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=f"NetCDF variable '{variable}' copied to GeoTIFF ({size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List variables in a NetCDF/HDF5 file.

        query as dict supports:
            path: str (required) — path to NetCDF/HDF5 file
        """
        if isinstance(query, dict):
            path = query.get("path")
        elif isinstance(query, str):
            path = query
        else:
            raise MaterializeError("discover", "Query must specify 'path'")

        if not path:
            raise MaterializeError("discover", "No path specified")

        if not Path(path).exists():
            raise MaterializeError("discover", f"File not found: {path}")

        fmt = _detect_format(path)
        variables = self._list_variables(path, fmt)

        entries = []
        for var_info in variables:
            name = var_info["name"]
            # Build source_ref with :: separator
            source_ref = f"{path}::{name}"
            entries.append(
                CatalogEntry(
                    source_ref=source_ref,
                    name=name,
                    metadata={
                        "dimensions": var_info.get("dimensions"),
                        "shape": var_info.get("shape"),
                        "attributes": var_info.get("attributes", {}),
                        "format": fmt,
                    },
                )
            )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata about a NetCDF/HDF5 variable without materializing."""
        path, variable, fmt = self._parse_source_ref(source_ref)

        if not Path(path).exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        # Auto-select variable if not specified
        if variable is None:
            variable = self._auto_select_variable(path, fmt)
            if variable is None:
                raise MaterializeError(source_ref, "No variables found in file")

        subdataset_uri = _build_subdataset_uri(path, variable, fmt)

        try:
            spatial, band_count, dtype = self._inspect(subdataset_uri)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open variable '{variable}': {e}") from e

        # Get variable-level metadata
        var_meta = self._get_variable_metadata(path, variable, fmt)

        return {
            "path": path,
            "variable": variable,
            "format": fmt,
            "crs": spatial.crs,
            "extent": spatial.extent,
            "resolution": spatial.resolution,
            "band_count": band_count,
            "dtype": dtype,
            "dimensions": var_meta.get("dimensions"),
            "shape": var_meta.get("shape"),
            "attributes": var_meta.get("attributes", {}),
            "global_attributes": var_meta.get("global_attributes", {}),
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None, str]:
        """Parse source_ref into (path, variable, format).

        Returns:
            (file_path, variable_name_or_None, format)
        """
        from quarry_core.source_ref import SourceRef, SourceRefKind

        if isinstance(source_ref, SourceRef):
            if source_ref.kind in (
                SourceRefKind.LOCAL_PATH,
                SourceRefKind.LOCAL_RASTER,
            ):
                params = source_ref.params or {}
                path = params.get("path", source_ref.raw)
                variable = params.get("variable", self._default_variable)
                return path, variable, _detect_format(path)
            raw = source_ref.raw.strip()
        else:
            raw = source_ref.strip()

        # Parse :: separator
        if "::" in raw:
            path_part, var_part = raw.split("::", 1)
            path = path_part.strip()
            variable = var_part.strip()
            return path, variable, _detect_format(path)

        # No variable specified
        path = raw
        return path, None, _detect_format(path)

    # -----------------------------------------------------------------------
    # Variable selection and discovery
    # -----------------------------------------------------------------------

    def _list_variables(self, path: str, fmt: str) -> list[dict[str, Any]]:
        """List all variables in the file with their metadata."""
        variables = []

        if HAS_NETCDF4 and fmt == "netcdf":
            try:
                with netCDF4.Dataset(path, "r") as ds:  # noqa: N813
                    for var_name, var in ds.variables.items():
                        var_info = {
                            "name": var_name,
                            "dimensions": list(var.dimensions),
                            "shape": list(var.shape),
                            "attributes": {attr: getattr(var, attr) for attr in var.ncattrs()},
                        }
                        variables.append(var_info)
                    # Add global attributes to first variable for discover
                    if variables:
                        global_attrs = {attr: getattr(ds, attr) for attr in ds.ncattrs()}
                        variables[0]["global_attributes"] = global_attrs
            except OSError:
                # netCDF4 can't read the file — fall back to rasterio subdatasets
                pass

        if not variables:
            # Use rasterio subdatasets
            try:
                with rasterio.open(path) as src:
                    subdatasets = src.subdatasets
                    for subdataset in subdatasets:
                        _, var_name = _parse_subdataset(subdataset)
                        var_info = {"name": var_name}
                        # Try to get more info by opening the subdataset
                        try:
                            with rasterio.open(subdataset) as var_src:
                                var_info["shape"] = [var_src.height, var_src.width]
                                var_info["band_count"] = var_src.count
                                var_info["dtype"] = (
                                    str(var_src.dtypes[0]) if var_src.dtypes else None
                                )
                        except rasterio.errors.RasterioIOError:
                            pass  # subdataset may not be openable as raster
                        variables.append(var_info)
            except Exception as e:
                raise MaterializeError(path, f"Failed to list variables: {e}") from e

        return variables

    def _auto_select_variable(self, path: str, fmt: str) -> str | None:
        """Auto-select a variable with spatial dimensions.

        Prefers 2D+ variables with lat/lon or y/x dimensions (actual data variables,
        not 1D coordinate variables).
        Returns None if no suitable variable found.
        """
        variables = self._list_variables(path, fmt)

        if not variables:
            return None

        # Look for 2D+ variables with spatial dimensions (data variables, not coords)
        spatial_dims = {"lat", "latitude", "y", "lon", "longitude", "x", "xc", "yc"}

        for var in variables:
            shape = var.get("shape", [])
            if len(shape) >= 2:  # Must be 2D or higher
                dims = set(d.lower() for d in var.get("dimensions", []))
                if dims & spatial_dims:
                    return var["name"]

        # Fallback: return first variable that has 2D+ shape
        for var in variables:
            shape = var.get("shape", [])
            if len(shape) >= 2:
                return var["name"]

        # Ultimate fallback: first variable
        return variables[0]["name"] if variables else None

    def _get_variable_metadata(self, path: str, variable: str, fmt: str) -> dict[str, Any]:
        """Get detailed metadata for a specific variable."""
        if HAS_NETCDF4 and fmt == "netcdf":
            try:
                with netCDF4.Dataset(path, "r") as ds:  # noqa: N813
                    if variable in ds.variables:
                        var = ds.variables[variable]
                        result = {
                            "dimensions": list(var.dimensions),
                            "shape": list(var.shape),
                            "attributes": {attr: getattr(var, attr) for attr in var.ncattrs()},
                            "global_attributes": {attr: getattr(ds, attr) for attr in ds.ncattrs()},
                        }
                        return result
            except OSError:
                pass  # netCDF4 can't open — fall through to rasterio

        # Fallback: basic info from rasterio
        subdataset_uri = _build_subdataset_uri(path, variable, fmt)
        try:
            with rasterio.open(subdataset_uri) as src:
                return {
                    "shape": [src.height, src.width],
                    "band_count": src.count,
                    "dtype": str(src.dtypes[0]) if src.dtypes else None,
                }
        except rasterio.errors.RasterioIOError:
            return {}

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def _inspect(self, subdataset_uri: str) -> tuple[SpatialDescriptor, int, str]:
        """Open a subdataset and extract spatial metadata."""
        with rasterio.open(subdataset_uri) as src:
            bounds = src.bounds
            spatial = SpatialDescriptor(
                crs=str(src.crs) if src.crs else None,
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(abs(src.res[0]), abs(src.res[1])),
                band_count=src.count,
            )
            dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
            return spatial, src.count, dtype

    # -----------------------------------------------------------------------
    # Eager materialization
    # -----------------------------------------------------------------------

    def _copy_to_geotiff(self, subdataset_uri: str, workspace: Path, variable: str) -> Path:
        """Copy NetCDF variable to GeoTIFF in workspace."""
        output_path = workspace / f"{variable}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            rasterio.shutil.copy(subdataset_uri, output_path, driver="GTiff")
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise MaterializeError(subdataset_uri, f"Failed to copy to GeoTIFF: {e}") from e

        return output_path
