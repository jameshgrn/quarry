"""HDF5Connector — materializes arbitrary HDF5 datasets into canonical artifacts.

Lane: connector

Format-level connector using h5py for direct group/dataset access to HDF5 files.
Handles nested group structures, coordinate inference, CRS detection, and complex
number arrays. Product-agnostic.

Differentiation from NetCDFConnector:
- NetCDFConnector uses GDAL/rasterio for CF-convention NetCDF (subdataset URIs)
- HDF5Connector uses h5py for arbitrary HDF5 with nested groups that GDAL
  doesn't always expose well (e.g., SWOT products, complex scientific data)

source_ref format: "path.h5" (auto-select) or "path.h5::/group/dataset"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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
    import h5py as h5py_types
    from quarry_core.source_ref import SourceRef

# Names that identify coordinate arrays (lowercased basename)
_COORD_NAMES = frozenset(
    {
        "lat",
        "latitude",
        "lon",
        "longitude",
        "x",
        "y",
        "time",
        "z",
        "range",
        "azimuth",
        "range_index",
        "azimuth_index",
    }
)

# CF-convention coordinate unit patterns
_GEO_LAT_UNITS = frozenset({"degrees_north", "degree_north", "degree_n", "degrees_n"})
_GEO_LON_UNITS = frozenset({"degrees_east", "degree_east", "degree_e", "degrees_e"})

# Accepted file extensions
_EXTENSIONS = frozenset({".h5", ".hdf5", ".hdf", ".he5"})


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """Metadata about a single dataset discovered in an HDF5 file."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    ndim: int
    is_complex_pair: bool
    is_coordinate: bool
    attributes: dict[str, Any]
    chunk_shape: tuple[int, ...] | None
    compression: str | None
    size_bytes: int


class HDF5Connector:
    """Materializes arbitrary HDF5 datasets into canonical Quarry artifacts.

    Uses h5py for direct group/dataset access. Handles nested group structures,
    coordinate inference, CRS detection, and complex number arrays.
    """

    @property
    def name(self) -> str:
        return "hdf5"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.METADATA_ONLY
            | ConnectorCapability.MATERIALIZE_LAZY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize an HDF5 dataset as an artifact.

        source_ref formats:
            "path/to/file.h5"                  — auto-select dataset
            "path/to/file.h5::/group/dataset"   — explicit dataset path
        """
        import h5py

        file_path, dataset_path = self._parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        if path.suffix.lower() not in _EXTENSIONS:
            raise MaterializeError(source_ref, f"Not an HDF5 file: {path.suffix}")
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        with h5py.File(str(path), "r") as f:
            datasets = self._discover_datasets(f)

            if dataset_path is not None:
                info = self._find_dataset(datasets, dataset_path, source_ref)
            else:
                info = self._auto_select_dataset(datasets)
                if info is None:
                    raise MaterializeError(source_ref, "No suitable 2D+ dataset found in file")

            spatial, coord_mapping = self._infer_coordinates(f, info)

            if lazy:
                artifact = self._build_lazy_artifact(path, info, spatial)
                return MaterializeResult(
                    artifact=artifact,
                    strategy="lazy_handle",
                    source_ref=source_ref,
                    notes=f"HDF5 dataset '{info.path}' — metadata only",
                )

            artifact = self._build_eager_artifact(f, path, workspace, info, spatial)

        return MaterializeResult(
            artifact=artifact,
            strategy="normalized",
            source_ref=source_ref,
            notes=f"HDF5 dataset '{info.path}' copied to GeoTIFF",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List datasets in an HDF5 file, or HDF5 files in a directory.

        query as str: path to file (list datasets) or directory (list files).
        query as dict: {"path": str, "recursive": bool}.
        """
        import h5py

        if query is None:
            query = "."

        if isinstance(query, str):
            target = Path(query).resolve()
            recursive = False
        else:
            target = Path(query.get("path", ".")).resolve()
            recursive = query.get("recursive", False)

        if target.is_file():
            return self._discover_file(target, h5py)

        if target.is_dir():
            return self._discover_directory(target, recursive)

        return []

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata about an HDF5 dataset without materializing."""
        import h5py

        file_path, dataset_path = self._parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        with h5py.File(str(path), "r") as f:
            datasets = self._discover_datasets(f)

            if dataset_path is not None:
                info = self._find_dataset(datasets, dataset_path, source_ref)
            else:
                info = self._auto_select_dataset(datasets)
                if info is None:
                    raise MaterializeError(source_ref, "No suitable 2D+ dataset found in file")

            spatial, coord_mapping = self._infer_coordinates(f, info)
            group_tree = self._group_structure(f)
            global_attrs = self._decode_attrs(dict(f.attrs))

        return {
            "path": str(path),
            "dataset": info.path,
            "shape": info.shape,
            "dtype": info.dtype,
            "ndim": info.ndim,
            "is_complex_pair": info.is_complex_pair,
            "crs": spatial.crs,
            "extent": spatial.extent,
            "resolution": spatial.resolution,
            "chunk_shape": info.chunk_shape,
            "compression": info.compression,
            "dataset_attributes": info.attributes,
            "global_attributes": global_attrs,
            "group_structure": group_tree,
            "coordinate_arrays": coord_mapping,
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None]:
        """Parse source_ref into (file_path, dataset_path | None)."""
        from quarry_core.source_ref import SourceRef, SourceRefKind

        if isinstance(source_ref, SourceRef):
            if source_ref.kind in (SourceRefKind.LOCAL_PATH, SourceRefKind.LOCAL_RASTER):
                params = source_ref.params or {}
                path = params.get("path", source_ref.raw)
                dataset = params.get("dataset")
                return path, dataset
            raw = source_ref.raw.strip()
        else:
            raw = source_ref.strip()

        if "::" in raw:
            path_part, ds_part = raw.split("::", 1)
            return path_part.strip(), ds_part.strip()

        return raw, None

    # -----------------------------------------------------------------------
    # Dataset discovery
    # -----------------------------------------------------------------------

    def _discover_datasets(self, f: h5py_types.File) -> list[HDF5DatasetInfo]:
        """Recursively discover all datasets in an HDF5 file."""
        import h5py

        results: list[HDF5DatasetInfo] = []

        def _visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return

            ds_path = f"/{name}"
            shape = tuple(obj.shape)
            ndim = obj.ndim
            dtype_str = str(obj.dtype)
            basename = name.rsplit("/", 1)[-1].lower()

            # Coordinate classification
            is_coord = ndim == 1 and basename in _COORD_NAMES
            if not is_coord and ndim == 1:
                attrs = dict(obj.attrs)
                if "axis" in attrs:
                    is_coord = True
                units = self._decode_val(attrs.get("units", ""))
                if isinstance(units, str) and units.lower() in (_GEO_LAT_UNITS | _GEO_LON_UNITS):
                    is_coord = True

            # Complex pair classification: (H, W, 2) with float dtype
            is_complex = ndim >= 3 and shape[-1] == 2 and np.issubdtype(obj.dtype, np.floating)

            results.append(
                HDF5DatasetInfo(
                    path=ds_path,
                    shape=shape,
                    dtype=dtype_str,
                    ndim=ndim,
                    is_complex_pair=is_complex,
                    is_coordinate=is_coord,
                    attributes=self._decode_attrs(dict(obj.attrs)),
                    chunk_shape=tuple(obj.chunks) if obj.chunks else None,
                    compression=obj.compression,
                    size_bytes=obj.id.get_storage_size(),
                )
            )

        f.visititems(_visitor)
        return results

    def _auto_select_dataset(self, datasets: list[HDF5DatasetInfo]) -> HDF5DatasetInfo | None:
        """Pick the best 2D+ non-coordinate dataset (largest H*W, alphabetical tiebreak)."""
        candidates = [d for d in datasets if not d.is_coordinate and d.ndim >= 2]
        if not candidates:
            return None

        def _sort_key(d: HDF5DatasetInfo) -> tuple[int, str]:
            # For complex pairs, spatial footprint is shape[0] * shape[1]
            # For regular 2D+, it's shape[-2] * shape[-1]
            if d.is_complex_pair and d.ndim >= 3:
                area = d.shape[0] * d.shape[1]
            else:
                area = d.shape[-2] * d.shape[-1]
            return (-area, d.path)

        candidates.sort(key=_sort_key)
        return candidates[0]

    def _find_dataset(
        self,
        datasets: list[HDF5DatasetInfo],
        dataset_path: str,
        source_ref: SourceRef | str,
    ) -> HDF5DatasetInfo:
        """Find a specific dataset by path, raising if not found."""
        # Normalize: ensure leading slash
        if not dataset_path.startswith("/"):
            dataset_path = f"/{dataset_path}"

        for d in datasets:
            if d.path == dataset_path:
                return d

        available = [d.path for d in datasets]
        raise MaterializeError(
            source_ref,
            f"Dataset '{dataset_path}' not found. Available: {available}",
        )

    # -----------------------------------------------------------------------
    # Coordinate and CRS inference
    # -----------------------------------------------------------------------

    def _infer_coordinates(
        self,
        f: h5py_types.File,
        info: HDF5DatasetInfo,
    ) -> tuple[SpatialDescriptor, dict[str, str]]:
        """Infer spatial coordinates for a dataset.

        Returns (SpatialDescriptor, coord_mapping) where coord_mapping maps
        axis role ('x', 'y') to the HDF5 path of the coordinate array.
        """
        # Spatial dimensions of the data
        if info.is_complex_pair and info.ndim >= 3:
            height, width = info.shape[0], info.shape[1]
        else:
            height, width = info.shape[-2], info.shape[-1]

        coord_mapping: dict[str, str] = {}

        # Strategy 1: CF coordinates attribute on the dataset
        ds = f[info.path]
        coord_attr = self._decode_val(ds.attrs.get("coordinates", ""))
        if isinstance(coord_attr, str) and coord_attr.strip():
            coord_mapping = self._resolve_cf_coordinates(f, coord_attr, info, height, width)

        # Strategy 2: Sibling arrays in same group
        if not coord_mapping:
            parent_path = info.path.rsplit("/", 1)[0] or "/"
            coord_mapping = self._find_coord_arrays(f, parent_path, height, width)

        # Strategy 3: Root-level coordinate arrays
        if not coord_mapping:
            coord_mapping = self._find_coord_arrays(f, "/", height, width)

        # Build extent from coordinate arrays
        extent = None
        resolution = None

        if "x" in coord_mapping and "y" in coord_mapping:
            x_arr = f[coord_mapping["x"]][:]
            y_arr = f[coord_mapping["y"]][:]

            if x_arr.ndim == 2 and y_arr.ndim == 2:
                # 2D coordinate grids
                extent = (
                    float(np.nanmin(x_arr)),
                    float(np.nanmin(y_arr)),
                    float(np.nanmax(x_arr)),
                    float(np.nanmax(y_arr)),
                )
            elif x_arr.ndim == 1 and y_arr.ndim == 1:
                extent = (
                    float(x_arr.min()),
                    float(y_arr.min()),
                    float(x_arr.max()),
                    float(y_arr.max()),
                )
                if len(x_arr) > 1 and len(y_arr) > 1:
                    resolution = (
                        float(abs(x_arr[1] - x_arr[0])),
                        float(abs(y_arr[1] - y_arr[0])),
                    )

        # Fallback: pixel indices
        if extent is None:
            extent = (0, 0, width, height)

        crs = self._infer_crs(f, info, coord_mapping)

        return (
            SpatialDescriptor(
                crs=crs,
                extent=extent,
                resolution=resolution,
                band_count=2 if info.is_complex_pair else 1,
            ),
            {k: v for k, v in coord_mapping.items()},
        )

    def _resolve_cf_coordinates(
        self,
        f: h5py_types.File,
        coord_attr: str,
        info: HDF5DatasetInfo,
        height: int,
        width: int,
    ) -> dict[str, str]:
        """Resolve CF-style 'coordinates' attribute to axis mappings."""
        mapping: dict[str, str] = {}
        parent_path = info.path.rsplit("/", 1)[0] or "/"

        for name in coord_attr.split():
            # Try relative to parent group, then root
            for prefix in [parent_path, "/"]:
                candidate = f"{prefix}/{name}".replace("//", "/")
                if candidate in f:
                    role = self._classify_coord_role(name, f[candidate])
                    if role:
                        mapping[role] = candidate
                    break

        return mapping

    def _find_coord_arrays(
        self,
        f: h5py_types.File,
        group_path: str,
        height: int,
        width: int,
    ) -> dict[str, str]:
        """Find coordinate arrays in a group that match the data dimensions."""
        import h5py

        mapping: dict[str, str] = {}
        grp = f[group_path]

        for key in grp:
            item = grp[key]
            if not isinstance(item, h5py.Dataset):
                continue
            if item.ndim == 1:
                size = item.shape[0]
                if size not in (height, width):
                    continue
            elif item.ndim == 2:
                if item.shape != (height, width):
                    continue
            else:
                continue

            role = self._classify_coord_role(key, item)
            if role and role not in mapping:
                full_path = f"{group_path}/{key}".replace("//", "/")
                mapping[role] = full_path

        return mapping

    def _classify_coord_role(self, name: str, ds: h5py_types.Dataset) -> str | None:
        """Classify a dataset as 'x' or 'y' coordinate, or None."""
        basename = name.rsplit("/", 1)[-1].lower()
        units = self._decode_val(ds.attrs.get("units", ""))
        if isinstance(units, str):
            units = units.lower()
        else:
            units = ""

        x_names = {"lon", "longitude", "x"}
        y_names = {"lat", "latitude", "y"}

        if basename in x_names or units in _GEO_LON_UNITS:
            return "x"
        if basename in y_names or units in _GEO_LAT_UNITS:
            return "y"

        axis_attr = self._decode_val(ds.attrs.get("axis", ""))
        if isinstance(axis_attr, str):
            if axis_attr.upper() == "X":
                return "x"
            if axis_attr.upper() == "Y":
                return "y"

        return None

    def _infer_crs(
        self,
        f: h5py_types.File,
        info: HDF5DatasetInfo,
        coord_mapping: dict[str, str],
    ) -> str | None:
        """Infer CRS from attributes or coordinate units."""
        # Check dataset -> parent group -> root for crs/spatial_ref
        ds = f[info.path]
        for attrs_source in [ds.attrs, f[info.path.rsplit("/", 1)[0] or "/"].attrs, f.attrs]:
            for key in ("crs", "spatial_ref", "grid_mapping"):
                if key in attrs_source:
                    val = self._decode_val(attrs_source[key])
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # If coordinate arrays have geographic units, assume EPSG:4326
        for role, path in coord_mapping.items():
            units = self._decode_val(f[path].attrs.get("units", ""))
            if isinstance(units, str) and units.lower() in (_GEO_LAT_UNITS | _GEO_LON_UNITS):
                return "EPSG:4326"

        return None

    # -----------------------------------------------------------------------
    # Artifact construction
    # -----------------------------------------------------------------------

    def _build_lazy_artifact(
        self,
        path: Path,
        info: HDF5DatasetInfo,
        spatial: SpatialDescriptor,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.RASTER,
            name=self._sanitize_name(info.path),
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=f"{path}::{info.path}",
            ),
            spatial=spatial,
            lineage=Lineage(
                operation="materialize",
                params={
                    "source": "hdf5",
                    "dataset": info.path,
                    "lazy": True,
                },
            ),
            metadata={
                "format": "hdf5",
                "dataset_path": info.path,
                "shape": info.shape,
                "dtype": info.dtype,
                "is_complex_pair": info.is_complex_pair,
            },
        )

    def _build_eager_artifact(
        self,
        f: h5py_types.File,
        path: Path,
        workspace: Path,
        info: HDF5DatasetInfo,
        spatial: SpatialDescriptor,
    ) -> Artifact:
        data = f[info.path][:]

        if info.is_complex_pair:
            # Convert (H, W, 2) float -> 2-band (real, imag)
            real_part = data[..., 0].astype(np.float32)
            imag_part = data[..., 1].astype(np.float32)
            band_data = np.stack([real_part, imag_part], axis=0)
            band_names = ["real", "imag"]
        else:
            arr = data.astype(np.float32) if not np.issubdtype(data.dtype, np.floating) else data
            if arr.ndim == 2:
                band_data = arr[np.newaxis, ...]
            else:
                band_data = arr
            band_names = None

        output_name = self._sanitize_name(info.path)
        output_path = workspace / f"{output_name}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_geotiff(band_data, output_path, spatial, band_names)

        return Artifact(
            type=ArtifactType.RASTER,
            name=output_name,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs=spatial.crs,
                extent=spatial.extent,
                resolution=spatial.resolution,
                band_count=band_data.shape[0],
            ),
            lineage=Lineage(
                operation="materialize",
                params={
                    "source": "hdf5",
                    "dataset": info.path,
                    "output_format": "geotiff",
                },
            ),
            metadata={
                "format": "hdf5",
                "dataset_path": info.path,
                "dtype": info.dtype,
                "is_complex_pair": info.is_complex_pair,
            },
        )

    # -----------------------------------------------------------------------
    # GeoTIFF writing
    # -----------------------------------------------------------------------

    def _write_geotiff(
        self,
        data: np.ndarray,
        output_path: Path,
        spatial: SpatialDescriptor,
        band_names: list[str] | None = None,
    ) -> None:
        """Write (bands, H, W) array to GeoTIFF."""
        import rasterio
        from rasterio.transform import from_bounds

        bands, height, width = data.shape
        extent = spatial.extent or (0, 0, width, height)
        transform = from_bounds(extent[0], extent[1], extent[2], extent[3], width, height)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": str(data.dtype),
            "crs": spatial.crs,
            "transform": transform,
            "compress": "deflate",
            "nodata": np.nan if np.issubdtype(data.dtype, np.floating) else None,
        }

        with rasterio.open(str(output_path), "w", **profile) as dst:
            for i in range(bands):
                dst.write(data[i], i + 1)
                if band_names and i < len(band_names):
                    dst.set_band_description(i + 1, band_names[i])

    # -----------------------------------------------------------------------
    # Discover helpers
    # -----------------------------------------------------------------------

    def _discover_file(self, path: Path, h5py_mod: Any) -> list[CatalogEntry]:
        """List datasets within a single HDF5 file."""
        entries: list[CatalogEntry] = []

        try:
            with h5py_mod.File(str(path), "r") as f:
                datasets = self._discover_datasets(f)
                for info in datasets:
                    if info.is_coordinate:
                        continue
                    entries.append(
                        CatalogEntry(
                            source_ref=f"{path}::{info.path}",
                            name=info.path,
                            metadata={
                                "shape": info.shape,
                                "dtype": info.dtype,
                                "is_complex_pair": info.is_complex_pair,
                                "compression": info.compression,
                                "chunk_shape": info.chunk_shape,
                                "attributes": info.attributes,
                            },
                        )
                    )
        except OSError as e:
            raise MaterializeError(str(path), f"Cannot open HDF5 file: {e}") from e

        return entries

    def _discover_directory(self, directory: Path, recursive: bool) -> list[CatalogEntry]:
        """List HDF5 files in a directory."""
        entries: list[CatalogEntry] = []
        pattern = "**/*" if recursive else "*"

        for p in directory.glob(pattern):
            if p.suffix.lower() in _EXTENSIONS:
                entries.append(
                    CatalogEntry(
                        source_ref=str(p),
                        name=p.stem,
                        metadata={"size_bytes": p.stat().st_size},
                    )
                )

        return entries

    # -----------------------------------------------------------------------
    # Group structure
    # -----------------------------------------------------------------------

    def _group_structure(self, f: h5py_types.File) -> dict[str, list[str]]:
        """Build a lightweight tree: group path -> list of dataset names."""
        import h5py

        tree: dict[str, list[str]] = {}

        # Root-level items
        root_datasets = [k for k in f if isinstance(f[k], h5py.Dataset)]
        if root_datasets:
            tree["/"] = root_datasets

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                group_path = f"/{name}"
                ds_names = [k for k in obj if isinstance(obj[k], h5py.Dataset)]
                if ds_names:
                    tree[group_path] = ds_names

        f.visititems(_visitor)
        return tree

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _decode_val(self, val: Any) -> Any:
        """Decode a single HDF5 attribute value."""
        if isinstance(val, bytes):
            return val.decode()
        if isinstance(val, np.ndarray):
            if val.size == 1:
                item = val.item()
                return item.decode() if isinstance(item, bytes) else item
            return val.tolist()
        if isinstance(val, np.generic):
            item = val.item()
            return item.decode() if isinstance(item, bytes) else item
        return val

    def _decode_attrs(self, attrs: dict[str, Any]) -> dict[str, Any]:
        """Decode all attribute values in a dict."""
        return {k: self._decode_val(v) for k, v in attrs.items()}

    def _sanitize_name(self, dataset_path: str) -> str:
        """Convert '/group/subgroup/dataset' to 'group_subgroup_dataset'."""
        return dataset_path.strip("/").replace("/", "_") or "dataset"
