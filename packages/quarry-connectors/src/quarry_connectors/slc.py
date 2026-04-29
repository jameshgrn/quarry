"""SLCConnector — structural mapper for SWOT L1B HR SLC data.

Lane: connector

Maps the SWOT SLC HDF5 product structure to individual artifacts.
Each dataset (SLC arrays, calibration factors, noise vectors) is
materialized as a separate artifact via the HDF5Connector.

No processing — calibration, interferometry, and multi-look belong
in operators (SLCCalibrationOperator).

Reference: JPL D-56410 SWOT Product Description L1B HR SLC
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from quarry_core.artifact import Artifact
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

from quarry_connectors.hdf5 import HDF5Connector
from quarry_connectors.swot_utils import extract_slc_metadata

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# SWOT SLC HDF5 structural map: logical name -> (hdf5_path, role)
_SLC_DATASETS: dict[str, tuple[str, str]] = {
    "slc_plus_y": ("/slc/slc_plus_y", "data"),
    "slc_minus_y": ("/slc/slc_minus_y", "data"),
    "xfactor_plus_y": ("/xfactor/xfactor_plus_y", "calibration"),
    "xfactor_minus_y": ("/xfactor/xfactor_minus_y", "calibration"),
    "noise_plus_y": ("/noise/noise_plus_y", "noise"),
    "noise_minus_y": ("/noise/noise_minus_y", "noise"),
}

_EXTENSIONS = frozenset({".h5", ".hdf5", ".he5", ".nc"})


class SLCConnector:
    """Structural mapper for SWOT L1B HR SLC HDF5 files.

    Composes HDF5Connector for format-level reading. Adds:
    - Semantic discovery: labels each dataset with role (data/calibration/noise)
    - SWOT metadata enrichment on every artifact
    - Validation that the file has expected SLC structure

    Each dataset is materialized as a separate artifact. Processing
    (sigma0, interferogram, multi-look) belongs in operators.
    """

    def __init__(self) -> None:
        self._hdf5 = HDF5Connector()

    @property
    def name(self) -> str:
        return "slc"

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
        """Materialize a single SLC dataset as an artifact.

        source_ref formats:
            "path.h5::slc_plus_y"                — by logical name
            "path.h5::/slc/slc_plus_y"            — by HDF5 path
            "path.h5"                             — auto-select (largest 2D+ dataset)
        """
        file_path, dataset_key = self._parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        if path.suffix.lower() not in _EXTENSIONS:
            raise MaterializeError(source_ref, f"Not an HDF5 file: {path.suffix}")
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        # Resolve logical name to HDF5 path
        hdf5_dataset_path = self._resolve_dataset(dataset_key)

        # Build HDF5Connector source_ref with resolved path
        if hdf5_dataset_path is not None:
            hdf5_ref = f"{path}::{hdf5_dataset_path}"
        else:
            hdf5_ref = str(path)

        # Delegate to HDF5Connector
        result = self._hdf5.materialize(hdf5_ref, workspace, lazy=lazy)

        # Enrich with SWOT metadata
        swot_meta = self._read_swot_metadata(path)
        role = self._get_role(dataset_key)

        enriched_metadata = {
            **dict(result.artifact.metadata),
            "source": "slc",
            "role": role,
            "swot": swot_meta,
            "geographic_bounds": {
                "crs": "EPSG:4326",
                "lat_bounds": swot_meta.get("lat_bounds"),
                "lon_bounds": swot_meta.get("lon_bounds"),
            },
        }

        # Build new artifact with enriched metadata (Artifact is frozen)
        a = result.artifact
        enriched_artifact = Artifact(
            id=a.id,
            type=a.type,
            name=a.name,
            backing=a.backing,
            spatial=a.spatial,
            lineage=a.lineage,
            checks=a.checks,
            metadata=enriched_metadata,
            created_at=a.created_at,
        )

        return MaterializeResult(
            artifact=enriched_artifact,
            strategy=result.strategy,
            source_ref=source_ref,
            notes=f"SWOT SLC dataset '{dataset_key or 'auto'}' — {role}",
        )

    def discover(self, query: str | dict | None = None) -> list[CatalogEntry]:
        """Discover SLC datasets in a file or SLC files in a directory.

        When given a file path: returns one CatalogEntry per SLC dataset
        (6 entries for a standard SLC file), with role labels.

        When given a directory: scans for HDF5 files and returns one
        CatalogEntry per file.
        """
        if query is None:
            query = "."

        if isinstance(query, str):
            target = Path(query).resolve()
            recursive = False
        else:
            target = Path(query.get("path", ".")).resolve()
            recursive = query.get("recursive", False)

        if target.is_file():
            return self._discover_file(target)

        if target.is_dir():
            return self._discover_directory(target, recursive)

        return []

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get SWOT SLC metadata without full materialization."""
        file_path, _ = self._parse_source_ref(source_ref)
        path = Path(file_path).resolve()

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        swot_meta = self._read_swot_metadata(path)

        # Also get the HDF5 group structure
        hdf5_meta = self._hdf5.metadata(str(path))

        return {
            **swot_meta,
            "datasets": list(_SLC_DATASETS.keys()),
            "group_structure": hdf5_meta.get("group_structure", {}),
        }

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None]:
        """Parse source_ref into (file_path, dataset_key | None)."""
        raw = str(source_ref).strip()

        if "::" in raw:
            path_part, ds_part = raw.split("::", 1)
            return path_part.strip(), ds_part.strip()

        return raw, None

    def _resolve_dataset(self, dataset_key: str | None) -> str | None:
        """Resolve a logical name or HDF5 path to the actual HDF5 dataset path.

        Returns None if dataset_key is None (auto-select).
        """
        if dataset_key is None:
            return None

        # Check if it's a logical name (e.g., "slc_plus_y")
        if dataset_key in _SLC_DATASETS:
            return _SLC_DATASETS[dataset_key][0]

        # Check if it's already an HDF5 path (e.g., "/slc/slc_plus_y")
        normalized = dataset_key if dataset_key.startswith("/") else f"/{dataset_key}"
        for _, (hdf5_path, _) in _SLC_DATASETS.items():
            if hdf5_path == normalized:
                return hdf5_path

        # Pass through as-is — let HDF5Connector validate
        return dataset_key

    def _get_role(self, dataset_key: str | None) -> str:
        """Get the role (data/calibration/noise) for a dataset key."""
        if dataset_key is None:
            return "auto"

        if dataset_key in _SLC_DATASETS:
            return _SLC_DATASETS[dataset_key][1]

        normalized = dataset_key if dataset_key.startswith("/") else f"/{dataset_key}"
        for _, (hdf5_path, role) in _SLC_DATASETS.items():
            if hdf5_path == normalized:
                return role

        return "unknown"

    def _read_swot_metadata(self, path: Path) -> dict[str, Any]:
        """Extract SWOT SLC metadata from HDF5 root attributes."""
        import h5py

        with h5py.File(str(path), "r") as f:
            return extract_slc_metadata(dict(f.attrs))

    def _discover_file(self, path: Path) -> list[CatalogEntry]:
        """List all SLC datasets in a file with semantic labels."""
        import h5py

        entries: list[CatalogEntry] = []

        with h5py.File(str(path), "r") as f:
            swot_meta = extract_slc_metadata(dict(f.attrs))
            available = self._walk_datasets(f)

            for logical_name, (hdf5_path, role) in _SLC_DATASETS.items():
                if hdf5_path.lstrip("/") not in available:
                    continue

                ds = f[hdf5_path]
                entries.append(
                    CatalogEntry(
                        source_ref=f"{path}::{logical_name}",
                        name=logical_name,
                        spatial_hint={
                            "crs": "EPSG:4326",
                            "extent": (
                                swot_meta["lon_bounds"][0],
                                swot_meta["lat_bounds"][0],
                                swot_meta["lon_bounds"][1],
                                swot_meta["lat_bounds"][1],
                            ),
                        },
                        metadata={
                            "role": role,
                            "shape": ds.shape,
                            "dtype": str(ds.dtype),
                            "cycle": swot_meta["cycle"],
                            "pass": swot_meta["pass_number"],
                            "swath": swot_meta["swath_side"],
                            "tile": swot_meta["tile_name"],
                        },
                    )
                )

        return entries

    def _discover_directory(self, directory: Path, recursive: bool) -> list[CatalogEntry]:
        """Scan for SLC HDF5 files in a directory."""
        entries: list[CatalogEntry] = []
        pattern = "**/*" if recursive else "*"

        for p in directory.glob(pattern):
            if p.suffix.lower() not in _EXTENSIONS:
                continue
            try:
                swot_meta = self._read_swot_metadata(p)
                entries.append(
                    CatalogEntry(
                        source_ref=str(p),
                        name=swot_meta.get("tile_name") or p.stem,
                        spatial_hint={
                            "crs": "EPSG:4326",
                            "extent": (
                                swot_meta["lon_bounds"][0],
                                swot_meta["lat_bounds"][0],
                                swot_meta["lon_bounds"][1],
                                swot_meta["lat_bounds"][1],
                            ),
                        },
                        metadata={
                            "cycle": swot_meta["cycle"],
                            "pass": swot_meta["pass_number"],
                            "swath": swot_meta["swath_side"],
                            "size_bytes": p.stat().st_size,
                        },
                    )
                )
            except KeyError:
                # Not a SWOT SLC file — skip
                continue

        return entries

    def _walk_datasets(self, f: Any) -> set[str]:
        """Get all dataset paths in the HDF5 file (without leading slash)."""
        import h5py

        paths: set[str] = set()

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                paths.add(name)

        f.visititems(_visitor)
        return paths
