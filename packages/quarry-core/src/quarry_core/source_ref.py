"""SourceRef — typed envelope for source references.

A SourceRef wraps the raw string that connectors consume, adding:
- kind: classification tag for routing
- params: optional parsed fields for connector-specific structure
- raw: the original string, always preserved, always round-trippable

SourceRef does NOT replace source_ref: str in the Connector protocol.
It lives alongside — callers construct them, routers inspect them,
connectors receive str(ref) or ref.raw.

Design principles:
- Thin tagged envelope, not a class hierarchy
- Factory methods for explicit construction (eliminates heuristic parsing)
- infer() for best-effort classification from raw strings
- Frozen, hashable, equality by value
- Zero deps (quarry-core)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceRefKind(Enum):
    """Classification of source reference shapes."""

    LOCAL_PATH = "local_path"  # filesystem path
    REMOTE_URI = "remote_uri"  # http/https/s3/gs URL
    CATALOG_ITEM = "catalog_item"  # catalog/collection reference (STAC-like)
    DATABASE_REF = "database_ref"  # schema.table or SQL query
    UNKNOWN = "unknown"  # unclassified raw string


def _freeze_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of params for immutability."""
    return dict(params)


@dataclass(frozen=True)
class SourceRef:
    """Typed envelope for source references.

    Always carries the raw string. Optionally carries parsed structure.
    Can be passed to any connector via str(ref) or ref.raw.
    """

    raw: str
    kind: SourceRefKind = SourceRefKind.UNKNOWN
    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=True)

    def __str__(self) -> str:
        """Return raw string — always works where source_ref: str is expected."""
        return self.raw

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceRef):
            return NotImplemented
        return self.raw == other.raw and self.kind == other.kind and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.raw, self.kind))

    # -----------------------------------------------------------------------
    # Factory methods: explicit construction
    # -----------------------------------------------------------------------

    @classmethod
    def local(cls, path: str) -> SourceRef:
        """Construct a local path reference."""
        return cls(
            raw=path,
            kind=SourceRefKind.LOCAL_PATH,
            params={"path": path},
        )

    @classmethod
    def uri(cls, url: str) -> SourceRef:
        """Construct a remote URI reference."""
        scheme = url.split("://", 1)[0] if "://" in url else ""
        return cls(
            raw=url,
            kind=SourceRefKind.REMOTE_URI,
            params={"scheme": scheme, "url": url},
        )

    @classmethod
    def stac(cls, collection: str, item: str, *, asset: str | None = None) -> SourceRef:
        """Construct a STAC catalog item reference.

        Produces raw in the format connectors already expect:
            "collection/item" or "collection/item::asset"
        """
        raw = f"{collection}/{item}"
        if asset:
            raw += f"::{asset}"

        params: dict[str, Any] = {"collection": collection, "item": item}
        if asset:
            params["asset"] = asset

        return cls(raw=raw, kind=SourceRefKind.CATALOG_ITEM, params=params)

    @classmethod
    def postgis(cls, schema: str, table: str) -> SourceRef:
        """Construct a PostGIS table reference.

        Produces raw in the format: "schema.table"
        """
        return cls(
            raw=f"{schema}.{table}",
            kind=SourceRefKind.DATABASE_REF,
            params={"schema": schema, "table": table},
        )

    @classmethod
    def postgis_query(cls, sql: str) -> SourceRef:
        """Construct a PostGIS query reference.

        Produces raw = the SQL string itself.
        """
        return cls(
            raw=sql,
            kind=SourceRefKind.DATABASE_REF,
            params={"query": sql},
        )

    # -----------------------------------------------------------------------
    # Inference: best-effort classification from raw string
    # -----------------------------------------------------------------------

    @classmethod
    def infer(cls, raw: str) -> SourceRef:
        """Best-effort classification of a raw source reference string.

        This is what routing/adapter layers use to classify incoming refs.
        Honest about ambiguity — returns UNKNOWN when genuinely unclear.

        Classification priority:
        1. URL schemes (http, https, s3, gs, az) → REMOTE_URI
        2. SQL query (starts with SELECT) → DATABASE_REF
        3. Absolute/relative path (starts with /, ./, ../) → LOCAL_PATH
        4. Path with file extension and separators → LOCAL_PATH
        5. STAC pattern (word/word or word/word::word) → CATALOG_ITEM
        6. Database pattern (word.word, no path separators or extension) → DATABASE_REF
        7. Otherwise → UNKNOWN
        """
        stripped = raw.strip()

        if not stripped:
            return cls(raw=raw, kind=SourceRefKind.UNKNOWN)

        # 1. URL schemes
        if "://" in stripped:
            scheme = stripped.split("://", 1)[0].lower()
            if scheme in ("http", "https", "s3", "gs", "az", "ftp"):
                return cls(
                    raw=raw,
                    kind=SourceRefKind.REMOTE_URI,
                    params={"scheme": scheme, "url": stripped},
                )

        # 2. SQL query
        if stripped.upper().startswith("SELECT "):
            return cls(
                raw=raw,
                kind=SourceRefKind.DATABASE_REF,
                params={"query": stripped},
            )

        # 3. Absolute or relative path
        if stripped.startswith("/") or stripped.startswith("./") or stripped.startswith("../"):
            return cls(
                raw=raw,
                kind=SourceRefKind.LOCAL_PATH,
                params={"path": stripped},
            )

        # 4. Has path separator AND file extension → LOCAL_PATH
        if "/" in stripped and "." in stripped.rsplit("/", 1)[-1]:
            # Looks like a path with filename.ext
            return cls(
                raw=raw,
                kind=SourceRefKind.LOCAL_PATH,
                params={"path": stripped},
            )

        # 5. STAC pattern: word/word or word/word::asset
        if "/" in stripped and not stripped.startswith("/"):
            parts = stripped.split("/", 1)
            if "::" in parts[1]:
                item_part, asset = parts[1].rsplit("::", 1)
                return cls(
                    raw=raw,
                    kind=SourceRefKind.CATALOG_ITEM,
                    params={"collection": parts[0], "item": item_part, "asset": asset},
                )
            return cls(
                raw=raw,
                kind=SourceRefKind.CATALOG_ITEM,
                params={"collection": parts[0], "item": parts[1]},
            )

        # 6. Database pattern: word.word (no path separators, no file extension pattern)
        if "." in stripped and "/" not in stripped:
            # Check it's not obviously a filename (common extensions)
            ext = stripped.rsplit(".", 1)[-1].lower()
            file_extensions = {
                "tif",
                "tiff",
                "geotiff",
                "jp2",
                "nc",
                "vrt",
                "hgt",
                "shp",
                "geojson",
                "gpkg",
                "kml",
                "gml",
                "fgb",
                "parquet",
                "csv",
                "json",
                "xml",
                "txt",
                "pdf",
                "png",
                "jpg",
            }
            if ext not in file_extensions:
                parts = stripped.split(".", 1)
                return cls(
                    raw=raw,
                    kind=SourceRefKind.DATABASE_REF,
                    params={"schema": parts[0], "table": parts[1]},
                )

        # 7. Unknown
        return cls(raw=raw, kind=SourceRefKind.UNKNOWN)
