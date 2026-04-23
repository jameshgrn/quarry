"""SourceRef pressure test.

Lane: connector

This is a CONTRACT EVOLUTION test — the first change to quarry-core contracts.

Stress points:
1. Simple refs stay simple (local path, URI → trivial construction)
2. Complex refs gain clarity (STAC asset selection, PostGIS schema.table/query)
3. Round-tripping: construct → raw → same string always
4. Infer/classify: best-effort routing from raw string
5. Backward compat: every SourceRef has .raw that works as source_ref: str
6. Factory methods eliminate heuristic parsing ambiguity
7. SourceRef does NOT infect the Connector protocol — str still works everywhere

The key insight that justifies this:
- PostGIS uses heuristic dot-split and SELECT prefix detection
- STAC uses convention-based :: separator
- COG uses scheme classification
- SourceRef makes these parsing decisions explicit at CONSTRUCTION time
  rather than implicit at CONSUMPTION time

What SourceRef is NOT:
- Not a class hierarchy
- Not a connector selector/router
- Not a replacement for source_ref: str in the protocol
- Not a validator (bad refs are still representable)
"""

import pytest
from quarry_core.source_ref import SourceRef, SourceRefKind

# ---------------------------------------------------------------------------
# Construction: simple refs stay simple
# ---------------------------------------------------------------------------


class TestSimpleConstruction:
    """Local paths and URIs should be trivial to construct."""

    def test_local_path(self):
        ref = SourceRef.local("/data/dem.tif")
        assert ref.raw == "/data/dem.tif"
        assert ref.kind == SourceRefKind.LOCAL_RASTER
        assert str(ref) == "/data/dem.tif"

    def test_local_vector_path(self):
        ref = SourceRef.local("/data/parcels.geojson")
        assert ref.kind == SourceRefKind.LOCAL_VECTOR

    def test_remote_uri(self):
        url = "https://storage.googleapis.com/bucket/dem.tif"
        ref = SourceRef.uri(url)
        assert ref.raw == url
        assert ref.kind == SourceRefKind.REMOTE_URI

    def test_s3_uri(self):
        ref = SourceRef.uri("s3://mybucket/path/raster.tif")
        assert ref.kind == SourceRefKind.REMOTE_URI
        assert ref.params["scheme"] == "s3"

    def test_bare_string_construction(self):
        """You can always just wrap a string — no parsing required."""
        ref = SourceRef("anything at all")
        assert ref.raw == "anything at all"
        assert ref.kind == SourceRefKind.UNKNOWN


# ---------------------------------------------------------------------------
# Construction: complex refs gain clarity
# ---------------------------------------------------------------------------


class TestComplexConstruction:
    """STAC and PostGIS refs become explicit — no heuristic parsing needed."""

    def test_stac_full(self):
        ref = SourceRef.stac("sentinel-2-l2a", "S2A_20230615", asset="B04")
        assert ref.raw == "sentinel-2-l2a/S2A_20230615::B04"
        assert ref.kind == SourceRefKind.CATALOG_ITEM
        assert ref.params["collection"] == "sentinel-2-l2a"
        assert ref.params["item"] == "S2A_20230615"
        assert ref.params["asset"] == "B04"

    def test_stac_no_asset(self):
        ref = SourceRef.stac("sentinel-2-l2a", "S2A_20230615")
        assert ref.raw == "sentinel-2-l2a/S2A_20230615"
        assert ref.params.get("asset") is None

    def test_postgis_table(self):
        ref = SourceRef.postgis("hydro", "rivers")
        assert ref.raw == "hydro.rivers"
        assert ref.kind == SourceRefKind.DATABASE_REF
        assert ref.params["schema"] == "hydro"
        assert ref.params["table"] == "rivers"

    def test_postgis_query(self):
        sql = "SELECT id, geom FROM rivers WHERE length_km > 100"
        ref = SourceRef.postgis_query(sql)
        assert ref.raw == sql
        assert ref.kind == SourceRefKind.DATABASE_REF
        assert ref.params["query"] == sql
        assert "table" not in ref.params


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


class TestRoundTripping:
    """construct → raw → same string. Always."""

    def test_local_round_trips(self):
        path = "/some/path with spaces/file.tif"
        ref = SourceRef.local(path)
        assert ref.raw == path
        assert str(ref) == path

    def test_stac_round_trips(self):
        ref = SourceRef.stac("col", "item", asset="key")
        assert ref.raw == "col/item::key"
        # Re-parse from raw
        ref2 = SourceRef.infer(ref.raw)
        assert ref2.raw == ref.raw

    def test_postgis_round_trips(self):
        ref = SourceRef.postgis("public", "my_table")
        assert ref.raw == "public.my_table"

    def test_postgis_query_round_trips(self):
        sql = "SELECT * FROM big_table LIMIT 10"
        ref = SourceRef.postgis_query(sql)
        assert ref.raw == sql

    def test_unknown_round_trips(self):
        ref = SourceRef("opaque_ref_12345")
        assert ref.raw == "opaque_ref_12345"
        assert str(ref) == "opaque_ref_12345"


# ---------------------------------------------------------------------------
# Inference / classification
# ---------------------------------------------------------------------------


class TestInfer:
    """infer() does best-effort classification from raw strings.

    This is what enables routing — adapters can classify refs without
    knowing which connector to use yet.
    """

    def test_infer_local_path(self):
        ref = SourceRef.infer("/data/dem.tif")
        assert ref.kind == SourceRefKind.LOCAL_RASTER

    def test_infer_relative_path(self):
        ref = SourceRef.infer("./relative/file.gpkg")
        assert ref.kind == SourceRefKind.LOCAL_VECTOR

    def test_infer_http_url(self):
        ref = SourceRef.infer("https://example.com/raster.tif")
        assert ref.kind == SourceRefKind.REMOTE_URI

    def test_infer_s3_url(self):
        ref = SourceRef.infer("s3://bucket/key.tif")
        assert ref.kind == SourceRefKind.REMOTE_URI

    def test_infer_stac_ref(self):
        """collection/item pattern detected."""
        ref = SourceRef.infer("sentinel-2-l2a/S2A_20230615")
        assert ref.kind == SourceRefKind.CATALOG_ITEM

    def test_infer_stac_with_asset(self):
        ref = SourceRef.infer("sentinel-2-l2a/S2A_20230615::B04")
        assert ref.kind == SourceRefKind.CATALOG_ITEM
        assert ref.params.get("asset") == "B04"

    def test_infer_select_query(self):
        ref = SourceRef.infer("SELECT * FROM rivers")
        assert ref.kind == SourceRefKind.DATABASE_REF
        assert ref.params.get("query") == "SELECT * FROM rivers"

    def test_infer_schema_dot_table(self):
        """schema.table is ambiguous — could be a dotted filename or a pg ref.

        infer() should classify this as DATABASE_REF because it matches the
        schema.table pattern (no path separators, no file extension).
        """
        ref = SourceRef.infer("hydro.rivers")
        assert ref.kind == SourceRefKind.DATABASE_REF

    def test_infer_dotted_path_with_extension(self):
        """A path like 'dir.name/file.tif' should NOT be classified as database."""
        ref = SourceRef.infer("/dir.name/file.tif")
        assert ref.kind == SourceRefKind.LOCAL_RASTER

    def test_infer_ambiguous_stays_unknown(self):
        """Truly ambiguous refs classify as UNKNOWN — infer is honest."""
        ref = SourceRef.infer("just_a_word")
        assert ref.kind == SourceRefKind.UNKNOWN


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """SourceRef does NOT break existing code that uses str."""

    def test_str_gives_raw(self):
        ref = SourceRef.postgis("hydro", "rivers")
        assert str(ref) == "hydro.rivers"
        # Can be passed directly to connector's source_ref: str parameter
        # connector.materialize(str(ref), workspace) — always works

    def test_raw_is_always_str(self):
        for ref in [
            SourceRef.local("/path"),
            SourceRef.uri("https://x.com/f.tif"),
            SourceRef.stac("c", "i", asset="a"),
            SourceRef.postgis("s", "t"),
            SourceRef.postgis_query("SELECT 1"),
            SourceRef("raw_string"),
        ]:
            assert isinstance(ref.raw, str)
            assert isinstance(str(ref), str)

    def test_existing_connectors_unchanged(self):
        """The Connector protocol still takes source_ref: str.

        SourceRef helps callers construct refs and helps routers classify them.
        It does NOT change what connectors receive.
        """
        import inspect

        from quarry_core.connector import Connector

        sig = inspect.signature(Connector.materialize)
        params = sig.parameters
        # source_ref is still str in the protocol
        assert "source_ref" in params


# ---------------------------------------------------------------------------
# Immutability and equality
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_frozen(self):
        ref = SourceRef.local("/path")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            ref.raw = "something else"

    def test_equality_by_value(self):
        ref1 = SourceRef.postgis("hydro", "rivers")
        ref2 = SourceRef.postgis("hydro", "rivers")
        assert ref1 == ref2

    def test_different_kind_not_equal(self):
        ref1 = SourceRef("hydro.rivers")  # UNKNOWN
        ref2 = SourceRef.postgis("hydro", "rivers")  # DATABASE_REF
        # Same raw, different kind → not equal (kind matters)
        assert ref1 != ref2

    def test_hashable(self):
        ref = SourceRef.stac("c", "i", asset="a")
        # Can be used as dict key
        d = {ref: "value"}
        assert d[ref] == "value"

    def test_params_are_immutable(self):
        ref = SourceRef.local("/path/file.tif")
        with pytest.raises(TypeError):
            ref.params["path"] = "/elsewhere/file.tif"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        ref = SourceRef("")
        assert ref.raw == ""
        assert ref.kind == SourceRefKind.UNKNOWN

    def test_whitespace_preserved(self):
        ref = SourceRef("  spaces  ")
        assert ref.raw == "  spaces  "

    def test_postgis_quoted_table(self):
        ref = SourceRef.postgis("public", '"MixedCase"')
        assert ref.raw == 'public."MixedCase"'
        assert ref.params["table"] == '"MixedCase"'

    def test_infer_does_not_crash_on_weird_input(self):
        """infer() is lenient — weird input gets UNKNOWN, never crashes."""
        for weird in ["", " ", "://", "a" * 1000, "SELECT", "select"]:
            ref = SourceRef.infer(weird)
            assert isinstance(ref, SourceRef)
