"""Pressure test: CLI `route` command.

Lane: adapter

Exercises:
  - `route` with local raster path (.tif) selects COG connector
  - `route` with ambiguous geojson falls back to local_file
  - `route` with remote S3 COG URI selects COG connector with scheme info
  - `route` with unsupported source returns exit code 2
  - `route` does not materialize (pure introspection)

Failure signals:
  - CLI returns non-zero for supported sources
  - CLI returns zero for unsupported sources
  - Selected connector name not in stdout
  - File created at synthetic path (should never happen)
"""

from quarry_cli.main import main


class TestRoute:
    def test_route_local_raster_selects_cog(self, capsys):
        """Pass a synthetic .tif path — file does not need to exist for routing."""
        rc = main(["route", "/tmp/nonexistent_dem.tif"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "kind: local_raster" in out
        assert "cog" in out
        assert "Selected" in out
        # Selected line should name cog
        for line in out.splitlines():
            if line.strip().startswith("cog") or "Selected" in line:
                if "cog" in line.lower() and "selected" not in line.lower():
                    continue  # This is a match line, not the Selected section
        # Verify Selected section contains cog
        lines = out.splitlines()
        in_selected = False
        selected_connector = None
        for line in lines:
            if line == "Selected":
                in_selected = True
                continue
            if in_selected and line.startswith("  ") and not line.startswith("   "):
                selected_connector = line.strip()
                break
        assert selected_connector == "cog"

    def test_route_ambiguous_geojson_falls_back_to_local_file(self, capsys):
        """Pass /tmp/x.geojson — ambiguous extension falls back to local_file."""
        rc = main(["route", "/tmp/x.geojson"])
        assert rc == 0

        out = capsys.readouterr().out
        # Selected line should name local_file
        lines = out.splitlines()
        in_selected = False
        selected_connector = None
        for line in lines:
            if line == "Selected":
                in_selected = True
                continue
            if in_selected and line.startswith("  ") and not line.startswith("   "):
                selected_connector = line.strip()
                break
        assert selected_connector == "local_file"

    def test_route_remote_cog_uri(self, capsys):
        """Pass s3://bucket/dem.tif — should select COG with scheme info."""
        rc = main(["route", "s3://bucket/dem.tif"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "kind: remote_uri" in out

        # Selected should be cog
        lines = out.splitlines()
        in_selected = False
        selected_connector = None
        for line in lines:
            if line == "Selected":
                in_selected = True
                continue
            if in_selected and line.startswith("  ") and not line.startswith("   "):
                selected_connector = line.strip()
                break
        assert selected_connector == "cog"

        # Matches block should show scheme= containing s3
        in_matches = False
        found_scheme = False
        for line in out.splitlines():
            if line == "Matches":
                in_matches = True
                continue
            if in_matches and line == "Selected":
                break
            if in_matches and "scheme=" in line and "s3" in line:
                found_scheme = True
                break
        assert found_scheme, "Expected scheme= to contain s3 in Matches block"

    def test_route_unsupported_source_returns_2(self, capsys):
        """Pass a URI with scheme not in _REMOTE_OBJECT_SCHEMES and unknown extension."""
        rc = main(["route", "ftp://bucket/file.unsupported"])
        assert rc == 2

        out = capsys.readouterr().out
        # Matches block contains (none)
        assert "(none)" in out
        # Selected line contains 'no connector'
        assert "no connector" in out.lower()

    def test_route_does_not_materialize(self, tmp_path, capsys):
        """Route should be pure introspection — no file I/O."""
        fake_path = tmp_path / "fake.tif"
        assert not fake_path.exists()

        rc = main(["route", str(fake_path)])
        assert rc == 0

        # Verify no file was created
        assert not fake_path.exists()

        # Verify introspection worked (kind shows local_raster)
        out = capsys.readouterr().out
        assert "kind: local_raster" in out
