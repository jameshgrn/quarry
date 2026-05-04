"""Policy-surface pressure tests.

Lane: check

These tests make repo doctrine executable:
- quarry-core stays zero-dependency
- canonical test commands include active pressure-test locations
- docs report the actual connector/operator ontology surface
- local scratch artifacts stay ignored
"""

from __future__ import annotations

import ast
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def _top_level_class_names(package_src: Path, suffix: str) -> set[str]:
    names: set[str] = set()
    for path in package_src.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith(suffix):
                names.add(node.name)
    return names


def test_quarry_core_has_no_runtime_dependencies():
    data = tomllib.loads((ROOT / "packages/quarry-core/pyproject.toml").read_text())

    assert data["project"]["dependencies"] == []


def test_canonical_test_gate_includes_package_connector_pressure_tests():
    pytest_ini = (ROOT / "pytest.ini").read_text()
    justfile = (ROOT / "justfile").read_text()

    assert "tests/pressure_test" in pytest_ini
    assert "packages/quarry-connectors/tests" in pytest_ini
    assert "tests/pressure_test/" in justfile
    assert "packages/quarry-connectors/tests/" in justfile


def test_docs_report_actual_connector_and_operator_surface():
    connector_count = len(
        _top_level_class_names(
            ROOT / "packages/quarry-connectors/src/quarry_connectors",
            "Connector",
        )
    )
    operator_count = len(
        _top_level_class_names(
            ROOT / "packages/quarry-operators/src/quarry_operators",
            "Operator",
        )
    )

    docs = {
        "AGENTS.md": (ROOT / "AGENTS.md").read_text(),
        "CLAUDE.md": (ROOT / "CLAUDE.md").read_text(),
        "README.md": (ROOT / "README.md").read_text(),
    }

    for name, text in docs.items():
        assert f"{connector_count} connector" in text, name
        assert f"{operator_count} operator" in text, name


def test_router_scope_is_not_overclaimed():
    cli_main = (ROOT / "packages/quarry-cli/src/quarry_cli/main.py").read_text()
    connector_router = (
        ROOT / "packages/quarry-connectors/src/quarry_connectors/router.py"
    ).read_text()
    router_tests = (ROOT / "tests/pressure_test/test_router_integration.py").read_text()
    docs = (ROOT / "AGENTS.md").read_text() + (ROOT / "CLAUDE.md").read_text()

    assert "default CLI router" in cli_main
    assert "build_default_router" in connector_router
    assert "default CLI connector types" in router_tests
    assert "extension/scheme/prefix" in docs
    assert "Semantic product connectors" in docs


def test_local_scratch_artifacts_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text()

    assert ".playwright-mcp/" in gitignore
    assert "explorer-*.png" in gitignore
