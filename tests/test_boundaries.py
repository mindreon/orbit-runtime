"""Source-level boundaries import-linter cannot express.

import-linter refuses to name a subpackage of an external distribution, so
the ban on ``agentscope.app`` (Agent Service) is checked by reading our own
imports. The other two contracts stay in pyproject.toml.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    found: set[str] = set()
    for item in path.rglob("*.py"):
        tree = ast.parse(item.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
    return found


def test_worker_does_not_import_agent_service() -> None:
    modules = _imported_modules(ROOT / "packages" / "orbit_worker" / "src")
    assert not any(name == "agentscope.app" or name.startswith("agentscope.app.") for name in modules)


def test_orch_and_contracts_do_not_import_agentscope() -> None:
    for package in ("orbit_orch", "orbit_contracts"):
        modules = _imported_modules(ROOT / "packages" / package / "src")
        assert not any(name == "agentscope" or name.startswith("agentscope.") for name in modules)
