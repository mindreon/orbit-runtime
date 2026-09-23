"""Write JSON Schema for every contract model.

orbit-control generates Go types from these files. The package stays on
pydantic only, so this module does not import Temporal or AgentScope.
"""

import json
from pathlib import Path

from orbit_contracts.models import contract_models


def export_schemas(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in contract_models():
        path = directory / f"{model.__name__}.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    export_schemas(root / "schema")


if __name__ == "__main__":
    main()
