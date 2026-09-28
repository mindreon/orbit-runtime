"""Write JSON Schema for every contract model.

orbit-control generates Go types from these files. The package stays on
pydantic only, so this module does not import Temporal or AgentScope.
The v2 room models go to schema/*.json; contract v3 goes to schema/v3/.
"""

import json
from pathlib import Path

from orbit_contracts.models import contract_models
from orbit_contracts.v3.export import export as export_v3


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
    export_v3(root / "schema" / "v3")


if __name__ == "__main__":
    main()
