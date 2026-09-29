"""Write the JSON Schema bundle of contract v3 to schema/v3/.

orbit-control generates its Go types from that bundle. The package stays on pydantic only, so this module does not
import Temporal or AgentScope.
"""

from pathlib import Path

from orbit_contracts.v3.export import export as export_v3


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    export_v3(root / "schema" / "v3")


if __name__ == "__main__":
    main()
