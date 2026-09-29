from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).parents[2] / "docs" / "architecture" / "evidence" / "as-2.0.9"
SANDBOX = Path("/tmp/as_verify")


def _run(*args: str, expected: int = 0) -> str:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=SANDBOX,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == expected, f"{args}: {output}"
    return output


@pytest.mark.characterization
def test_agentscope_209_checkpoint_hitl_and_sop_behaviors() -> None:
    if not (EVIDENCE / "common.py").is_file():
        pytest.fail("AgentScope evidence scripts are missing")
    SANDBOX.mkdir(parents=True, exist_ok=True)
    for path in EVIDENCE.glob("*.py"):
        shutil.copy2(path, SANDBOX / path.name)

    _run("s7a.py", "crash", "1", expected=9)
    assert "next reply ok" in _run("s7a.py", "resume")
    _run("s7a2.py", "first", expected=9)
    _run("s7a2.py", "second", expected=9)
    assert "third EXEC" in _run("s7a2.py", "third")
    _run("s7b.py", "crash", expected=9)
    assert "resume EXEC" in _run("s7b.py", "resume")
    _run("fmt.py")

    _run("s2.py", "park")
    assert "replay rejected" in _run("s2.py", "decide")
    assert "re-asked" in _run("s2p.py", "one")
    assert "after second EXEC" in _run("s2p.py", "two")
    assert "next reply ok" in _run("intr.py")
    assert "after step2" in _run("sop_step.py")
