from __future__ import annotations

import json
import logging

import pytest
import structlog
from orbit_orch.logs import LogSettings, configure_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def test_json_format_carries_fields_context_and_foreign_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(LogSettings(format="json"))
    structlog.contextvars.bind_contextvars(task_id="task_1", attempt_id="att_1")
    structlog.get_logger("orbit_worker.test").warning("chat model", mode="real", model="m")
    logging.getLogger("orbit_worker.test").warning("mcp connector %s skipped", "c1")
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert lines[0]["event"] == "chat model"
    assert lines[0]["mode"] == "real"
    assert lines[0]["task_id"] == "task_1"
    assert lines[0]["level"] == "warning"
    assert lines[1]["event"] == "mcp connector c1 skipped"
    assert lines[1]["attempt_id"] == "att_1"


def test_console_format_is_readable_and_third_party_info_stays_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(LogSettings())
    logging.getLogger("httpx").info("GET https://example.test")
    structlog.get_logger("orbit_worker.test").warning("state store", kind="memory")
    err = capsys.readouterr().err
    assert "GET https://example.test" not in err
    assert "state store" in err and "kind=memory" in err


def test_an_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError):
        LogSettings(format="xml")  # type: ignore[arg-type]
