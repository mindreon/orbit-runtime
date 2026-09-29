"""Structured logging for both processes.

Every record, ours and the libraries', goes through one structlog pipeline on stderr. ``ORBIT_LOG_FORMAT=json``
prints one JSON object per line (as orbit-control does); anything else prints readable lines. Context bound with
``structlog.contextvars`` (tenant, task, attempt) is on every record of the running activity.
"""

import logging
import sys
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    format: Literal["console", "json"] = Field("console", validation_alias="ORBIT_LOG_FORMAT")
    # WARNING keeps third-party transport chatter (HTTP clients, the Temporal core) out unless asked for.
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "WARNING", validation_alias="ORBIT_LOG_LEVEL"
    )


def configure_logging(settings: LogSettings | None = None) -> None:
    settings = settings or LogSettings()
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.level)
    # Our own loggers always show their warnings and info, whatever the root level.
    logging.getLogger("orbit_orch").setLevel(logging.INFO)
    logging.getLogger("orbit_worker").setLevel(logging.INFO)
