"""
logging_utils.py
================
Structured JSON logging configuration used across every agent.

Using `structlog` gives us machine-parseable logs that flow cleanly into
Google Cloud Logging, with per-request correlation via bound context vars.
"""

from __future__ import annotations

import logging
import sys

import structlog

from config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotently configure structlog + stdlib logging for the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Render as JSON for Cloud Logging ingestion.
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, ensuring configuration has run."""
    configure_logging()
    return structlog.get_logger(name)
