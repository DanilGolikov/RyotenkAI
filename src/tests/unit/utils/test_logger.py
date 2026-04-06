from __future__ import annotations

import logging
import sys

from src.utils.logger import setup_logger


def test_setup_logger_uses_stderr_for_console_handler() -> None:
    logger = setup_logger("ryotenkai.test_logger_stream", level=logging.INFO, log_file=None, use_color=False)

    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]

    assert len(stream_handlers) == 1
    assert stream_handlers[0].stream is sys.__stderr__
