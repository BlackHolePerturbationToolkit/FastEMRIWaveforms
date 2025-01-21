"""Implementation of FEW logger"""

import logging
import logging.handlers
import sys

LOGGER = logging.getLogger("few")
LOGGER.setLevel(logging.DEBUG)

INITIAL_CAPACITY = 1024

class MultiHandlerTarget():
    def __init__(self, *handlers):
        self.handlers = handlers

    def handle(self, record):
        for handler in self.handlers:
            handler.handle(record)

_initial_handler = logging.handlers.MemoryHandler(capacity=INITIAL_CAPACITY)

LOGGER.addHandler(_initial_handler)

def postconfig_install_handlers():
    from few import cfg


    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(cfg.log_level)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(cfg.log_level)
    stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

    if cfg.log_format is not None:
        formatter = logging.Formatter(fmt=cfg.log_format)
        stdout_handler.setFormatter(formatter)
        stderr_handler.setFormatter(formatter)

    _initial_handler.setTarget(MultiHandlerTarget(stdout_handler, stderr_handler))
    _initial_handler.close()

    LOGGER.setLevel(cfg.log_level)
    LOGGER.removeHandler(_initial_handler)
    LOGGER.addHandler(stdout_handler)
    LOGGER.addHandler(stderr_handler)
