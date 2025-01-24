"""Definition of global states (logger, config, file manager, fast backend, ...)"""

import logging
import logging.handlers
import typing

from .exceptions import FewException
from .config import (
    InitialConfigConsumer,
    CompleteConfigConsumer as Configuration,
    detect_cfg_file,
)
from ..files import FileManager
from ..cutils.fast import load_backend


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FewGlobalsInitializedTwice(FewException):
    """Exception raised if globals are initialized multiple times."""


class FewGlobalsReadOnly(FewException):
    """Exception raised hen trying to modify the global structure."""


class MultiHandlerTarget:
    """Helper class to transition logger from memory-buffer to stream handler during globals init"""

    handlers: typing.List[logging.Handler]

    def __init__(self, *handlers):
        self.handlers = handlers

    def handle(self, record):
        for handler in self.handlers:
            handler.handle(record)



class Globals(metaclass=Singleton):
    _logger: logging.Logger
    _initial_config: InitialConfigConsumer
    _config: Configuration
    _file_manager: FileManager

    _to_initialize: bool

    def __init__(self):
        """Initiliaze the logger"""
        self._preinit_logger()
        super().__setattr__("_to_initialize", True)

    def init(self, *cli_args):
        """Initialize config, file manager and logger with optional CLI arguments."""
        if not super().__getattribute__("_to_initialize"):
            raise FewGlobalsInitializedTwice("FEW globals are already initialized.")
        self._init_config(*cli_args)
        self._postconfig_logger()
        self._init_file_manager()
        self._init_fast_backend()

        super().__setattr__("_to_initialize", False)

    @property
    def logger(self) -> logging.Logger:
        return super().__getattribute__("_logger")

    @property
    def config(self) -> Configuration:
        if super().__getattribute__("_to_initialize"):
            self.init()
        return super().__getattribute__("_config")

    @property
    def file_manager(self) -> FileManager:
        if super().__getattribute__("_to_initialize"):
            self.init()
        return super().__getattribute__("_file_manager")

    def __setattr__(self, name, value):
        raise FewGlobalsReadOnly("Cannot set attribute on Globals structure.")

    def _preinit_logger(self):
        """Pre-initialize logger."""
        logger = logging.getLogger("few")
        logger.setLevel(logging.DEBUG)
        INITIAL_CAPACITY = 1024  # Log up to 1024 messages until globals are initialized
        handler = logging.handlers.MemoryHandler(capacity=INITIAL_CAPACITY)
        handler.set_name("_few_initial_handler")
        logger.addHandler(handler)
        super().__setattr__("_logger", logger)

    def _init_config(self, *cli_args):
        """Initialize configurations"""
        import os

        ignores_cfg = InitialConfigConsumer(
            cli_args=cli_args
        )  # Read only CLI args (ignores)

        file_cfg = InitialConfigConsumer(
            env_vars=None if ignores_cfg.ignore_env else os.environ, cli_args=cli_args
        )  # Read CLI args (and env if not ignored)

        cfg_file = (
            None
            if file_cfg.ignore_cfg
            else file_cfg.config_file
            if file_cfg.config_file is not None
            else detect_cfg_file()
        )

        _, extra_env_vars, extra_cli_args = file_cfg.get_extras()

        config = Configuration(
            config_file=cfg_file, env_vars=extra_env_vars, cli_args=extra_cli_args
        )
        super().__setattr__("_initial_config", file_cfg)
        super().__setattr__("_config", config)

    def _postconfig_logger(self):
        """Initialize logger after config is initialized."""
        import sys

        logger: logging.Logger = super().__getattribute__("_logger")
        cfg: Configuration = super().__getattribute__("_config")

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

        for handler in logger.handlers:
            if handler.get_name() == "_few_initial_handler":
                assert isinstance(handler, logging.handlers.MemoryHandler)
                handler.setTarget(MultiHandlerTarget(stdout_handler, stderr_handler))
                handler.close()
                break

        logger.handlers.clear()
        logger.setLevel(cfg.log_level)
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    def _init_file_manager(self):
        cfg: Configuration = super().__getattribute__("_config")
        file_manager = FileManager(cfg)
        super().__setattr__("_file_manager", file_manager)

    def _init_fast_backend(self):
        cfg: Configuration = super().__getattribute__("_config")
        load_backend(cfg.fast_backend)

def get_logger() -> logging.Logger:
    """Get FEW logger"""
    return Globals().logger

def get_file_manager() -> FileManager:
    """Get FEW File Manager"""
    return Globals().file_manager

def get_config() -> Configuration:
    """Get FEW configuration"""
    return Globals().config

def initialize(*cli_args):
    """Initialize FEW configuration, logger and file manager with CLI arguments"""
    Globals().init(*cli_args)

__all__ = ["Globals", "get_logger", "get_file_manager", "get_config", "initialize"]
