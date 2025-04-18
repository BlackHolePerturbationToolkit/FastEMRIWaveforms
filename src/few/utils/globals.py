"""Definition of global states (logger, config, file manager, fast backend, ...)"""

from __future__ import annotations

import logging
import logging.handlers
import os
import typing

from ..cutils import Backend, BackendsManager
from ..files import FileManager
from .config import (
    ConfigConsumer,
    ConfigEntry,
    ConfigSource,
    Configuration,
    InitialConfigConsumer,
    detect_cfg_file,
)
from .exceptions import FewException


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FewGlobalsInitializedTwice(FewException):
    """Exception raised if globals are initialized multiple times."""


class FewGlobalsReadOnly(FewException):
    """Exception raised when trying to modify the global structure."""


class MultiHandlerTarget:
    """Helper class to transition logger from memory-buffer to stream handler during globals init"""

    handlers: typing.List[logging.Handler]
    level: int

    def __init__(self, level: int, *handlers):
        self.handlers = handlers
        self.level = level

    def handle(self, record):
        if record.levelno >= self.level:
            for handler in self.handlers:
                handler.handle(record)


class ConfigurationSetter:
    """Helper class to define configuration options."""

    _args: typing.Dict[str, typing.Any]
    _entries: typing.Dict[str, ConfigEntry]
    _finalizer: typing.Optional[typing.Callable[[typing.Dict[str, typing.Any]], None]]

    def __init__(
        self,
        finalizer: typing.Optional[
            typing.Callable[[typing.Dict[str, typing.Any]], None]
        ] = None,
    ):
        self._args = {}
        self._entries = {entry.label: entry for entry in Configuration.config_entries()}
        self._finalizer = finalizer

    def enable_backends(self, *backends: str) -> ConfigurationSetter:
        """Enable one or multiple backends"""
        return self._convert_and_set("enabled_backends", backends)

    def set_log_level(self, level: typing.Union[str, int]) -> ConfigurationSetter:
        """Set a specific log level"""
        return self._convert_and_set("log_level", level)

    def set_storage_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Modify the storage path"""
        return self._convert_and_set("file_storage_path", path)

    def set_log_format(self, format: str) -> ConfigurationSetter:
        """Change the log format"""
        return self._convert_and_set("log_format", format)

    def set_file_registry_path(self, registry_path: os.PathLike) -> ConfigurationSetter:
        """Set the file registry to use"""
        return self._convert_and_set("file_registry_path", registry_path)

    def set_file_download_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Set the download path"""
        return self._convert_and_set("file_download_path", path)

    def enable_file_download(self) -> ConfigurationSetter:
        """Authorize the file manager to download missing files"""
        return self._convert_and_set("file_allow_download", True)

    def disable_file_download(self) -> ConfigurationSetter:
        """Authorize the file manager to download missing files"""
        return self._convert_and_set("file_allow_download", False)

    def set_file_integrity_check(self, when: str) -> ConfigurationSetter:
        """Define when integrity checks should be performed (never, once, always)"""
        return self._convert_and_set("file_integrity_check", when)

    def disable_file_tags(self, *tags: str) -> ConfigurationSetter:
        """Disable files associated to given tags"""
        return self._convert_and_set("file_disabled_tags", tags)

    def add_file_extra_paths(
        self, *paths: typing.List[os.PathLike]
    ) -> ConfigurationSetter:
        """Add supplementary research paths to file manager"""
        return self._convert_and_set("file_extra_paths", paths)

    def _convert_and_set(self, label: str, value: typing.Any) -> ConfigurationSetter:
        if Globals().is_initialized:
            get_logger().warning(
                "FEW configurations is already initialized. Option {}={} might not be taken into account".format(
                    label, value
                )
            )
        self._args[label] = self._entries[label].convert(value)
        return self

    def get_args(self) -> dict[str, typing.Any]:
        """Get a dictionnary of set options."""
        return self._args

    def finalize(self):
        """Finalize FEW initialization with specified parameters."""
        if self._finalizer is not None:
            self._finalizer(self.get_args())


class Globals(metaclass=Singleton):
    _logger: logging.Logger
    _initial_config: InitialConfigConsumer
    _config: Configuration
    _file_manager: FileManager
    _config_setter: ConfigurationSetter
    _backends_manager: BackendsManager

    _to_initialize: bool

    def __init__(self):
        """Initiliaze the logger"""
        self._preinit_logger()
        self._preinit_setter()
        super().__setattr__("_to_initialize", True)

    def init(
        self,
        cli_args: typing.Optional[typing.Sequence[typing.Any]] = None,
        set_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Initialize config, file manager and logger with optional CLI arguments."""
        if not super().__getattribute__("_to_initialize"):
            raise FewGlobalsInitializedTwice("FEW globals are already initialized.")
        if set_args is None:
            config_setter = self.get_configuration_setter()
            set_args = config_setter.get_args()
        self._init_config(cli_args=cli_args, set_args=set_args)
        self._postconfig_logger()
        self._init_file_manager()
        self._init_backends_manager()

        super().__setattr__("_to_initialize", False)
        super().__setattr__("_config_setter", None)

        self.logger.debug("FEW globals initialized.")

    def reset(self):
        """Reset the global structure."""

        # Remove attributes existing only in initialized globals
        if self.is_initialized:
            super().__delattr__("_initial_config")
            super().__delattr__("_config")
            super().__delattr__("_file_manager")

        # Remove attributes always existing
        super().__delattr__("_logger")
        super().__delattr__("_config_setter")
        super().__delattr__("_to_initialize")

        # Reinitiliaze the structure
        Globals.__init__(self)

    @property
    def is_initialized(self) -> bool:
        """Whether global properties are initialized."""
        return not super().__getattribute__("_to_initialize")

    @property
    def logger(self) -> logging.Logger:
        return super().__getattribute__("_logger")

    @property
    def config(self) -> Configuration:
        if not self.is_initialized:
            self.init()
        return super().__getattribute__("_config")

    @property
    def file_manager(self) -> FileManager:
        if not self.is_initialized:
            self.init()
        return super().__getattribute__("_file_manager")

    @property
    def backends_manager(self) -> BackendsManager:
        if super().__getattribute__("_to_initialize"):
            self.init()
        return super().__getattribute__("_backends_manager")

    def get_configuration_setter(self, reset: bool = False) -> ConfigurationSetter:
        """
        Access a configuration setter.

        raises FewGlobalsInitializedTwice if globals are already initialized and reset is False.
        """
        if self.is_initialized:
            if reset:
                self.reset()
            else:
                raise FewGlobalsInitializedTwice(
                    "FEW globals are already initialized. Cannot access a setter."
                )
        return super().__getattribute__("_config_setter")

    def __setattr__(self, name, value):
        raise FewGlobalsReadOnly("Cannot set attribute on Globals structure.")

    def _preinit_logger(self):
        """Pre-initialize logger."""
        logger = logging.getLogger("few")
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            logger.removeHandler(handler)

        INITIAL_CAPACITY = 1024  # Log up to 1024 messages until globals are initialized
        handler = logging.handlers.MemoryHandler(capacity=INITIAL_CAPACITY)
        handler.set_name("_few_initial_handler")
        logger.addHandler(handler)
        super().__setattr__("_logger", logger)

    def _preinit_setter(self):
        """Initialize the configuration setter."""
        setter = ConfigurationSetter(finalizer=lambda args: self.init(set_args=args))
        super().__setattr__("_config_setter", setter)

    def _init_config(self, cli_args, set_args):
        """Initialize configurations"""
        import os

        ignores_cfg = InitialConfigConsumer(
            cli_args=cli_args
        )  # Read only CLI args (ignores)

        logger = get_logger()
        if ignores_cfg.ignore_env:
            logger.debug(
                "ConfigInitialization: ignoring environment as requested by command-line options"
            )

        file_cfg = InitialConfigConsumer(
            env_vars=None if ignores_cfg.ignore_env else os.environ, cli_args=cli_args
        )  # Read CLI args (and env if not ignored)

        if file_cfg.ignore_cfg:
            logger.debug(
                "ConfigInitialization: ignoring configuration file as requested by {} options".format(
                    "command-line"
                    if file_cfg.get_item("ignore_cfg")[0].source == ConfigSource.CLIOPT
                    else "environment"
                )
            )

        cfg_file = (
            None
            if file_cfg.ignore_cfg
            else file_cfg.config_file
            if file_cfg.config_file is not None
            else detect_cfg_file()
        )

        if cfg_file is not None:
            logger.debug(
                "ConfigInitialization: using configuration file '{}'.".format(cfg_file)
            )

        _, extra_env_vars, extra_cli_args = file_cfg.get_extras()

        config = Configuration(
            config_file=cfg_file,
            env_vars=extra_env_vars,
            cli_args=extra_cli_args,
            set_args=set_args,
        )
        super().__setattr__("_initial_config", file_cfg)
        super().__setattr__("_config", config)

        logger.debug("ConfigInitialization: final configuration entries are")
        self._log_config(file_cfg)
        self._log_config(config)

    def _log_config(self, config: ConfigConsumer, log_level=logging.DEBUG):
        """Print configuration options in logs"""
        logger = get_logger()
        for item, entry in config.get_items():
            logger.log(
                level=log_level,
                msg=f" {entry.label}={item.value} (from: {str(item.source)})",
            )

    def _postconfig_logger(self):
        """Initialize logger after config is initialized."""
        import sys

        logger: logging.Logger = super().__getattribute__("_logger")
        cfg: Configuration = super().__getattribute__("_config")

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

        if cfg.log_format is not None:
            formatter = logging.Formatter(fmt=cfg.log_format)
            stdout_handler.setFormatter(formatter)
            stderr_handler.setFormatter(formatter)

        for handler in logger.handlers:
            if handler.get_name() == "_few_initial_handler":
                assert isinstance(handler, logging.handlers.MemoryHandler)
                handler.setTarget(
                    MultiHandlerTarget(cfg.log_level, stdout_handler, stderr_handler)
                )
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

    def _init_backends_manager(self):
        cfg: Configuration = super().__getattribute__("_config")
        backends_manager = BackendsManager(enabled_backends=cfg.enabled_backends)
        super().__setattr__("_backends_manager", backends_manager)


def get_logger() -> logging.Logger:
    """Get FEW logger"""
    return Globals().logger


def get_file_manager() -> FileManager:
    """Get FEW File Manager"""
    return Globals().file_manager


def get_config() -> Configuration:
    """Get FEW configuration"""
    return Globals().config


def get_backend(backend_name: str) -> Backend:
    """
    Get a backend by its name.

    If the backend name is "cuda", return a CUDA backend if any available.
    If the backend name is "gpu", return a GPU backend if any available.
    """
    if backend_name == "cuda":
        return get_first_backend(["cuda12x", "cuda11x"])
    if backend_name == "gpu":
        return get_backend("cuda")
    return Globals().backends_manager.get_backend(backend_name=backend_name)


def has_backend(backend_name: str) -> bool:
    """
    Test if a backend is available.

    If the backend name is "cuda", return true if any of "cuda11x" or "cuda12x" is available.
    If the backend name is "gpu", return true if any GPU backend is available.
    """
    if backend_name == "cuda":
        return has_backend("cuda11x") or has_backend("cuda12x")
    if backend_name == "gpu":
        return has_backend("cuda")
    return Globals().backends_manager.has_backend(backend_name=backend_name)


def get_first_backend(backend_names: typing.Sequence[str]) -> Backend:
    """Get the first available backend from a list of backend names"""
    return Globals().backends_manager.get_first_backend(backend_names)


def initialize(*cli_args):
    """Initialize FEW configuration, logger and file manager with CLI arguments"""
    Globals().init(*cli_args)


def get_config_setter(reset: bool = False) -> ConfigurationSetter:
    """Get a configuration setter."""
    return Globals().get_configuration_setter(reset=reset)


def reset(quiet: bool = False):
    """Reset global states."""
    if not quiet:
        get_logger().warning(
            "FEW globals are about to be reset. Objects built up until now should be deleted to prevent unexpected side-effects."
        )
    Globals().reset()


# Initialize the globals singleton when first importing this file
Globals()

__all__ = [
    "Globals",
    "ConfigurationSetter",
    "get_logger",
    "get_file_manager",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_first_backend",
    "has_backend",
    "initialize",
]
