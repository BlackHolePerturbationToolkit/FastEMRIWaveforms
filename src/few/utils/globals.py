"""Global state of FEW: logger, configuration and file manager.

:class:`Globals` is a singleton created when this module is imported. Until it is
initialised the ``"few"`` logger buffers its records in memory and the configuration can
still be changed with the :class:`ConfigurationSetter` (:func:`get_config_setter`). The
first access to :func:`get_config` or :func:`get_file_manager` (e.g. the first data-file
lookup of a FEW module) initialises it: the configuration is read, the logger switches to
stdout / stderr handlers, and the :class:`~few.files.FileManager` is built. Compute
backends are not managed here: ``import few`` registers them with the ``gpubackendtools``
registry.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import typing

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
    """Metaclass that creates one instance per class and returns it on every later call."""

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
    """Flush target of the initial memory handler during :meth:`Globals.init`.

    Forwards the buffered records at or above ``level`` to the new stream handlers.

    Args:
        level: Minimum level of the forwarded records (the configured ``log_level``).
        *handlers: Handlers that receive the records (stdout and stderr).
    """

    handlers: typing.List[logging.Handler]
    level: int

    def __init__(self, level: int, *handlers):
        self.handlers = handlers
        self.level = level

    def handle(self, record):
        """Pass ``record`` to every handler if its level is at least ``level``."""
        if record.levelno >= self.level:
            for handler in self.handlers:
                handler.handle(record)


class ConfigurationSetter:
    """Set configuration options in Python before FEW initialises its globals.

    Obtained with :func:`few.get_config_setter`. Each setter converts its value with the
    entry's ``convert`` and returns the setter, so calls can be chained; the values are
    applied by :meth:`finalize` or, if it is not called, when the globals initialise. A
    value set after initialisation only logs a warning ("might not be taken into
    account").

    Args:
        finalizer: Callable receiving the dict of set values; the global setter's
            finalizer initialises the globals with them. Default ``None`` (no-op).

    Example:
        .. code-block:: python

            import few

            few.get_config_setter().set_log_level("info").add_file_extra_paths(
                "/tmp/few_data"
            ).finalize()
    """

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
        """Set ``enabled_backends`` (no FEW code reads it any more; see ``Configuration``)."""
        return self._convert_and_set("enabled_backends", backends)

    def set_log_level(self, level: typing.Union[str, int]) -> ConfigurationSetter:
        """Set ``log_level`` (a level name, case-insensitive, or an int)."""
        return self._convert_and_set("log_level", level)

    def set_storage_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Set ``file_storage_path`` (made absolute; must be an existing directory)."""
        return self._convert_and_set("file_storage_path", path)

    def set_log_format(self, format: str) -> ConfigurationSetter:
        """Set ``log_format`` (a :class:`logging.Formatter` format string)."""
        return self._convert_and_set("log_format", format)

    def set_file_registry_path(self, registry_path: os.PathLike) -> ConfigurationSetter:
        """Set ``file_registry_path`` (a custom ``registry.yml``, must exist)."""
        return self._convert_and_set("file_registry_path", registry_path)

    def set_file_download_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Set ``file_download_path`` (absolute, or relative to the storage path)."""
        return self._convert_and_set("file_download_path", path)

    def enable_file_download(self) -> ConfigurationSetter:
        """Authorize the file manager to download missing files."""
        return self._convert_and_set("file_allow_download", True)

    def disable_file_download(self) -> ConfigurationSetter:
        """Forbid the file manager to download missing files (offline use)."""
        return self._convert_and_set("file_allow_download", False)

    def set_file_integrity_check(self, when: str) -> ConfigurationSetter:
        """Define when integrity checks should be performed (never, once, always)"""
        return self._convert_and_set("file_integrity_check", when)

    def disable_file_tags(self, *tags: str) -> ConfigurationSetter:
        """Disable access to the files carrying any of ``tags`` (``file_disabled_tags``)."""
        return self._convert_and_set("file_disabled_tags", tags)

    def add_file_extra_paths(
        self, *paths: typing.List[os.PathLike]
    ) -> ConfigurationSetter:
        """Add search directories of the file manager (``file_extra_paths``, accumulated)."""
        return self._convert_and_set("file_extra_paths", paths)

    def _convert_and_set(self, label: str, value: typing.Any) -> ConfigurationSetter:
        """Convert ``value`` with the entry's ``convert``, store it and return ``self``.

        Logs a warning if the globals are already initialised.
        """
        if Globals().is_initialized:
            get_logger().warning(
                "FEW configurations is already initialized. Option {}={} might not be taken into account".format(
                    label, value
                )
            )
        self._args[label] = self._entries[label].convert(value)
        return self

    def get_args(self) -> dict[str, typing.Any]:
        """Return the dict of the options set so far (label to converted value)."""
        return self._args

    def finalize(self):
        """Call the finalizer with the set options (for the global setter: initialise FEW now).

        Raises:
            FewGlobalsInitializedTwice: For the global setter, if the globals are already
                initialised.
        """
        if self._finalizer is not None:
            self._finalizer(self.get_args())


class Globals(metaclass=Singleton):
    """The FEW global state (a singleton): logger, configuration, file manager.

    Created at import of this module with a pre-initialised logger (the ``"few"`` logger
    at ``DEBUG``, records buffered by a memory handler of capacity 1024) and a
    configuration setter. :meth:`init` (called by the first :attr:`config` /
    :attr:`file_manager` access, by the setter's ``finalize``, or explicitly with the CLI
    arguments of a command-line tool) builds the configuration, moves the logger to
    stdout (``<= INFO``) / stderr (``> INFO``) handlers with the configured level and
    format (flushing the buffered records at or above that level), and builds the
    :class:`~few.files.FileManager`. Attributes cannot be set from outside
    (:class:`FewGlobalsReadOnly`).
    """

    _logger: logging.Logger
    _initial_config: InitialConfigConsumer
    _config: Configuration
    _file_manager: FileManager
    _config_setter: ConfigurationSetter

    _to_initialize: bool

    def __init__(self):
        """Pre-initialise the logger and the configuration setter."""
        self._preinit_logger()
        self._preinit_setter()
        super().__setattr__("_to_initialize", True)

    def init(
        self,
        cli_args: typing.Optional[typing.Sequence[typing.Any]] = None,
        set_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Initialise the configuration, the logger and the file manager.

        Args:
            cli_args: Command-line arguments to read options from (e.g. ``sys.argv[1:]``).
                Default ``None``.
            set_args: Option values by label. Default ``None`` uses the values of the
                configuration setter.

        Raises:
            FewGlobalsInitializedTwice: If already initialised.
            few.utils.exceptions.ConfigurationError: If the configuration is invalid
                (subclasses, or an ``ExceptionGroup`` of several).
        """
        if not super().__getattribute__("_to_initialize"):
            raise FewGlobalsInitializedTwice("FEW globals are already initialized.")
        if set_args is None:
            config_setter = self.get_configuration_setter()
            set_args = config_setter.get_args()
        self._init_config(cli_args=cli_args, set_args=set_args)
        self._postconfig_logger()
        self._init_file_manager()

        super().__setattr__("_to_initialize", False)
        super().__setattr__("_config_setter", None)

        self.logger.debug("FEW globals initialized.")

    def reset(self):
        """Drop the configuration, file manager and logger state and pre-initialise again."""

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
        """The ``"few"`` logger (does not trigger initialisation)."""
        return super().__getattribute__("_logger")

    @property
    def config(self) -> Configuration:
        """The configuration; initialises the globals on first access."""
        if not self.is_initialized:
            self.init()
        return super().__getattribute__("_config")

    @property
    def file_manager(self) -> FileManager:
        """The file manager; initialises the globals on first access."""
        if not self.is_initialized:
            self.init()
        return super().__getattribute__("_file_manager")

    def get_configuration_setter(self, reset: bool = False) -> ConfigurationSetter:
        """Return the configuration setter.

        Args:
            reset: If the globals are already initialised, reset them first (see
                :meth:`reset`) instead of raising. Default ``False``.

        Returns:
            ConfigurationSetter: The setter, whose ``finalize`` initialises these globals.

        Raises:
            FewGlobalsInitializedTwice: If the globals are initialised and ``reset`` is
                ``False``.
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
        """Refuse attribute assignment (:class:`FewGlobalsReadOnly`)."""
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
        """Build the configuration: the two first passes, the config file, then ``Configuration``.

        The first :class:`~few.utils.config.InitialConfigConsumer` reads only the CLI
        arguments (``--ignore-env``), the second also the environment unless ignored; the
        config file is skipped with ``ignore_cfg``, else taken from ``config_file`` or
        :func:`~few.utils.config.detect_cfg_file`. Their leftovers and ``set_args`` build
        the :class:`~few.utils.config.Configuration`; every entry is logged at debug level.
        """
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
        """Build the :class:`~few.files.FileManager` from the configuration."""
        cfg: Configuration = super().__getattribute__("_config")
        file_manager = FileManager(cfg)
        super().__setattr__("_file_manager", file_manager)

    # TODO check: keep/remove/document -- no callers; BackendsManager undefined here (NameError)
    def _init_backends_manager(self):
        cfg: Configuration = super().__getattribute__("_config")
        backends_manager = BackendsManager(enabled_backends=cfg.enabled_backends)
        super().__setattr__("_backends_manager", backends_manager)


def get_logger() -> logging.Logger:
    """Return the ``"few"`` logger (buffered in memory until the globals initialise)."""
    return Globals().logger


def get_file_manager() -> FileManager:
    """Return the FEW :class:`~few.files.FileManager` (initialising the globals if needed)."""
    return Globals().file_manager


def get_config() -> Configuration:
    """Return the FEW :class:`~few.utils.config.Configuration` (initialising if needed)."""
    return Globals().config


def initialize(*cli_args):
    """Initialise the FEW globals (configuration, logger, file manager).

    Args:
        *cli_args: Forwarded positionally to :meth:`Globals.init`, so pass the command-line
            arguments as ONE list, e.g. ``initialize(["--log-level", "DEBUG"])``; a second
            positional argument would become ``set_args``.

    Raises:
        FewGlobalsInitializedTwice: If already initialised.
    """
    Globals().init(*cli_args)


def get_config_setter(reset: bool = False) -> ConfigurationSetter:
    """Return the configuration setter (see :meth:`Globals.get_configuration_setter`).

    Args:
        reset: Reset already-initialised globals instead of raising. Default ``False``.

    Returns:
        ConfigurationSetter: The global setter.
    """
    return Globals().get_configuration_setter(reset=reset)


def reset(quiet: bool = False):
    """Reset the global state (objects built before keep the old state).

    Args:
        quiet: Do not log the warning about existing objects. Default ``False``.
    """
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
