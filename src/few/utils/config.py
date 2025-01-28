"""Implementation of a centralized configuration management for FEW."""

import argparse
import dataclasses
import enum
import logging
import os
import pathlib
from typing import (
    Any,
    TypeVar,
    Generic,
    Optional,
    List,
    Union,
    Sequence,
    Mapping,
    Dict,
    Callable,
    Tuple,
)
from . import exceptions
from ..cutils.fast_selector import BackendSelectionMode


class ConfigSource(enum.Enum):
    """Enumeration of config option sources."""

    DEFAULT = "default"
    CFGFILE = "config_file"
    ENVVAR = "environment_var"
    CLIOPT = "command_line"


T = TypeVar("T")

ENVVAR_PREFIX: str = "FEW_"


@dataclasses.dataclass
class ConfigEntry(Generic[T]):
    """Description of a configuration entry."""

    label: str  # How the entry is referred to in Python code (config.get("label"))
    description: str  # Description of the configuration entry
    type: TypeVar = T  # Type of the value
    default: Optional[T] = None  # Default value
    cfg_entry: Optional[str] = None  # Name of the entry in a config file
    env_var: Optional[str] = (
        None  # Entry corresponding env var (uppercase, without FEW_ prefix)
    )
    cli_flags: Optional[Union[str, List[str]]] = (
        None  # Flag(s) for CLI arguments of this entry (e.g. "-f")
    )
    cli_kwargs: Optional[Dict[str, Any]] = (
        None  # Supplementary arguments to argparse add_argument method for CLI options
    )
    convert: Callable[[str], T] = None
    validate: Callable[[T], bool] = lambda _: True

    def __post_init__(self):
        if self.convert is None:
            self.convert = lambda v: self.type(v)


@dataclasses.dataclass
class ConfigItem(Generic[T]):
    """Actual configuration entry with its run-time properties (value, source, ...)"""

    value: T  # Item value
    source: ConfigSource  # Source of the item current value


class ConfigConsumer:
    """
    Base class for actual configs.

    This class handles building config values from default, file, environment and CLI options.
    It keeps a list of unused parameters for an other consumer.
    """

    _entries: Dict[str, ConfigEntry]
    _items: Dict[str, ConfigItem]

    _extra_file: Dict[str, str]
    _extra_env: Dict[str, str]
    _extra_cli: List[str]

    def __init__(
        self,
        config_entries: Sequence[ConfigEntry],
        config_file: Union[os.PathLike, Mapping[str, str], None] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
    ):
        """Initialize the items list and extra parameters."""

        # Build the entries mapping
        self._entries = {entry.label: entry for entry in config_entries}

        # Build default items
        default_items = ConfigConsumer._build_items_from_default(config_entries)

        # Retrieve option from sources
        opt_from_file = ConfigConsumer._get_from_config_file(config_file)
        opt_from_env = ConfigConsumer._get_from_envvars(env_vars)
        opt_from_cli = ConfigConsumer._get_from_cli_args(cli_args)

        # Consume options to build other item lists
        file_items, self._extra_file = ConfigConsumer._build_items_from_file(
            config_entries, opt_from_file
        )
        env_items, self._extra_env = ConfigConsumer._build_items_from_env(
            config_entries, opt_from_env
        )
        cli_items, self._extra_cli = ConfigConsumer._build_items_from_cli(
            config_entries, opt_from_cli
        )

        # Build final item mapping
        self._items = default_items | file_items | env_items | cli_items

        # Validate items:
        errors: List[Exception] = []
        for label, entry in self._entries.items():
            if label not in self._items:
                errors.append(
                    exceptions.ConfigurationMissing(
                        "Configuration entry '{}' is missing.".format(label)
                    )
                )
                continue
            item = self._items[label]
            if not entry.validate(item.value):
                errors.append(
                    exceptions.ConfigurationValidationError(
                        "Configuration entry '{}' has invalid value '{}'".format(
                            label, item.value
                        )
                    )
                )

        if errors:
            raise (
                errors[0]
                if len(errors) == 1
                else exceptions.ExceptionGroup(
                    "Invalid configuration due to previous issues.", errors
                )
            )

    def __getitem__(self, key: str) -> Any:
        """Get the value of a config entry."""
        return self._items[key].value

    def __getattr__(self, attr: str) -> Any:
        """Get the value of a config entry via attributes."""
        if attr in self._items:
            return self._items[attr].value
        return self.__getattribute__(attr)

    def get_item(self, key: str) -> Tuple[ConfigItem, ConfigEntry]:
        return self._items[key], self._entries[key]

    def get_items(self) -> List[Tuple[ConfigItem, ConfigEntry]]:
        return [(self._items[key], entry) for key, entry in self._entries.items()]

    def get_extras(self) -> Tuple[Mapping[str, str], Mapping[str, str], Sequence[str]]:
        """Return extra file, env and cli entries for other consumer."""
        return self._extra_file, self._extra_env, self._extra_cli

    @staticmethod
    def _get_from_config_file(
        config_file: Union[os.PathLike, Mapping[str, str], None],
    ) -> Dict[str, str]:
        """Read a config file and return its items as a dictionary."""
        if config_file is None:
            return {}
        if isinstance(config_file, Mapping):
            return {key: value for key, value in config_file.items()}
        import configparser

        config = configparser.ConfigParser()
        config.read(config_file)
        if "few" not in config:
            return {}
        few_section = config["few"]
        return {key: few_section[key] for key in few_section}

    @staticmethod
    def _get_from_envvars(env_vars: Optional[Mapping[str, str]]) -> Dict[str, str]:
        """Filter-out environment variables not matching a given prefix."""
        return (
            {
                key: value
                for key, value in env_vars.items()
                if key.startswith(ENVVAR_PREFIX)
            }
            if env_vars is not None
            else {}
        )

    @staticmethod
    def _get_from_cli_args(cli_args: Optional[Sequence[str]]) -> List[str]:
        """Build list of CLI arguments."""
        return [arg for arg in cli_args] if cli_args is not None else []

    @staticmethod
    def _compatibility_isinstance(obj, cls) -> bool:
        import sys
        import typing

        if (sys.version_info >= (3, 10)) or (typing.get_origin(cls) is None):
            return isinstance(obj, cls)

        if typing.get_origin(cls) is typing.Union:
            for arg in typing.get_args(cls):
                if ConfigConsumer._compatibility_isinstance(obj, arg):
                    return True
            return False

        raise NotImplementedError(
            "compatiblity wrapper for isinstance on Python 3.9 does not support given type."
        )

    @staticmethod
    def _build_items_from_default(
        config_entries: Sequence[ConfigEntry],
    ) -> Dict[str, ConfigItem]:
        """Build a list of ConfigItem built from default-valued ConfigEntries."""
        return {
            entry.label: ConfigItem(value=entry.default, source=ConfigSource.DEFAULT)
            for entry in config_entries
            if ConfigConsumer._compatibility_isinstance(entry.default, entry.type)
        }

    @staticmethod
    def _build_items_from_file(
        config_entries: Sequence[ConfigEntry], opt_from_file: Mapping[str, str]
    ) -> Tuple[Dict[str, ConfigItem], Dict[str, str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        extras_from_file = {**opt_from_file}
        items_from_file = {
            entry.label: ConfigItem(
                value=entry.convert(extras_from_file.pop(entry.cfg_entry)),
                source=ConfigSource.CFGFILE,
            )
            for entry in config_entries
            if entry.cfg_entry in extras_from_file
        }

        return items_from_file, extras_from_file

    @staticmethod
    def _build_items_from_env(
        config_entries: Sequence[ConfigEntry], opt_from_env: Mapping[str, str]
    ) -> Tuple[Dict[str, ConfigItem], Dict[str, str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        extras_from_env = {**opt_from_env}
        items_from_env = {
            entry.label: ConfigItem(
                value=entry.convert(extras_from_env.pop(ENVVAR_PREFIX + entry.env_var)),
                source=ConfigSource.ENVVAR,
            )
            for entry in config_entries
            if entry.env_var is not None
            and ENVVAR_PREFIX + entry.env_var in extras_from_env
        }

        return items_from_env, extras_from_env

    @staticmethod
    def _build_parser(config_entries: Sequence[ConfigEntry]) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for config_entry in config_entries:
            if config_entry.cli_flags:
                cli_options = {
                    "help": config_entry.description,
                    "dest": config_entry.label,
                }
                if config_entry.cli_kwargs is not None:
                    cli_options = cli_options | config_entry.cli_kwargs
                flags = (
                    [config_entry.cli_flags]
                    if isinstance(config_entry.cli_flags, str)
                    else config_entry.cli_flags
                )
                parser.add_argument(*flags, **cli_options)
        return parser

    @staticmethod
    def _build_items_from_cli(
        config_entries: Sequence[ConfigEntry], opt_from_cli: Sequence[str]
    ) -> Tuple[Dict[str, ConfigItem], List[str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        parser = ConfigConsumer._build_parser(config_entries)
        parsed_options, extras_from_cli = parser.parse_known_args(opt_from_cli)

        items_from_cli = {
            option_label: ConfigItem(value=option_value, source=ConfigSource.CLIOPT)
            for option_label, option_value in vars(parsed_options).items()
            if option_value is not None
        }
        return items_from_cli, extras_from_cli


def userstr_to_bool(user_str: str) -> Optional[bool]:
    """Convert a yes/no, on/off or true/false to bool."""
    if user_str.lower().startswith(("y", "t", "on")):
        return True
    if user_str.lower().startswith("n", "f", "off"):
        return False
    return None


class InitialConfigConsumer(ConfigConsumer):
    """
    Class implementing first-pass config consumer.

    On first pass, we only detect if there are CLI arguments which disable
    config file or environment variables.
    """

    ignore_cfg: bool
    ignore_env: bool
    config_file: Optional[pathlib.Path]

    def __init__(
        self,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
    ):
        if cli_args is None:
            import sys

            cli_args = sys.argv[1:]

        config_entries = [
            ConfigEntry(
                label="ignore_cfg",
                description="Whether to ignore config file options",
                type=bool,
                default=False,
                cli_flags="--ignore-config-file",
                cli_kwargs={"action": "store_const", "const": True},
                convert=userstr_to_bool,
                validate=lambda x: isinstance(x, bool),
            ),
            ConfigEntry(
                label="ignore_env",
                description="Whether to ignore environment variables",
                type=bool,
                default=False,
                cli_flags="--ignore-env",
                cli_kwargs={"action": "store_const", "const": True},
                convert=userstr_to_bool,
                validate=lambda x: isinstance(x, bool),
            ),
            ConfigEntry(
                label="config_file",
                description="Path to FEW configuration file",
                type=Optional[pathlib.Path],
                default=None,
                cli_flags=["-C", "--config-file"],
                env_var="CONFIG_FILE",
                convert=lambda p: None if p is None else pathlib.Path(p),
                validate=lambda p: True if p is None else p.is_file(),
            ),
        ]

        super().__init__(
            config_entries, config_file=None, env_vars=env_vars, cli_args=cli_args
        )


class CompleteConfigConsumer(ConfigConsumer):
    """
    Class implementing FEW complete configuration for the library.
    """

    fast_backend: BackendSelectionMode
    log_level: int
    log_format: str
    file_registry_path: Optional[pathlib.Path]
    file_storage_path: Optional[pathlib.Path]
    file_download_path: Optional[pathlib.Path]

    def __init__(
        self,
        config_file: Union[os.PathLike, Mapping[str, str], None] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
    ):
        config_entries = [
            ConfigEntry(
                label="fast_backend",
                description="Fast backend selection mode",
                type=BackendSelectionMode,
                default=BackendSelectionMode.LAZY,
                cli_flags="--fast-backend",
                cli_kwargs={
                    "type": BackendSelectionMode,
                },
                env_var="FAST_BACKEND",
                cfg_entry="fast-backend",
            ),
            ConfigEntry(
                label="log_level",
                description="Application log level",
                type=int,
                default=logging.WARN,
                cli_flags=["--log-level"],
                env_var="LOG_LEVEL",
                cfg_entry="log-level",
                convert=CompleteConfigConsumer._str_to_logging_level,
            ),
            ConfigEntry(
                label="log_format",
                description="Application log format",
                type=Optional[str],
                default=None,
                cli_flags=["--log-format"],
                env_var="LOG_FORMAT",
                cfg_entry="log-format",
                convert=lambda input: input,
                validate=lambda input: input is None or isinstance(input, str),
            ),
            ConfigEntry(
                label="file_registry_path",
                description="File Registry path",
                type=Optional[pathlib.Path],
                default=None,
                cli_flags=["--file-registry"],
                env_var="FILE_REGISTRY",
                cfg_entry="file-registry",
                convert=lambda p: None if p is None else pathlib.Path(p),
                validate=lambda p: True if p is None else p.is_file(),
            ),
            ConfigEntry(
                label="file_storage_path",
                description="File Manager storage directory (absolute or relative to current working directory, must exist)",
                type=Optional[pathlib.Path],
                default=None,
                cli_flags=["--storage-dir"],
                env_var="FILE_STORAGE_DIR",
                cfg_entry="file-storage-dir",
                convert=lambda p: None if p is None else pathlib.Path(p).absolute(),
                validate=lambda p: True if p is None else p.is_dir(),
            ),
            ConfigEntry(
                label="file_download_path",
                description="File Manager download directory (absolute, or relative to storage_path)",
                type=Optional[pathlib.Path],
                default=None,
                cli_flags=["--download-dir"],
                env_var="FILE_DOWNLOAD_DIR",
                cfg_entry="file-download-dir",
                convert=lambda p: None if p is None else pathlib.Path(p),
                validate=lambda p: True
                if p is None
                else (p.is_dir() if p.is_absolute() else True),
            ),
        ]

        super().__init__(
            config_entries,
            config_file=config_file,
            env_vars=env_vars,
            cli_args=cli_args,
        )

        # Post-init task: read -v and -q options
        self._handle_verbosity()

    @staticmethod
    def _str_to_logging_level(input: str) -> int:
        as_int_level = logging.getLevelName(input.upper())
        if isinstance(as_int_level, int):
            return as_int_level
        raise exceptions.ConfigurationValidationError(
            "'{}' is not a valid log level.".format(input)
        )

    def _apply_verbosity(self, level):
        self._items["log_level"].value = level

    def _handle_verbosity(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v", "--verbose", dest="verbose_count", action="count", default=0
        )
        parser.add_argument("-Q", "--quiet", dest="quiet", action="store_true")
        parsed_options, self._extra_cli = parser.parse_known_args(self._extra_cli)

        if parsed_options.quiet:
            self._apply_verbosity(logging.CRITICAL)
        else:
            new_level = self.log_level - parsed_options.verbose_count * 10
            self._apply_verbosity(
                new_level if new_level > logging.DEBUG else logging.DEBUG
            )


def detect_cfg_file() -> Optional[pathlib.Path]:
    """Test common path locations for config and return highest-priority existing one (if any)."""
    import platformdirs
    from .. import __version_tuple__

    LOCATIONS = [
        pathlib.Path.cwd() / "few.ini",
        platformdirs.user_config_path() / "few.ini",
        platformdirs.site_config_path()
        / "few"
        / "v{}.{}".format(__version_tuple__[0], __version_tuple__[1])
        / "few.ini",
    ]
    for location in LOCATIONS:
        if location.is_file():
            return location
    return None
