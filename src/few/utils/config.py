"""Implementation of a centralized configuration management for FEW."""

from __future__ import annotations

import abc
import argparse
import dataclasses
import enum
import logging
import os
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from ..cutils import KNOWN_BACKENDS
from . import exceptions


class ConfigSource(enum.Enum):
    """Enumeration of config option sources."""

    DEFAULT = "default"
    """Config value comes from its default value"""

    CFGFILE = "config_file"
    """Config value comes from the configuration file"""

    ENVVAR = "environment_var"
    """Config value comes from environment variable"""

    CLIOPT = "command_line"
    """Config value comes from command line parameter"""

    SETTER = "setter"
    """Config value set by config setter after importing FEW"""


T = TypeVar("T")

ENVVAR_PREFIX: str = "FEW_"


@dataclasses.dataclass
class ConfigEntry(Generic[T]):
    """Description of a configuration entry."""

    label: str
    """How the entry is referred to in Python code (config.get("label"))"""
    description: str
    """Description of the configuration entry"""

    type: TypeVar = T
    """Type of the value"""

    default: Optional[T] = None
    """Default value"""

    cfg_entry: Optional[str] = None
    """Name of the entry in a config file"""

    env_var: Optional[str] = None
    """Entry corresponding env var (uppercase, without FEW_ prefix)"""

    cli_flags: Optional[Union[str, List[str]]] = None
    """Flag(s) for CLI arguments of this entry (e.g. "-f")"""

    cli_kwargs: Optional[Dict[str, Any]] = None
    """Supplementary arguments to argparse add_argument method for CLI options"""

    convert: Callable[[str], T] = None
    """Method used to convert a user input to expected type"""

    validate: Callable[[T], bool] = lambda _: True
    """Method used to validate the provided option value"""

    overwrite: Callable[[T, T], T] = lambda _, new: new
    """Method used to update the value if given by multiple means"""

    def __post_init__(self):
        if self.convert is None:
            self.convert = lambda v: self.type(v)


def compatibility_isinstance(obj, cls) -> bool:
    import sys
    import typing

    if (sys.version_info >= (3, 10)) or (typing.get_origin(cls) is None):
        try:
            return isinstance(obj, cls)
        except TypeError:
            pass

    if typing.get_origin(cls) is typing.Union:
        for arg in typing.get_args(cls):
            if compatibility_isinstance(obj, arg):
                return True
        return False

    if typing.get_origin(cls) is list:
        if not isinstance(obj, list):
            return False
        for item in obj:
            if not compatibility_isinstance(item, typing.get_args(cls)[0]):
                return False
        return True

    import collections.abc

    if typing.get_origin(cls) is collections.abc.Sequence:
        if not hasattr(obj, "__iter__"):
            return False
        for item in obj:
            if not compatibility_isinstance(item, typing.get_args(cls)[0]):
                return False
        return True

    raise NotImplementedError(
        "Compatiblity wrapper for isinstance on Python 3.9 does not support given type."
    )


@dataclasses.dataclass
class ConfigItem(Generic[T]):
    """Actual configuration entry with its run-time properties (value, source, ...)"""

    value: T  # Item value
    source: ConfigSource  # Source of the item current value


class ConfigConsumer(abc.ABC):
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

    @classmethod
    @abc.abstractmethod
    def config_entries(cls) -> List[ConfigEntry]:
        """Return the list of the class config entries"""
        raise NotImplementedError(
            "A ConfigConsumer must implement 'config_entries' method."
        )

    def __init__(
        self,
        config_file: Union[os.PathLike, Mapping[str, str], None] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
        set_args: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the items list and extra parameters."""
        config_entries = self.config_entries()

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
        set_items = ConfigConsumer._build_items_from_set(config_entries, set_args)

        # Build final item mapping
        self._items = self._overwrite(
            default_items, file_items, env_items, cli_items, set_items
        )

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

    def _overwrite(
        self,
        old_items: Dict[str, ConfigItem],
        new_items: Dict[str, ConfigItem],
        *other_items,
    ) -> Dict[str, ConfigItem]:
        merged_items = {}
        for label, old_item in old_items.items():
            if label not in new_items:
                merged_items[label] = old_item

        for label, new_item in new_items.items():
            if label not in old_items:
                merged_items[label] = new_item
                continue
            old_item = old_items[label]
            entry = self._entries[label]
            merged_items[label] = ConfigItem(
                value=entry.overwrite(old_item.value, new_item.value),
                source=new_item.source,
            )
        if len(other_items) > 0:
            return self._overwrite(merged_items, *other_items)
        return merged_items

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
    def _build_items_from_default(
        config_entries: Sequence[ConfigEntry],
    ) -> Dict[str, ConfigItem]:
        """Build a list of ConfigItem built from default-valued ConfigEntries."""
        return {
            entry.label: ConfigItem(value=entry.default, source=ConfigSource.DEFAULT)
            for entry in config_entries
            if compatibility_isinstance(entry.default, entry.type)
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
    def _build_parser(
        config_entries: Sequence[ConfigEntry],
        parent_parsers: Optional[Sequence[argparse.ArgumentParser]] = None,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            add_help=False,
            argument_default=argparse.SUPPRESS,
            parents=[] if parent_parsers is None else parent_parsers,
        )
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

        parsed_dict = vars(parsed_options)

        items_from_cli = {}

        for entry in config_entries:
            if entry.label not in parsed_dict:
                continue
            parsed_value = parsed_dict[entry.label]
            if compatibility_isinstance(parsed_dict[entry.label], entry.type):
                value = parsed_value
            else:
                value = entry.convert(parsed_value)

            items_from_cli[entry.label] = ConfigItem(
                value=value, source=ConfigSource.CLIOPT
            )

        return items_from_cli, extras_from_cli

    @staticmethod
    def _build_items_from_set(
        config_entries: Sequence[ConfigEntry], set_values: Optional[Dict[str, Any]]
    ) -> Dict[str, ConfigItem]:
        """Check that provided items match the entries."""
        set_items = {}

        if set_values is None:
            return set_items

        for config_entry in config_entries:
            if (label := config_entry.label) in set_values:
                set_value = set_values[label]
                if not config_entry.validate(set_value):
                    raise exceptions.ConfigurationValidationError(
                        "Configuration entry '{}' has invalid value '{}'".format(
                            label, set_value
                        )
                    )
                set_items[label] = ConfigItem(set_value, ConfigSource.SETTER)

        return set_items


def userstr_to_bool(user_str: str) -> Optional[bool]:
    """Convert a yes/no, on/off or true/false to bool."""
    if user_str.lower().startswith(("y", "t", "on", "1")):
        return True
    if user_str.lower().startswith(("n", "f", "off", "0")):
        return False
    return None


def userinput_to_pathlist(user_input) -> List[pathlib.Path]:
    """Convert a user input to a list of paths"""
    if user_input is None:
        return []
    if isinstance(user_input, str):
        return userinput_to_pathlist(user_input.split(";"))
    if compatibility_isinstance(user_input, Sequence[str]):
        return [pathlib.Path(path_str) for path_str in user_input]
    if compatibility_isinstance(user_input, Sequence[pathlib.Path]):
        return user_input
    raise ValueError(
        "User input '{}' of type '{}' is not convertible to a list of paths".format(
            user_input, type(user_input)
        )
    )


def userinput_to_strlist(user_input) -> List[str]:
    """Convert a user input to a list of paths"""
    if user_input is None:
        return []
    if isinstance(user_input, str):
        return user_input.replace(" ", ";").split(";")
    if compatibility_isinstance(user_input, List[str]):
        return user_input
    if compatibility_isinstance(user_input, Sequence[str]):
        return [input for input in user_input]
    raise ValueError(
        "User input '{}' of type '{}' is not convertible to a list of strings".format(
            user_input, type(user_input)
        )
    )


class InitialConfigConsumer(ConfigConsumer):
    """
    Class implementing first-pass config consumer.

    On first pass, we only detect if there are CLI arguments which disable
    config file or environment variables.
    """

    ignore_cfg: bool
    ignore_env: bool
    config_file: Optional[pathlib.Path]

    @staticmethod
    def config_entries() -> List[ConfigEntry]:
        return [
            ConfigEntry(
                label="ignore_cfg",
                description="Whether to ignore config file options",
                type=bool,
                default=False,
                env_var="IGNORE_CFG_FILE",
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

    def __init__(
        self,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
    ):
        super().__init__(config_file=None, env_vars=env_vars, cli_args=cli_args)


def get_package_basepath() -> pathlib.Path:
    import few

    return pathlib.Path(few.__file__).parent


class Configuration(ConfigConsumer):
    """
    Class implementing FEW complete configuration for the library.
    """

    log_level: int
    log_format: str
    file_registry_path: Optional[pathlib.Path]
    file_storage_path: Optional[pathlib.Path]
    file_download_path: Optional[pathlib.Path]
    file_allow_download: bool
    file_integrity_check: str
    file_extra_paths: List[pathlib.Path]
    file_disabled_tags: Optional[List[str]]
    enabled_backends: Optional[List[str]]

    @staticmethod
    def config_entries() -> List[ConfigEntry]:
        from few import _is_editable as is_editable_mode

        return [
            ConfigEntry(
                label="log_level",
                description="Application log level",
                type=int,
                default=logging.WARN,
                cli_flags=["--log-level"],
                env_var="LOG_LEVEL",
                cfg_entry="log-level",
                convert=Configuration._str_to_logging_level,
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
                default=(get_package_basepath() / "data") if is_editable_mode else None,
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
                default=(get_package_basepath() / "data") if is_editable_mode else None,
                cli_flags=["--download-dir"],
                env_var="FILE_DOWNLOAD_DIR",
                cfg_entry="file-download-dir",
                convert=lambda p: None if p is None else pathlib.Path(p),
                validate=lambda p: True
                if p is None
                else (p.is_dir() if p.is_absolute() else True),
            ),
            ConfigEntry(
                label="file_allow_download",
                description="Whether file manager can download missing files from internet",
                type=bool,
                default=True,
                cli_flags="--file-download",
                cli_kwargs={"action": argparse.BooleanOptionalAction},
                env_var="FILE_ALLOW_DOWNLOAD",
                cfg_entry="file-allow-download",
                convert=lambda x: userstr_to_bool(x) if isinstance(x, str) else bool(x),
            ),
            ConfigEntry(
                label="file_integrity_check",
                description="When should th file manager perform integrity checks (never, once, always)",
                type=str,
                default="once",
                cli_flags="--file-integrity-check",
                cli_kwargs={"choices": ("never", "once", "always")},
                env_var="FILE_INTEGRITY_CHECK",
                cfg_entry="file-integrity-check",
                validate=lambda x: x in ("never", "once", "always"),
            ),
            ConfigEntry(
                label="file_extra_paths",
                description="Supplementary paths in which FEW will search for files",
                type=Optional[List[pathlib.Path]],
                default=[get_package_basepath() / "data"],
                cli_flags=["--extra-path"],
                cli_kwargs={"action": "append"},
                env_var="FILE_EXTRA_PATHS",
                cfg_entry="file-extra-paths",
                convert=userinput_to_pathlist,
                overwrite=lambda old, new: old + new
                if old is not None
                else new,  # concatenate extra path lists
            ),
            ConfigEntry(
                label="file_disabled_tags",
                description="Tags for which access to associated files is disabled",
                type=Optional[List[str]],
                default=None,
                env_var="DISABLED_TAGS",
                convert=userinput_to_strlist,
                overwrite=lambda old, new: old + new
                if old is not None
                else new,  # concatenate tag lists
            ),
            ConfigEntry(
                label="enabled_backends",
                description="List of backends that must be enabled",
                type=Optional[List[str]],
                default=None,
                cli_flags="--enable-backend",
                cli_kwargs={"action": "append"},
                env_var="ENABLED_BACKENDS",
                cfg_entry="enabled-backends",
                convert=lambda x: [v.lower() for v in userinput_to_strlist(x)],
                validate=lambda x: all(v in KNOWN_BACKENDS for v in x)
                if x is not None
                else True,
            ),
        ]

    def __init__(
        self,
        config_file: Union[os.PathLike, Mapping[str, str], None] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
        set_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            config_file=config_file,
            env_vars=env_vars,
            cli_args=cli_args,
            set_args=set_args,
        )

        # Post-init task: read -v and -q options
        self._handle_verbosity()

    @staticmethod
    def _str_to_logging_level(input: Union[str, int]) -> int:
        if isinstance(input, int):
            return input
        as_int_level = logging.getLevelName(input.upper())
        if isinstance(as_int_level, int):
            return as_int_level
        raise exceptions.ConfigurationValidationError(
            "'{}' is not a valid log level.".format(input)
        )

    def _apply_verbosity(self, level):
        self._items["log_level"] = ConfigItem(value=level, source=ConfigSource.CLIOPT)

    @staticmethod
    def _build_verbosity_parser() -> argparse.ArgumentParser:
        """Build a parser to handle -v and -Q options"""
        parser = argparse.ArgumentParser(add_help=False)
        exclusive_groups = parser.add_mutually_exclusive_group()
        exclusive_groups.add_argument(
            "-v",
            "--verbose",
            dest="verbose_count",
            action="count",
            default=0,
            help="Increase the verbosity (can be used multiple times)",
        )
        exclusive_groups.add_argument(
            "-Q",
            "--quiet",
            dest="quiet",
            action="store_true",
            help="Disable all logging outputs",
        )
        return parser

    def _handle_verbosity(self):
        parser = self._build_verbosity_parser()
        parsed_options, self._extra_cli = parser.parse_known_args(self._extra_cli)

        from .globals import get_logger

        if parsed_options.quiet:
            get_logger().debug(
                "Logger level set to CRITICAL since quiet mode is requested."
            )
            self._apply_verbosity(logging.CRITICAL)
        else:
            if (count := parsed_options.verbose_count) == 0:
                return
            old_level = self.log_level
            new_level = old_level - count * 10
            get_logger().debug(
                "Logger level is decreased from {} to {} since verbose flag was set {} times".format(
                    old_level, new_level, count
                )
            )
            self._apply_verbosity(
                new_level if new_level > logging.DEBUG else logging.DEBUG
            )

    def build_cli_parent_parsers(self) -> List[argparse.ArgumentParser]:
        """Build a Parser that can be used as parent parser from CLI-specific parsers"""
        init_parser = ConfigConsumer._build_parser(
            InitialConfigConsumer.config_entries()
        )
        verbosity_parser = self._build_verbosity_parser()
        complete_parser = ConfigConsumer._build_parser(
            Configuration.config_entries(),
            parent_parsers=[init_parser, verbosity_parser],
        )
        return [complete_parser]


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
    from .globals import get_logger

    for location in LOCATIONS:
        if location.is_file():
            get_logger().debug("Configuration file located in '{}'".format(location))
            return location
        get_logger().debug("Configuration file not found in '{}'".format(location))
    return None
