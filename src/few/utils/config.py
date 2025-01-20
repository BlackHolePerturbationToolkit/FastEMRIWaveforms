"""Implementation of a centralized configuration management for FEW."""

import dataclasses
import enum
import os
from typing import TypeVar, Generic, Optional, List, Union, Sequence, Mapping, Dict, Callable, Tuple
from . import exceptions

class ConfigSource(enum.Enum):
    """Enumeration of config option sources."""
    DEFAULT = "default"
    CFGFILE = "config_file"
    ENVVAR = "environment_var"
    CLIOPT = "command_line"

T = TypeVar['T']

ENVVAR_PREFIX: str = "FEW_"

@dataclasses.dataclass
class ConfigEntry(Generic[T]):
    """Description of a configuration entry."""
    label: str  # How the entry is referred to in Python code (config.get("label"))
    description: str  # Description of the configuration entry
    default: Optional[T] = None  # Default value
    cfg_entry: Optional[str] = None  # Name of the entry in a config file
    env_var: Optional[str] = None  # Entry corresponding env var (uppercase, without FEW_ prefix)
    cli_flags: Optional[Union[str, List[str]]] = None  # Flag(s) for CLI arguments of this entry (e.g. "-f")
    type: TypeVar = T  # Type of the value
    convert: Callable[[str], T] = lambda val_str: T(val_str)
    validate: Callable[[T], bool] = lambda _: True

@dataclasses.dataclass
class ConfigItem(Generic[T]):
    """Actual configuration entry with its run-time properties (value, source, ...)"""
    value: T    # Item value
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

    def __init__(self,
                 config_entries: Sequence[ConfigEntry],
                 config_file: Optional[os.PathLike] = None,
                 env_vars: Optional[Mapping[str, str]] = None,
                 cli_args: Optional[Sequence[str]] = None):
        """Initialize the items list and extra parameters."""

        # Build the entries mapping
        self._entries = {entry.label: entry for entry in config_entries}

        # Build default items
        default_items = ConfigConsumer._build_items_from_default()

        # Retrieve option from sources
        opt_from_file = ConfigConsumer._get_from_config_file(config_file)
        opt_from_env = ConfigConsumer._get_from_envvars(env_vars)
        opt_from_cli = ConfigConsumer._get_from_cli_args(cli_args)

        # Consume options to build other item lists
        file_items, self._extra_file = ConfigConsumer._build_items_from_file(config_entries, opt_from_file)
        env_items, self._extra_env = ConfigConsumer._build_items_from_env(config_entries, opt_from_env)
        cli_items, self._extra_cli = ConfigConsumer._build_items_from_cli(config_entries, opt_from_cli)

        # Build final item mapping
        self._items = default_items | file_items | env_items | cli_items

        # Validate items:
        errors: List[Exception] = []
        for label, entry in self._entries.items():
            if not label in self._items:
                errors.append(exceptions.ConfigurationMissing("Configuration entry '{}' is missing.".format(label)))
                continue
            item = self._items[label]
            if not entry.validate(item.value):
                errors.append(exceptions.ConfigurationValidationError("Configuration entry '{}' has invalid value '{}'".format(label, item.value)))

        if errors:
            raise errors[0] if len(errors) == 1 else exceptions.ExceptionGroup("Invalid configuration due to previous issues.", errors)


    @staticmethod
    def _get_from_config_file(config_file: Optional[os.PathLike]) -> Dict[str, str]:
        """Read a config file (if existing) and return its items as a dictionary."""
        raise NotImplementedError("Implement reading options from file.")
        return {}

    @staticmethod
    def _get_from_envvars(env_vars: Optional[Mapping[str, str]]) -> Dict[str, str]:
        """Filter-out environment variables not matching a given prefix."""
        return {key: value for key, value
                in env_vars.items() if key.startswith(ENVVAR_PREFIX)} if env_vars is not None else {}

    @staticmethod
    def _get_from_cli_args(cli_args: Optional[Sequence[str]]) -> List[str]:
        """Build list of CLI arguments."""
        return [arg for arg in cli_args] if cli_args is not None else []

    @staticmethod
    def _build_items_from_default(config_entries: Sequence[ConfigEntry]) -> Dict[str, ConfigItem]:
        """Build a list of ConfigItem built from default-valued ConfigEntries."""
        return {
            entry.label: ConfigItem(value=entry.default, source=ConfigSource.DEFAULT) for entry in config_entries if isinstance(entry.default, entry.type)
        }

    @staticmethod
    def _build_items_from_file(config_entries: Sequence[ConfigEntry],
                               opt_from_file: Mapping[str, str]
                               ) -> Tuple[Dict[str, ConfigItem], Dict[str, str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        extras_from_file = {**opt_from_file}
        items_from_file = {entry.label: ConfigItem(value=entry.convert(extras_from_file.pop(entry.cfg_entry)), source=ConfigSource.CFGFILE)
                            for entry in config_entries if entry.cfg_entry in extras_from_file}

        return items_from_file, extras_from_file

    @staticmethod
    def _build_items_from_env(config_entries: Sequence[ConfigEntry],
                               opt_from_env: Mapping[str, str]
                               ) -> Tuple[Dict[str, ConfigItem], Dict[str, str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        extras_from_env = {**opt_from_env}
        items_from_env = {entry.label: ConfigItem(value=entry.convert(extras_from_env.pop(ENVVAR_PREFIX + entry.env_var)), source=ConfigSource.ENVVAR)
                            for entry in config_entries if entry.cfg_entry in extras_from_env}

        return items_from_env, extras_from_env

    @staticmethod
    def _build_items_from_cli(config_entries: Sequence[ConfigEntry],
                               opt_from_cli: Sequence[str]
                               ) -> Tuple[Dict[str, ConfigItem], List[str]]:
        """Extract configuration items from file option and build dict of unconsumed extra items."""
        raise NotImplementedError("Implement building options from CLI using argparse.")
