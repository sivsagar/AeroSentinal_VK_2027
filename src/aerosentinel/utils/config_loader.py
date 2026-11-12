"""
AeroSentinel Configuration Loader
===================================
Loads, merges, and validates YAML configuration files.
Supports environment variable overrides and nested key access via dot notation.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required: pip install pyyaml") from e

logger = logging.getLogger(__name__)

_ENV_PREFIX = "AEROSENTINEL_"
_SENTINEL = object()


class ConfigError(Exception):
    """Raised for configuration loading or validation errors."""


class ConfigLoader:
    """
    Load and merge YAML configuration files from the config/ directory.

    Features:
      - Load single or multiple YAML files and deep-merge them.
      - Override any key via environment variables prefixed with
        ``AEROSENTINEL_``.  Nested keys are expressed with double-underscores:
        ``AEROSENTINEL_SYSTEM__LOG_LEVEL=DEBUG`` overrides
        ``system.log_level``.
      - Retrieve values with dot-notation: ``cfg.get("camera.rgb.resolution")``.
      - Validate that a set of required keys are present.
      - Reload the config in-place without creating a new instance.

    Example::

        loader = ConfigLoader(config_dir="config/")
        loader.load("system.yaml")
        loader.load("camera.yaml")
        level = loader.get("system.log_level", default="INFO")
    """

    def __init__(self, config_dir: Union[str, Path] = "config/") -> None:
        """
        Initialise the loader.

        Parameters
        ----------
        config_dir:
            Directory that contains the YAML files.  Relative paths are
            resolved relative to the current working directory at construction
            time.
        """
        self._config_dir: Path = Path(config_dir).resolve()
        self._data: Dict[str, Any] = {}
        self._loaded_files: List[Path] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, *filenames: str) -> "ConfigLoader":
        """
        Load one or more YAML files and deep-merge them into the current config.

        Parameters
        ----------
        filenames:
            One or more YAML filenames (relative to *config_dir*).

        Returns
        -------
        ConfigLoader
            *self*, so calls can be chained:
            ``loader.load("system.yaml").load("camera.yaml")``.

        Raises
        ------
        ConfigError
            If a file is not found, cannot be read, or contains invalid YAML.
        """
        for filename in filenames:
            path = self._config_dir / filename
            self._load_file(path)
        self._apply_env_overrides()
        return self

    def load_all(self) -> "ConfigLoader":
        """
        Load every ``*.yaml`` / ``*.yml`` file found in *config_dir*.

        Returns
        -------
        ConfigLoader
            *self* for chaining.
        """
        yaml_files = sorted(
            list(self._config_dir.glob("*.yaml"))
            + list(self._config_dir.glob("*.yml"))
        )
        for path in yaml_files:
            self._load_file(path)
        self._apply_env_overrides()
        return self

    def reload(self) -> "ConfigLoader":
        """
        Re-read all previously loaded files from disk and rebuild the config.

        Any environment variable overrides are re-applied afterwards.

        Returns
        -------
        ConfigLoader
            *self* for chaining.
        """
        files = list(self._loaded_files)
        self._data = {}
        self._loaded_files = []
        for path in files:
            self._load_file(path)
        self._apply_env_overrides()
        logger.debug("Configuration reloaded from %d file(s)", len(self._loaded_files))
        return self

    def get(self, key: str, default: Any = _SENTINEL) -> Any:
        """
        Retrieve a configuration value using dot-notation.

        Parameters
        ----------
        key:
            Dot-separated path, e.g. ``"system.log_level"`` or
            ``"depth_processing.filters.spatial.alpha"``.
        default:
            Value to return when the key is absent.  If omitted and the key
            is absent, a :class:`KeyError` is raised.

        Returns
        -------
        Any
            The value stored at *key*.

        Raises
        ------
        KeyError
            If *key* is not found and no *default* was provided.
        """
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                if default is _SENTINEL:
                    raise KeyError(f"Configuration key not found: '{key}'")
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot-notation (in-memory only).

        Parameters
        ----------
        key:
            Dot-separated path.
        value:
            Value to store.
        """
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Return True if *key* exists in the configuration."""
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    def validate(self, required_keys: List[str]) -> None:
        """
        Raise :class:`ConfigError` if any of *required_keys* are absent.

        Parameters
        ----------
        required_keys:
            List of dot-notation keys that must be present.

        Raises
        ------
        ConfigError
            Contains the list of all missing keys.
        """
        missing = [k for k in required_keys if not self.has(k)]
        if missing:
            raise ConfigError(
                "Missing required configuration keys: " + ", ".join(missing)
            )

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the entire config as a plain dict."""
        return copy.deepcopy(self._data)

    def as_json(self, indent: int = 2) -> str:
        """Return the entire config serialised as a JSON string."""
        return json.dumps(self._data, indent=indent, default=str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> None:
        """Read *path*, parse YAML, and deep-merge into ``self._data``."""
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc

        try:
            parsed: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ConfigError(
                f"Expected a YAML mapping at the top level of {path}, "
                f"got {type(parsed).__name__}"
            )

        self._data = _deep_merge(self._data, parsed)
        if path not in self._loaded_files:
            self._loaded_files.append(path)
        logger.debug("Loaded config: %s", path)

    def _apply_env_overrides(self) -> None:
        """
        Apply environment-variable overrides to ``self._data``.

        Variables must be prefixed with ``AEROSENTINEL_``.  Nested keys use
        ``__`` as separator (double underscore), and key names are lowercased::

            AEROSENTINEL_SYSTEM__LOG_LEVEL=DEBUG
            → self._data["system"]["log_level"] = "DEBUG"
        """
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(_ENV_PREFIX):
                continue
            stripped = env_key[len(_ENV_PREFIX):]
            parts = [p.lower() for p in stripped.split("__")]

            # Attempt to coerce the string value to a Python type
            coerced = _coerce_value(env_val)

            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = coerced
            logger.debug("Env override applied: %s -> %r", ".".join(parts), coerced)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ConfigLoader(config_dir={self._config_dir!r}, "
            f"files={[f.name for f in self._loaded_files]})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *override* into *base*.

    For dict values at the same key, the merge descends.  For all other
    types, the value from *override* wins.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _coerce_value(value: str) -> Any:
    """
    Try to coerce a string from an environment variable to a Python type.

    Priority: ``null/none`` → None, ``true/false`` → bool,
    integer literal → int, float literal → float, else str.
    """
    if value.lower() in ("null", "none", ""):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
