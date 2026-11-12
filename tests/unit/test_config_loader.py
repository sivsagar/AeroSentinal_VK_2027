"""
Unit tests for src/aerosentinel/utils/config_loader.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure src/ is importable
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aerosentinel.utils.config_loader import ConfigLoader, ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_dir: Path, filename: str, content: str) -> None:
    """Write *content* to *tmp_dir/filename*."""
    (tmp_dir / filename).write_text(textwrap.dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with a sample YAML file."""
    _write_yaml(
        tmp_path,
        "system.yaml",
        """\
        system:
          name: TestSentinel
          log_level: DEBUG
          nested:
            deep:
              value: 42
        performance:
          cpu_warning_temp_c: 75
        """,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadYaml:
    """ConfigLoader.load() — basic YAML loading."""

    def test_load_existing_file(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.name") == "TestSentinel"

    def test_loaded_file_tracked(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert len(loader._loaded_files) == 1

    def test_load_missing_file_raises(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        with pytest.raises(ConfigError, match="not found"):
            loader.load("does_not_exist.yaml")

    def test_merge_two_files(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "a.yaml", "key_a: 1\n")
        _write_yaml(tmp_path, "b.yaml", "key_b: 2\n")
        loader = ConfigLoader(config_dir=tmp_path)
        loader.load("a.yaml")
        loader.load("b.yaml")
        assert loader.get("key_a") == 1
        assert loader.get("key_b") == 2


class TestEnvVarOverride:
    """Environment variable override support."""

    def test_top_level_override(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEROSENTINEL_SYSTEM__LOG_LEVEL", "WARNING")
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.log_level") == "WARNING"

    def test_bool_coercion(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEROSENTINEL_SYSTEM__MOCK_MODE", "true")
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.mock_mode") is True

    def test_int_coercion(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEROSENTINEL_PERFORMANCE__CPU_WARNING_TEMP_C", "80")
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("performance.cpu_warning_temp_c") == 80

    def test_none_coercion(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEROSENTINEL_SYSTEM__NAME", "null")
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.name") is None

    def test_env_without_prefix_ignored(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYSTEM__LOG_LEVEL", "CRITICAL")
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        # Should NOT be overridden because prefix is missing
        assert loader.get("system.log_level") == "DEBUG"


class TestNestedKeyAccess:
    """Dot-notation key access."""

    def test_top_level_key(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("performance.cpu_warning_temp_c") == 75

    def test_deeply_nested_key(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.nested.deep.value") == 42

    def test_get_dict_node(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        node = loader.get("system.nested")
        assert isinstance(node, dict)
        assert "deep" in node


class TestMissingKeyDefault:
    """Default value behaviour when a key is absent."""

    def test_missing_key_returns_default(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("system.nonexistent", default="fallback") == "fallback"

    def test_missing_key_default_none(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        assert loader.get("missing.key", default=None) is None

    def test_missing_key_no_default_raises(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        with pytest.raises(KeyError):
            loader.get("missing.key")


class TestInvalidYaml:
    """Invalid YAML content handling."""

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text(
            "key: [unclosed\n  - broken\n",
            encoding="utf-8",
        )
        loader = ConfigLoader(config_dir=tmp_path)
        with pytest.raises(ConfigError, match="Invalid YAML"):
            loader.load("bad.yaml")

    def test_non_mapping_yaml_raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "list.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
        loader = ConfigLoader(config_dir=tmp_path)
        with pytest.raises(ConfigError, match="Expected a YAML mapping"):
            loader.load("list.yaml")


class TestReload:
    """ConfigLoader.reload() re-reads from disk."""

    def test_reload_picks_up_changes(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "dynamic.yaml"
        yaml_file.write_text("value: original\n", encoding="utf-8")

        loader = ConfigLoader(config_dir=tmp_path)
        loader.load("dynamic.yaml")
        assert loader.get("value") == "original"

        yaml_file.write_text("value: updated\n", encoding="utf-8")
        loader.reload()
        assert loader.get("value") == "updated"


class TestValidate:
    """ConfigLoader.validate() required keys."""

    def test_validate_passes_when_all_present(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        # Should not raise
        loader.validate(["system.name", "system.log_level"])

    def test_validate_raises_when_key_missing(self, tmp_config_dir: Path) -> None:
        from aerosentinel.utils.config_loader import ConfigError

        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load("system.yaml")
        with pytest.raises(ConfigError, match="Missing required"):
            loader.validate(["system.name", "system.nonexistent_key"])
