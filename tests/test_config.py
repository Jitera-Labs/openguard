"""Unit tests for src/config.py"""

import os

import pytest


class TestConfigValue:
    """Test Config class value resolution."""

    def test_str_default(self, monkeypatch):
        """Config resolves string default when env var absent."""
        monkeypatch.delenv("OPENGUARD_TEST_STR_MISSING", raising=False)
        from src.config import Config

        c = Config("OPENGUARD_TEST_STR_MISSING", str, "default-value")
        assert c.value == "default-value"

    def test_env_var_overrides_default(self, monkeypatch):
        """Config reads value from environment variable."""
        monkeypatch.setenv("OPENGUARD_TEST_STR", "from-env")
        from src.config import Config

        c = Config("OPENGUARD_TEST_STR", str, "default")
        assert c.value == "from-env"

    def test_int_conversion(self, monkeypatch):
        """Config converts string env var to int."""
        monkeypatch.setenv("OPENGUARD_TEST_INT", "42")
        from src.config import Config

        c = Config("OPENGUARD_TEST_INT", int, "0")
        assert c.value == 42

    def test_bool_conversion_true(self, monkeypatch):
        """Config converts 'true' string to bool True."""
        monkeypatch.setenv("OPENGUARD_TEST_BOOL", "true")
        from src.config import Config

        c = Config("OPENGUARD_TEST_BOOL", bool, "false")
        assert c.value is True

    def test_bool_conversion_false(self, monkeypatch):
        """Config converts 'false' string to bool False."""
        monkeypatch.setenv("OPENGUARD_TEST_BOOL2", "false")
        from src.config import Config

        c = Config("OPENGUARD_TEST_BOOL2", bool, "false")
        assert c.value is False

    def test_bool_conversion_variants(self, monkeypatch):
        """Config accepts 1/yes/on as truthy bool values."""
        from src.config import Config

        for truthy in ("1", "yes", "on", "true", "True", "TRUE"):
            monkeypatch.setenv("OPENGUARD_TEST_BOOL_VAR", truthy)
            c = Config("OPENGUARD_TEST_BOOL_VAR", bool, "false")
            assert c.value is True, f"Expected True for {truthy!r}"

    def test_float_conversion(self, monkeypatch):
        """Config converts string env var to float."""
        monkeypatch.setenv("OPENGUARD_TEST_FLOAT", "3.14")
        from src.config import Config

        c = Config("OPENGUARD_TEST_FLOAT", float, "0.0")
        assert c.value == pytest.approx(3.14)


class TestStrList:
    def test_empty_string(self):
        from src.config import StrList

        result = StrList.from_string("")
        assert result == []

    def test_single_item(self):
        from src.config import StrList

        result = StrList.from_string("item1")
        assert result == ["item1"]

    def test_multiple_items(self):
        from src.config import StrList

        result = StrList.from_string("a;b;c")
        assert result == ["a", "b", "c"]

    def test_strips_whitespace(self):
        from src.config import StrList

        result = StrList.from_string("  a ;  b  ; c  ")
        assert result == ["a", "b", "c"]


class TestConfigDict:
    def test_empty_string(self):
        from src.config import ConfigDict

        result = ConfigDict.from_string("")
        assert result == {}

    def test_key_value_pairs(self):
        from src.config import ConfigDict

        result = ConfigDict.from_string("key1=value1,key2=42")
        assert result["key1"] == "value1"
        assert result["key2"] == 42

    def test_bool_values(self):
        from src.config import ConfigDict

        result = ConfigDict.from_string("flag=true,other=false")
        assert result["flag"] is True
        assert result["other"] is False


class TestWildcardConfig:
    def test_wildcard_collects_matching_env_vars(self, monkeypatch):
        """Wildcard config collects all matching env vars."""
        monkeypatch.setenv("OPENGUARD_OPENAI_URL_ONE", "http://one.test")
        monkeypatch.setenv("OPENGUARD_OPENAI_URL_TWO", "http://two.test")
        from src.config import Config

        c = Config("OPENGUARD_OPENAI_URL_*", str, "")
        urls = c.value
        assert "http://one.test" in urls
        assert "http://two.test" in urls

    def test_wildcard_returns_default_when_no_matches(self, monkeypatch):
        """Wildcard config returns default list when no env vars match."""
        import src.config as config_module

        # Use a prefix guaranteed not to exist in any real environment
        prefix = "OPENGUARD_ZZZTESTONLY_UNIQUE_"
        for key in list(os.environ.keys()):
            if key.startswith(prefix):
                monkeypatch.delenv(key, raising=False)

        # Also clear persistent config so provider keys don't bleed in
        monkeypatch.setattr(config_module, "PERSISTENT_CONFIG", {})

        from src.config import Config

        c = Config(f"{prefix}*", str, "http://default.test")
        assert c.value == ["http://default.test"]


def _make_config(value: str):
    """Create a Config[str] with a pre-set __value__, bypassing env resolution."""
    from src.config import Config

    c = Config.__new__(Config)
    c.name = "OPENGUARD_CONFIG"
    c.type = str
    c.default = value
    c.description = None
    c.__value__ = value
    return c


def _clear_config_extras(monkeypatch):
    """Remove all OPENGUARD_CONFIG_* env vars so tests start clean."""
    for key in list(os.environ.keys()):
        if key.startswith("OPENGUARD_CONFIG_"):
            monkeypatch.delenv(key, raising=False)


class TestGetConfigPaths:
    """Tests for get_config_paths()."""

    def test_single_path(self, monkeypatch):
        """Single path returns a one-element list."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("./guards.yaml"))
        assert config.get_config_paths() == ["./guards.yaml"]

    def test_comma_separated(self, monkeypatch):
        """Comma-separated value is split into individual paths."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("a.yaml,b.yaml"))
        assert config.get_config_paths() == ["a.yaml", "b.yaml"]

    def test_comma_with_whitespace(self, monkeypatch):
        """Whitespace around comma-separated entries is stripped."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config(" a.yaml , b.yaml "))
        assert config.get_config_paths() == ["a.yaml", "b.yaml"]

    def test_named_extras(self, monkeypatch):
        """OPENGUARD_CONFIG_* env vars are appended after the base paths."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("a.yaml"))
        monkeypatch.setenv("OPENGUARD_CONFIG_EXTRA", "b.yaml")
        assert config.get_config_paths() == ["a.yaml", "b.yaml"]

    def test_named_extras_sorted(self, monkeypatch):
        """OPENGUARD_CONFIG_* extras are merged in alphabetical key order."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("a.yaml"))
        monkeypatch.setenv("OPENGUARD_CONFIG_Z", "z.yaml")
        monkeypatch.setenv("OPENGUARD_CONFIG_A", "aa.yaml")
        assert config.get_config_paths() == ["a.yaml", "aa.yaml", "z.yaml"]

    def test_deduplication(self, monkeypatch):
        """Duplicate paths appear only once, first occurrence wins."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("a.yaml"))
        monkeypatch.setenv("OPENGUARD_CONFIG_DUP", "a.yaml")
        assert config.get_config_paths() == ["a.yaml"]

    def test_empty_segment_stripped(self, monkeypatch):
        """Empty segments from double-commas are not included."""
        import src.config as config

        _clear_config_extras(monkeypatch)
        monkeypatch.setattr(config, "OPENGUARD_CONFIG", _make_config("a.yaml,,b.yaml"))
        assert config.get_config_paths() == ["a.yaml", "b.yaml"]
