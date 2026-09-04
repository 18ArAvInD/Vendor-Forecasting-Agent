"""Unit tests for config.py (sub-task 2.2).

Covers, using standard pytest only (NO Hypothesis):
- Documented defaults are applied when values are absent (Req 2.3).
- ConfigError is raised (naming the offending setting) on invalid settings
  (Req 2.4), and no Config is returned.
- ConfigError is raised when the source is absent/unreadable, on malformed JSON,
  and on a JSON value that is not an object (Req 2.5), and no Config is returned.
- A valid JSON config file loads correctly and absent keys receive defaults.
- Loading completes well under 5 seconds (Req 2.1).

_Requirements: 2.1, 2.3, 2.4, 2.5_
"""

import json
import time

import pytest

from vendor_forecasting_agent.config import (
    DEFAULT_RANDOM_SEED,
    ConfigError,
    load_config,
)
from vendor_forecasting_agent.schema import Config


# ---------------------------------------------------------------------------
# Defaults applied when values are absent (Req 2.3)
# ---------------------------------------------------------------------------


def test_load_config_none_applies_all_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.random_seed == DEFAULT_RANDOM_SEED == 0
    assert cfg.forecast_horizon == 12
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_load_config_default_argument_applies_all_defaults() -> None:
    # load_config() with no argument defaults source to None.
    cfg = load_config()
    assert cfg.random_seed == DEFAULT_RANDOM_SEED
    assert cfg.forecast_horizon == 12
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_load_config_mapping_fills_absent_keys_with_defaults() -> None:
    cfg = load_config({"random_seed": 42})
    assert cfg.random_seed == 42
    # Absent keys receive documented defaults.
    assert cfg.forecast_horizon == 12
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_load_config_mapping_omitting_seed_uses_default_seed() -> None:
    cfg = load_config({"forecast_horizon": 6})
    assert cfg.random_seed == DEFAULT_RANDOM_SEED
    assert cfg.forecast_horizon == 6
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_load_config_mapping_full_override_preserves_values() -> None:
    cfg = load_config(
        {
            "random_seed": 123,
            "forecast_horizon": 24,
            "record_count": 5000,
            "stable_threshold_pct": 7.5,
        }
    )
    assert cfg.random_seed == 123
    assert cfg.forecast_horizon == 24
    assert cfg.record_count == 5000
    assert cfg.stable_threshold_pct == 7.5


# ---------------------------------------------------------------------------
# ConfigError on invalid setting, naming it (Req 2.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mapping", "offending_setting"),
    [
        ({"random_seed": -1}, "random_seed"),
        ({"random_seed": 4_294_967_296}, "random_seed"),
        ({"random_seed": 0, "forecast_horizon": 0}, "forecast_horizon"),
        ({"random_seed": 0, "forecast_horizon": 121}, "forecast_horizon"),
        ({"random_seed": 0, "record_count": 0}, "record_count"),
        ({"random_seed": 0, "record_count": 1_000_001}, "record_count"),
    ],
)
def test_invalid_setting_raises_config_error_naming_setting(
    mapping, offending_setting
) -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_config(mapping)
    # The error message names the offending setting and no Config is returned
    # (the call raises before returning).
    assert offending_setting in str(exc_info.value)


# ---------------------------------------------------------------------------
# ConfigError on unreadable/absent/malformed source (Req 2.5)
# ---------------------------------------------------------------------------


def test_nonexistent_path_raises_config_error_indicating_load_failure(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ConfigError) as exc_info:
        load_config(missing)
    assert "could not be loaded" in str(exc_info.value)


def test_malformed_json_file_raises_config_error(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad)
    assert "could not be loaded" in str(exc_info.value)


def test_json_array_source_raises_config_error(tmp_path) -> None:
    arr = tmp_path / "array.json"
    arr.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(arr)
    assert "could not be loaded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Valid path load (Req 2.3, 2.5 happy path)
# ---------------------------------------------------------------------------


def test_valid_json_file_loads_and_fills_defaults(tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"random_seed": 7, "forecast_horizon": 6}), encoding="utf-8"
    )
    cfg = load_config(cfg_path)
    assert cfg.random_seed == 7
    assert cfg.forecast_horizon == 6
    # Absent keys received documented defaults.
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_valid_json_file_string_path_loads(tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"random_seed": 99}), encoding="utf-8")
    cfg = load_config(str(cfg_path))
    assert cfg.random_seed == 99
    assert cfg.forecast_horizon == 12


# ---------------------------------------------------------------------------
# Load timing < 5s (Req 2.1)
# ---------------------------------------------------------------------------


def test_load_config_none_completes_well_under_5_seconds() -> None:
    start = time.perf_counter()
    load_config(None)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0


def test_load_config_mapping_completes_well_under_5_seconds() -> None:
    start = time.perf_counter()
    load_config({"random_seed": 42, "forecast_horizon": 24})
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
