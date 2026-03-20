"""
tests/test_unit.py

24 unit tests for LLMWatch.
Runs without any API keys — safe for CI.

Each test maps to either a bug we fixed or a core
behavior that would fail silently if broken.
"""

import os
import time
import warnings
import inspect
import pytest


# ─────────────────────────────────────────────
# calculate_cost — 6 tests
# ─────────────────────────────────────────────

from llmwatch.metrics import calculate_cost, CostForecaster


def test_known_model_returns_correct_value():
    """If wrong, every cost in the DB is wrong."""
    cost = calculate_cost("gpt-4o-mini", 1000, 1000)
    expected = (1000 / 1000) * 0.000150 + (1000 / 1000) * 0.000600
    assert abs(cost - expected) < 1e-9


def test_unknown_model_returns_zero():
    """If wrong, code crashes on new models."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cost = calculate_cost("totally-fake-model", 1000, 500)
    assert cost == 0.0


def test_unknown_model_emits_warning():
    """If no warning, failure is silent — no one knows cost is wrong."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calculate_cost("totally-fake-model", 1000, 500)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "no pricing data" in str(w[0].message).lower()


def test_zero_tokens_returns_zero():
    """Empty calls should not be charged."""
    cost = calculate_cost("gpt-4o-mini", 0, 0)
    assert cost == 0.0


def test_cost_scales_linearly():
    """2000 tokens should cost exactly 2x 1000 tokens."""
    cost_1k = calculate_cost("gpt-4o-mini", 1000, 0)
    cost_2k = calculate_cost("gpt-4o-mini", 2000, 0)
    assert abs(cost_2k - cost_1k * 2) < 1e-9


def test_output_tokens_cost_more_than_input():
    """For every model, output rate > input rate."""
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "claude-opus-4-6"]:
        input_cost  = calculate_cost(model, 1000, 0)
        output_cost = calculate_cost(model, 0, 1000)
        assert output_cost > input_cost, \
            f"{model}: output should cost more than input"


# ─────────────────────────────────────────────
# CostForecaster — 4 tests
# ─────────────────────────────────────────────

def test_empty_forecaster_returns_zero():
    """No data should mean no projection."""
    f = CostForecaster()
    assert f.spend_rate_per_minute() == 0.0
    assert f.projected_hourly()      == 0.0
    assert f.projected_daily()       == 0.0
    assert f.projected_monthly()     == 0.0


def test_single_record_returns_zero_rate():
    """Can't calculate a rate with only one data point."""
    f = CostForecaster()
    f.record(0.001)
    assert f.spend_rate_per_minute() == 0.0


def test_two_records_returns_nonzero_rate():
    """Basic sanity — two data points should give a nonzero rate."""
    f = CostForecaster()
    f.record(0.001)
    time.sleep(0.1)
    f.record(0.001)
    assert f.spend_rate_per_minute() > 0.0


def test_projections_scale_correctly():
    """If math is wrong, budget projections are wrong."""
    f = CostForecaster()
    f.record(0.001)
    time.sleep(1.0)
    f.record(0.001)
    rate = f.spend_rate_per_minute()
    assert rate > 0
    assert abs(f.projected_hourly()  - rate * 60)           < rate * 60 * 0.01
    assert abs(f.projected_daily()   - rate * 60 * 24)      < rate * 60 * 24 * 0.01
    assert abs(f.projected_monthly() - rate * 60 * 24 * 30) < rate * 60 * 24 * 30 * 0.01

# ─────────────────────────────────────────────
# LLMLogger — 7 tests
# ─────────────────────────────────────────────

from llmwatch.logger import LLMLogger

TEST_DB = "/tmp/llmwatch_unit_test.db"


@pytest.fixture
def logger():
    """Fresh logger for each test, cleaned up after."""
    l = LLMLogger(db_path=TEST_DB)
    yield l
    l.clear()
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _sample_event(**overrides):
    base = {
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "input_tokens":  100,
        "output_tokens": 50,
        "duration":      1.5,
        "cost_usd":      0.000045,
        "prompt":        "test prompt",
        "response":      "test response",
        "error":         None
    }
    base.update(overrides)
    return base


def test_db_file_created_on_init(logger):
    """Basic sanity — database must exist after init."""
    assert os.path.exists(TEST_DB)


def test_log_and_retrieve_row(logger):
    """Core functionality — what goes in must come out."""
    logger.log(_sample_event())
    rows = logger.query(provider="openai")
    assert len(rows) == 1
    assert rows[0]["provider"]     == "openai"
    assert rows[0]["model"]        == "gpt-4o-mini"
    assert rows[0]["input_tokens"] == 100


def test_timestamp_is_utc_aware(logger):
    """Bug #6 — datetime.utcnow() deprecated. Must use timezone.utc."""
    logger.log(_sample_event())
    rows = logger.query()
    ts = rows[0]["timestamp"]
    assert "+00:00" in ts, \
        "Timestamp must include UTC offset — use datetime.now(timezone.utc)"

def test_cost_usd_stored_correctly(logger):
    """Bug #1 — cost_usd was always 0.0 before the fix."""
    logger.log(_sample_event(cost_usd=0.000072))
    rows = logger.query()
    assert rows[0]["cost_usd"] == 0.000072, \
        "cost_usd must be stored correctly — was always 0.0 before Bug #1 fix"


def test_query_filters_by_provider(logger):
    """Wrong filters return wrong data — silent corruption."""
    logger.log(_sample_event(provider="openai",    model="gpt-4o"))
    logger.log(_sample_event(provider="anthropic", model="claude-sonnet-4-6"))
    rows = logger.query(provider="anthropic")
    assert len(rows) == 1
    assert rows[0]["provider"] == "anthropic"


def test_query_returns_newest_first(logger):
    """Wrong order breaks time-series analysis."""
    for i in range(3):
        logger.log(_sample_event(input_tokens=i * 10))
        time.sleep(0.01)
    rows = logger.query()
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True), \
        "Rows must be returned newest first"

def test_summary_total_cost_adds_up(logger):
    """Bug #1 — if cost is wrong, budget tracking is broken."""
    logger.log(_sample_event(cost_usd=0.000045))
    logger.log(_sample_event(cost_usd=0.000090))
    summary = logger.summary()
    assert summary["total_calls"] == 2
    assert abs(summary["total_cost_usd"] - 0.000135) < 1e-9


def test_db_path_not_in_repo_root(logger):
    """Bug #7 — llmwatch.db in repo root gets committed to git."""
    assert logger.db_path != "llmwatch.db", \
        "DB path must not be 'llmwatch.db' — it will be committed to git"


# ─────────────────────────────────────────────
# middleware.py — 3 tests
# ─────────────────────────────────────────────

from llmwatch.middleware import (
    LLMWatch, PROVIDER_MAP,
    _call_openai, _call_anthropic, _call_groq
)


def test_invalid_provider_raises_value_error():
    """Clear error message, not a cryptic crash."""
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMWatch(provider="invalid_provider", model="gpt-4o-mini")


def test_all_provider_functions_accept_client_first():
    """Bug #5 — client must be first arg for connection pooling to work."""
    for fn in [_call_openai, _call_anthropic, _call_groq]:
        params = list(inspect.signature(fn).parameters.keys())
        assert params[0] == "client", \
            f"{fn.__name__} must accept 'client' as first argument — Bug #5 fix"


def test_provider_map_has_all_providers():
    """All documented providers must be in the map."""
    assert "openai"    in PROVIDER_MAP
    assert "anthropic" in PROVIDER_MAP
    assert "groq"      in PROVIDER_MAP


# ─────────────────────────────────────────────
# Public API — 3 tests
# ─────────────────────────────────────────────

def test_llmwatch_importable():
    """Package entry point must work."""
    from llmwatch import LLMWatch
    assert LLMWatch is not None


def test_record_agent_metrics_importable():
    """Public API must be intact."""
    from llmwatch import record_agent_metrics
    assert record_agent_metrics is not None


def test_version_defined():
    """Package version must exist for PyPI."""
    import llmwatch
    assert hasattr(llmwatch, "__version__")
    assert isinstance(llmwatch.__version__, str)
    assert len(llmwatch.__version__) > 0