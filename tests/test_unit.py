"""
tests/test_unit.py
10 unit tests for LLMWatch.
Runs without any API keys — safe for CI.
Every test maps to a real bug we fixed or
a core behavior that would fail silently if broken.
"""

import os
import warnings
import pytest

from llmwatch.metrics import calculate_cost
from llmwatch.logger import LLMLogger
from llmwatch.middleware import LLMWatch

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


# ─────────────────────────────────────────────
# calculate_cost — 3 tests
# ─────────────────────────────────────────────

def test_known_model_returns_correct_value():
    """Bug #1 — if cost math is wrong, every cost in the DB is wrong."""
    cost = calculate_cost("gpt-4o-mini", 1000, 1000)
    expected = (1000 / 1000) * 0.000150 + (1000 / 1000) * 0.000600
    assert abs(cost - expected) < 1e-9


def test_unknown_model_returns_zero():
    """Bug #9 — unknown models must return 0.0, not crash."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cost = calculate_cost("totally-fake-model", 1000, 500)
    assert cost == 0.0


def test_unknown_model_emits_warning():
    """Bug #9 — silent failure means no one knows cost is wrong."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calculate_cost("totally-fake-model", 1000, 500)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "no pricing data" in str(w[0].message).lower()


# ─────────────────────────────────────────────
# LLMLogger — 6 tests
# ─────────────────────────────────────────────

def test_log_and_retrieve_row(logger):
    """Core functionality — what goes in must come out."""
    logger.log(_sample_event())
    rows = logger.query(provider="openai")
    assert len(rows) == 1
    assert rows[0]["provider"]     == "openai"
    assert rows[0]["model"]        == "gpt-4o-mini"
    assert rows[0]["input_tokens"] == 100

def test_timestamp_is_utc_aware(logger):
    """Bug #6 — datetime.utcnow() deprecated, must use timezone.utc."""
    logger.log(_sample_event())
    rows = logger.query()
    assert "+00:00" in rows[0]["timestamp"], \
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
# middleware.py — 1 test
# ─────────────────────────────────────────────

def test_invalid_provider_raises_value_error():
    """Clear error message, not a cryptic crash."""
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMWatch(provider="invalid_provider", model="gpt-4o-mini")


