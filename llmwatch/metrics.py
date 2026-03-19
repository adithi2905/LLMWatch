# llmwatch/metrics.py
# Prometheus metrics for LLM observability
# Tracks latency, cost (actual + forecasted), and multi-agent behavior
# Records are updated via record_llm_call() and record_agent_metrics()

import time
import warnings
from prometheus_client import Counter, Histogram, Gauge
import os
import json

REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'Total request latency',
    ['provider', 'model']
)

TIME_TO_FIRST_TOKEN = Histogram(
    'llm_time_to_first_token_seconds',
    'Time until first token received',
    ['provider', 'model']
)

INPUT_TOKENS = Counter(
    'llm_input_tokens_total',
    'Total input tokens used',
    ['provider', 'model']
)

OUTPUT_TOKENS = Counter(
    'llm_output_tokens_total',
    'Total output tokens used',
    ['provider', 'model']
)

ACTUAL_COST_USD = Counter(
    'llm_actual_cost_usd_total',
    'Actual cumulative cost in USD',
    ['provider', 'model']
)

PROJECTED_HOURLY_COST = Gauge(
    'llm_projected_hourly_cost_usd',
    'Projected cost for current hour',
    ['provider', 'model']
)

PROJECTED_DAILY_COST = Gauge(
    'llm_projected_daily_cost_usd',
    'Projected cost for current day',
    ['provider', 'model']
)

PROJECTED_MONTHLY_COST = Gauge(
    'llm_projected_monthly_cost_usd',
    'Projected cost for current month',
    ['provider', 'model']
)

COST_PER_MINUTE = Gauge(
    'llm_cost_per_minute_usd',
    'Current spend rate per minute',
    ['provider', 'model']
)

AGENT_DISAGREEMENT = Histogram(
    'llm_agent_disagreement_score',
    'How much agents disagree (0=consensus, 1=max)',
    ['agent_name']
)

DEBATE_TURNS = Histogram(
    'llm_agent_debate_turns',
    'Number of debate turns before decision',
    ['agent_name']
)

ORCHESTRATOR_DECISIONS = Counter(
    'llm_orchestrator_decisions_total',
    'Distribution of orchestrator decisions',
    ['decision']
)

AGENT_CONFIDENCE = Histogram(
    'llm_agent_confidence_score',
    'Per agent certainty score',
    ['agent_name']
)
# built-in defaults — always available
COST_PER_1K_TOKENS = {
    "gpt-4o":           {"input": 0.0025,   "output": 0.010},
    "gpt-4o-mini":      {"input": 0.000150, "output": 0.000600},
    "claude-sonnet-4-6":{"input": 0.003,    "output": 0.015},
    "claude-opus-4-6":  {"input": 0.015,    "output": 0.075}
}

def _load_pricing() -> dict:
    custom = os.getenv("LLMWATCH_PRICING_PATH")
    if custom and os.path.exists(custom):
        with open(custom) as f:
            return json.load(f)
    return COST_PER_1K_TOKENS # fallback mechanism

PRICING = _load_pricing()

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        warnings.warn(  
            f"LLMWatch: no pricing data for model '{model}'. "
            f"Cost will be recorded as $0.00. "
            f"Add it to PRICING in metrics.py.",
            UserWarning,
            stacklevel=2
        )
        return 0.0
    rates = PRICING[model]
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


class CostForecaster:

    def __init__(self, window_minutes: int = 10):
        self.window_minutes = window_minutes
        self._cost_log: list[tuple[float, float]] = []
        self._start_time = time.time()

    def record(self, cost: float):
        now = time.time()
        self._cost_log.append((now, cost))
        self._cleanup(now)

    def _cleanup(self, now: float):
        cutoff = now - (self.window_minutes * 60)
        self._cost_log = [(ts, c) for ts, c in self._cost_log if ts >= cutoff]

    def spend_rate_per_minute(self) -> float:
        if len(self._cost_log) < 2:
            return 0.0
        now = time.time()
        self._cleanup(now)
        if not self._cost_log:
            return 0.0
        total_cost = sum(c for _, c in self._cost_log)
        elapsed_minutes = (now - self._cost_log[0][0]) / 60
        if elapsed_minutes < 0.01:
            return 0.0
        return total_cost / elapsed_minutes

    def projected_hourly(self) -> float:
        return self.spend_rate_per_minute() * 60

    def projected_daily(self) -> float:
        return self.spend_rate_per_minute() * 60 * 24

    def projected_monthly(self) -> float:
        return self.spend_rate_per_minute() * 60 * 24 * 30


_forecasters: dict[str, CostForecaster] = {}

def _get_forecaster(provider: str, model: str) -> CostForecaster:
    key = f"{provider}:{model}"
    if key not in _forecasters:
        _forecasters[key] = CostForecaster(window_minutes=10)
    return _forecasters[key]


def record_llm_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration: float,
):
    REQUEST_DURATION.labels(provider=provider, model=model).observe(duration)
    INPUT_TOKENS.labels(provider=provider, model=model).inc(input_tokens)
    OUTPUT_TOKENS.labels(provider=provider, model=model).inc(output_tokens)

    cost = calculate_cost(model, input_tokens, output_tokens)
    ACTUAL_COST_USD.labels(provider=provider, model=model).inc(cost)

    forecaster = _get_forecaster(provider, model)
    forecaster.record(cost)

    COST_PER_MINUTE.labels(provider=provider, model=model).set(forecaster.spend_rate_per_minute())
    PROJECTED_HOURLY_COST.labels(provider=provider, model=model).set(forecaster.projected_hourly())
    PROJECTED_DAILY_COST.labels(provider=provider, model=model).set(forecaster.projected_daily())
    PROJECTED_MONTHLY_COST.labels(provider=provider, model=model).set(forecaster.projected_monthly())


def record_agent_metrics(
    agent_name: str,
    disagreement_score: float,
    debate_turns: int,
    decision: str,
    confidence: float = 0.0
):
    AGENT_DISAGREEMENT.labels(agent_name=agent_name).observe(disagreement_score)
    DEBATE_TURNS.labels(agent_name=agent_name).observe(debate_turns)
    ORCHESTRATOR_DECISIONS.labels(decision=decision).inc()
    AGENT_CONFIDENCE.labels(agent_name=agent_name).observe(confidence)