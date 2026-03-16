```markdown
# LLMWatch

Generic LLM observability middleware for Python.
Track latency, cost, and multi-agent metrics across
any LLM provider with Prometheus and Grafana.

---

## What it does

LLMWatch wraps any LLM API call and automatically records:

- **Latency** — request duration, p95, p99 percentiles
- **Cost** — actual spend + hourly/daily/monthly forecasting
- **Tokens** — input and output token usage per provider
- **Reliability** — errors, retries, success rate
- **Multi-agent** — disagreement scores, debate turns, decision distribution

---

## Supported providers

| Provider  | Models                        |
|-----------|-------------------------------|
| OpenAI    | gpt-4o, gpt-4o-mini, o1       |
| Anthropic | claude-sonnet-4-6, claude-opus-4-6 |
| Groq      | llama, mixtral                |

---

## Quickstart

### 1. Install

```bash
pip install llmwatch
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY=your_key_here
export LLM_MODEL=gpt-4o-mini
```

Or create a `.env` file:

```
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
```

### 3. Wrap your LLM calls

```python
from llmwatch import LLMWatch

watch = LLMWatch(
    provider="openai",
    model="gpt-4o-mini"
)

response = watch.call(
    messages=[{"role": "user", "content": "What is supply chain disruption?"}]
)

print(response["content"])
print(f"Cost: ${response['cost_usd']:.6f}")
print(f"Duration: {response['duration']:.2f}s")
```

Or use the shorthand:

```python
answer = watch.ask("What is supply chain disruption?")
print(answer)
```

Or as a context manager:

```python
with LLMWatch("openai", "gpt-4o-mini") as watch:
    response = watch.call(messages=[...])
```

### 4. Start Prometheus and Grafana

```bash
docker-compose up
```

### 5. Expose metrics

```python
from prometheus_client import start_http_server
start_http_server(8000)
```

Metrics available at `http://localhost:8000/metrics`

### 6. Import Grafana dashboard

1. Open `http://localhost:3000`
2. Click **+** → **Import**
3. Upload `dashboards/llmwatch.json`
4. Select Prometheus datasource
5. Click **Import**

Full observability in under 5 minutes.

---

## Multi-agent support

LLMWatch has first-class support for multi-agent systems.
Track disagreement scores, debate turns, and decision
distribution across agents:

```python
from llmwatch.metrics import record_agent_metrics

record_agent_metrics(
    agent_name="supply_chain",
    disagreement_score=0.42,
    debate_turns=3,
    decision="WAIT",
    confidence=0.86
)
```

---

## Agent-style prompts

LLMWatch ships with realistic agent prompt templates
for testing multi-agent pipelines:

```python
from llmwatch.prompts import build_messages, rotate_agents
from llmwatch import LLMWatch

watch = LLMWatch(provider="openai", model="gpt-4o-mini")
agents = rotate_agents()

for agent_name, agent in agents:
    messages = build_messages(agent, prompt_index=0)
    response = watch.call(messages=messages)
    print(f"{agent_name}: {response['content']}")
```

---

## Metrics reference

### Latency

| Metric | Type | Description |
|--------|------|-------------|
| `llm_request_duration_seconds` | Histogram | Total request latency |
| `llm_time_to_first_token_seconds` | Histogram | Time to first token |

### Cost

| Metric | Type | Description |
|--------|------|-------------|
| `llm_input_tokens_total` | Counter | Total input tokens |
| `llm_output_tokens_total` | Counter | Total output tokens |
| `llm_actual_cost_usd_total` | Counter | Cumulative cost in USD |
| `llm_cost_per_minute_usd` | Gauge | Current spend rate |
| `llm_projected_hourly_cost_usd` | Gauge | Projected hourly cost |
| `llm_projected_daily_cost_usd` | Gauge | Projected daily cost |
| `llm_projected_monthly_cost_usd` | Gauge | Projected monthly cost |

### Reliability

| Metric | Type | Description |
|--------|------|-------------|
| `llm_errors_total` | Counter | Errors by type |
| `llm_retries_total` | Counter | Total retry attempts |

### Multi-agent

| Metric | Type | Description |
|--------|------|-------------|
| `llm_agent_disagreement_score` | Histogram | Agent disagreement (0–1) |
| `llm_agent_debate_turns` | Histogram | Turns before decision |
| `llm_orchestrator_decisions_total` | Counter | WAIT / EXPEDITE / SWITCH |
| `llm_agent_confidence_score` | Histogram | Per-agent confidence |

---

## Project structure

```
llmwatch/
├── llmwatch/
│   ├── __init__.py       — package entry point
│   ├── middleware.py     — core LLMWatch class
│   ├── metrics.py        — prometheus metrics + forecasting
│   ├── logger.py         — sqlite structured logging
│   └── prompts.py        — agent prompt templates
├── dashboards/
│   └── llmwatch.json     — grafana dashboard (import ready)
├── test_llmwatch.py      — integration test
├── prometheus.yml        — prometheus scrape config
├── docker-compose.yml    — prometheus + grafana setup
├── pyproject.toml        — package config
└── .env                  — environment variables
```

---

## Cost pricing reference

Pricing used for cost estimation (per 1k tokens):

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $0.0025 | $0.010 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| claude-sonnet-4-6 | $0.003 | $0.015 |
| claude-opus-4-6 | $0.015 | $0.075 |

---

## Requirements

```
python >= 3.10
prometheus-client
openai
anthropic
groq
python-dotenv
```

---

## License

MIT — built by Adithi Varadarajan
```
