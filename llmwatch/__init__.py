"""
llmwatch/__init__.py

LLMWatch — Generic LLM Observability with Prometheus.
Drop-in middleware for tracking latency, cost, and
multi-agent metrics across any LLM provider.

Supported providers: openai, anthropic, groq

Install:
    pip install llmwatch

Quickstart:
    from llmwatch import LLMWatch

    watch = LLMWatch(provider="openai", model="gpt-4o-mini")
    response = watch.call(
        messages=[{"role": "user", "content": "hello"}]
    )
    print(response["content"])
"""

from llmwatch.middleware import LLMWatch
from llmwatch.metrics import (
    record_llm_call,
    record_agent_metrics,
    calculate_cost,
)

__version__ = "0.1.0"
__author__  = "Adithi Varadarajan"
__license__ = "MIT"

__all__ = [
    "LLMWatch",
    "record_llm_call",
    "record_agent_metrics",
    "calculate_cost",
]