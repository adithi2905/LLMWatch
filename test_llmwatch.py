"""
test_llmwatch.py

Integration test for LLMWatch middleware.
Runs continuous multi-agent LLM calls and exposes
metrics at http://localhost:8000/metrics
"""

import time
import os
from dotenv import load_dotenv
from prometheus_client import start_http_server
from llmwatch import LLMWatch
from llmwatch.prompts import build_messages, rotate_agents
from llmwatch.metrics import record_agent_metrics

load_dotenv()

def extract_decision(response_content: str) -> str:
    content_upper = response_content.upper()
    for decision in ["EXPEDITE", "SWITCH", "WAIT"]:
        if decision in content_upper:
            return decision
    return "UNKNOWN"

if __name__ == "__main__":
    start_http_server(8000)
    print("Metrics at http://localhost:8000/metrics")

    watch = LLMWatch(
        provider="openai",
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=150,
        max_retries=3
    )

    agents = rotate_agents()
    i = 0

    while True:
        agent_name, agent = agents[i % len(agents)]
        messages = build_messages(agent, i)

        print(f"\nAgent: {agent_name}")
        print(f"Prompt: {messages[-1]['content'][:60]}...")

        response = watch.call(messages=messages)

        print(f"Response: {response['content'][:60]}...")
        print(f"Duration: {response['duration']:.2f}s")
        print(f"Cost:     ${response['cost_usd']:.6f}")

        # Record agent specific metrics
        record_agent_metrics(
            agent_name=agent_name,
            disagreement_score=0.0,
            debate_turns=1,
            decision=extract_decision(response['content']),
            confidence=0.85
        )

        i += 1
        print(f"Waiting 10 seconds... (call #{i})")
        time.sleep(10)
