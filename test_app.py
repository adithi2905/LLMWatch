import time
import os
from dotenv import load_dotenv
import openai
from prometheus_client import Histogram, Counter, start_http_server

load_dotenv()  # Load environment variables from .env file if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment!")

REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'LLM request latency',
    ['provider', 'model']
)

TOKENS_TOTAL = Counter(
    'llm_tokens_total',
    'Total tokens',
    ['provider', 'model', 'type']
)

def tracked_call(prompt: str):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    start = time.time()
    
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    duration = time.time() - start
    
    REQUEST_DURATION.labels(
        provider="openai",
        model=MODEL
    ).observe(duration)
    
    TOKENS_TOTAL.labels(
        provider="openai",
        model=MODEL,
        type="input"
    ).inc(response.usage.prompt_tokens)
    
    TOKENS_TOTAL.labels(
        provider="openai",
        model=MODEL,
        type="output"
    ).inc(response.usage.completion_tokens)
    
    return response

if __name__ == "__main__":
    start_http_server(8000)
    print(f"Metrics at http://localhost:8000/metrics")
    print(f"Using model: {MODEL}")
    
    response = tracked_call("What is supply chain disruption?")
    print(response.choices[0].message.content)
    
    input("Press Enter to exit...")