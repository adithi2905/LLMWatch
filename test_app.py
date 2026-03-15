import time
import os
from dotenv import load_dotenv
import openai
from prometheus_client import Histogram, Counter, start_http_server

load_dotenv()  # Load environment variables from .env file if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

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

# List of test prompts
PROMPTS = [
    "What is supply chain disruption?",
    "How do manufacturers handle supplier delays?",
    "What is safety stock in inventory management?",
    "How does port congestion affect supply chains?",
    "What is a just-in-time inventory system?",
]

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
    print("Making calls every 10 seconds...")
    
    i = 0
    while True:
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"\nCalling: {prompt}")
        
        response = tracked_call(prompt)
        print(f"Response: {response.choices[0].message.content[:50]}...")
        
        i += 1
        print(f"Waiting 10 seconds... (call #{i})")
        time.sleep(10)  # wait 10 seconds between calls