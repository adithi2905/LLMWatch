"""
llmwatch/middleware.py

Core middleware class for LLMWatch.
Wraps any LLM provider call and automatically records
latency, token usage, cost forecasting, errors and retries.

Supported providers: openai, anthropic, groq

Usage:
    watch = LLMWatch(provider="openai", model="gpt-4o-mini")
    response = watch.call(messages=[{"role": "user", "content": "hello"}])
    print(response["content"])

    Or shorthand:
    answer = watch.ask("What is supply chain disruption?")

    Or as context manager:
    with LLMWatch("openai", "gpt-4o-mini") as watch:
        response = watch.call(messages=[...])
"""

import time
import os
from llmwatch.logger import LLMLogger
from llmwatch.metrics import TIME_TO_FIRST_TOKEN, record_llm_call, calculate_cost
from prometheus_client import Counter

ERRORS_TOTAL = Counter(
    'llm_errors_total',
    'Total errors by type',
    ['provider', 'model', 'error_type']
)

RETRIES_TOTAL = Counter(
    'llm_retries_total',
    'Total retry attempts',
    ['provider', 'model']
)


def _get_openai_client():
    try:
        import openai
        return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("Run: pip install openai")


def _get_anthropic_client():
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    except ImportError:
        raise ImportError("Run: pip install anthropic")


def _get_groq_client():
    try:
        from groq import Groq
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    except ImportError:
        raise ImportError("Run: pip install groq")


def _call_openai(client, model, messages, max_tokens, **kwargs):
    start = time.time()
    first_token_time = None
    content = ""
    input_tokens = 0
    output_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs
    )

    for chunk in stream:
        # first chunk with actual content → record TTFT
        if (first_token_time is None
                and chunk.choices
                and chunk.choices[0].delta.content):
            first_token_time = time.time() - start

        # accumulate content
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

        # last chunk contains usage
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens

    return {
        "content":       content,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "ttft":          first_token_time or 0.0,
        "model":         model,
        "provider":      "openai"
    }


def _call_anthropic(client, model, messages, max_tokens, **kwargs):
    start = time.time()
    first_token_time = None
    content = ""
    input_tokens = 0
    output_tokens = 0

    with client.messages.stream(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **kwargs
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.time() - start
            content += text

        # usage available after stream completes
        final = stream.get_final_message()
        input_tokens = final.usage.input_tokens
        output_tokens = final.usage.output_tokens

    return {
        "content":       content,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "ttft":          first_token_time or 0.0,
        "model":         model,
        "provider":      "anthropic"
    }


def _call_groq(client, model, messages, max_tokens, **kwargs):
    start = time.time()
    first_token_time = None
    content = ""
    input_tokens = 0
    output_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        **kwargs
    )

    for chunk in stream:
        if (first_token_time is None
                and chunk.choices
                and chunk.choices[0].delta.content):
            first_token_time = time.time() - start

        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

        if hasattr(chunk, 'x_groq') and chunk.x_groq and chunk.x_groq.usage:
            input_tokens = chunk.x_groq.usage.prompt_tokens
            output_tokens = chunk.x_groq.usage.completion_tokens

    return {
        "content":       content,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "ttft":          first_token_time or 0.0,
        "model":         model,
        "provider":      "groq"
    }


PROVIDER_MAP = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "groq":      _call_groq,
}


class LLMWatch:

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int = 500,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._logger = LLMLogger()
        if provider not in PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Choose from: {list(PROVIDER_MAP.keys())}"
            )
        self.provider    = provider
        self.model       = model
        self.max_tokens  = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._caller     = PROVIDER_MAP[provider]
        self._client     = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            return _get_openai_client()
        elif self.provider == "anthropic":
            return _get_anthropic_client()
        elif self.provider == "groq":
            return _get_groq_client()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def call(self, messages: list, **kwargs) -> dict:
        attempt = 0
        while attempt <= self.max_retries:
            try:
                return self._tracked_call(messages, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                if attempt == self.max_retries:
                    ERRORS_TOTAL.labels(
                        provider=self.provider,
                        model=self.model,
                        error_type=error_type
                    ).inc()
                    raise
                attempt += 1
                RETRIES_TOTAL.labels(
                    provider=self.provider,
                    model=self.model
                ).inc()
                print(f"LLMWatch: retrying ({attempt}/{self.max_retries}) after {error_type}")
                time.sleep(self.retry_delay * attempt)

    def _tracked_call(self, messages: list, **kwargs) -> dict:
        start = time.time()
        response = self._caller(
            self._client,
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            **kwargs
        )
        duration = time.time() - start

        # record TTFT if available
        if response.get("ttft"):
            TIME_TO_FIRST_TOKEN.labels(
                provider=self.provider,
                model=self.model
            ).observe(response["ttft"])

        cost = calculate_cost(
            self.model,
            response["input_tokens"],
            response["output_tokens"]
        )
        response["cost_usd"] = cost

        record_llm_call(
            provider=self.provider,
            model=self.model,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            duration=duration,
        )
        self._logger.log({
            "provider":      self.provider,
            "model":         self.model,
            "input_tokens":  response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "duration":      duration,
            "cost_usd":      response["cost_usd"],
            "prompt":        messages[-1]["content"],
            "response":      response["content"],
            "error":         None
        })

        response["duration"] = duration
        return response

    def ask(self, question: str, **kwargs) -> str:
        response = self.call(
            messages=[{"role": "user", "content": question}],
            **kwargs
        )
        return response["content"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False