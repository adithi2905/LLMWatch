"""
llmwatch/bus.py

Lightweight, in-process pub/sub message bus.

Two built-in topics mirror a Kafka-style design:
  TOPIC_REQUESTS = "llm.requests"   — raw events from middleware
  TOPIC_METRICS  = "llm.metrics"    — derived events from processor

Consumers run in daemon threads so they never block the caller.

Usage
-----
    from llmwatch.bus import get_bus, TOPIC_REQUESTS, TOPIC_METRICS

    bus = get_bus()

    bus.subscribe(TOPIC_METRICS, my_dashboard_handler)
    bus.publish(TOPIC_REQUESTS, raw_event)
"""

from __future__ import annotations
import queue
import threading
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

TOPIC_REQUESTS = "llm.requests"
TOPIC_METRICS  = "llm.metrics"

class _TopicWorker:

    def __init__(self, topic: str):
        self.topic     = topic
        self._queue    = queue.Queue()
        self._handlers: list[Callable[[Any], None]] = []
        self._thread   = threading.Thread(
            target=self._run,
            name=f"llmwatch-bus-{topic}",
            daemon=True,
        )
        self._thread.start()

    def subscribe(self, handler: Callable[[Any], None]) -> None:
        self._handlers.append(handler)

    def publish(self, item: Any) -> None:
        self._queue.put_nowait(item)

    def _run(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            for handler in self._handlers:
                try:
                    handler(event)
                except Exception as exc:
                    log.exception(
                        "llmwatch bus: handler %s on topic %s raised %s",
                        handler.__qualname__, self.topic, exc,
                    )
            self._queue.task_done()


class MessageBus:

    def __init__(self):
        self._workers: dict[str, _TopicWorker] = {}
        self._lock    = threading.Lock()

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._worker(topic).subscribe(handler)

    def publish(self, topic: str, item: Any) -> None:
        self._worker(topic).publish(item)

    def join(self, topic: str, timeout: float = 5.0) -> None:
        if topic not in self._workers:
            raise RuntimeError(f"No such topic: {topic}")
        self._workers[topic]._queue.join()

    def _worker(self, topic: str) -> _TopicWorker:
        if topic not in self._workers:
            with self._lock:
                if topic not in self._workers:
                    self._workers[topic] = _TopicWorker(topic)
        return self._workers[topic]


_bus: MessageBus | None = None
_bus_lock = threading.Lock()

def get_bus(reset: bool = False) -> MessageBus:
    global _bus
    with _bus_lock:
        if _bus is None or reset:
            _bus = MessageBus()
    return _bus