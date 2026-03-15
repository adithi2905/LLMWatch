"""
llmwatch/logger.py

Structured SQLite logger for LLMWatch.
Records every LLM call to a local database
for historical analysis and auditing.

Usage:
    from llmwatch.logger import LLMLogger

    logger = LLMLogger()
    logger.log({
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "input_tokens":  13,
        "output_tokens": 100,
        "duration":      3.57,
        "cost_usd":      0.000072,
        "error":         None,
        "prompt":        "What is supply chain?",
        "response":      "Supply chain is..."
    })

    rows = logger.query(provider="openai", limit=10)
    summary = logger.summary()
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.getenv("LLMWATCH_DB_PATH", "llmwatch.db")


CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS llm_calls (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        provider        TEXT    NOT NULL,
        model           TEXT    NOT NULL,
        input_tokens    INTEGER NOT NULL,
        output_tokens   INTEGER NOT NULL,
        duration_ms     REAL    NOT NULL,
        cost_usd        REAL    NOT NULL,
        prompt          TEXT,
        response        TEXT,
        error           TEXT
    )
"""


class LLMLogger:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def log(self, event: dict):
        """
        Log a single LLM call event.

        Required keys:
            provider, model, input_tokens,
            output_tokens, duration, cost_usd

        Optional keys:
            prompt, response, error
        """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO llm_calls (
                    timestamp,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    duration_ms,
                    cost_usd,
                    prompt,
                    response,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                event.get("provider"),
                event.get("model"),
                event.get("input_tokens", 0),
                event.get("output_tokens", 0),
                round(event.get("duration", 0) * 1000, 2),
                event.get("cost_usd", 0.0),
                event.get("prompt"),
                event.get("response"),
                event.get("error"),
            ))

    def query(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Query recent logs with optional filters.

        Args:
            provider: filter by provider name
            model:    filter by model name
            limit:    max rows to return

        Returns:
            list of dicts, newest first
        """
        sql    = "SELECT * FROM llm_calls WHERE 1=1"
        params = []

        if provider:
            sql += " AND provider = ?"
            params.append(provider)

        if model:
            sql += " AND model = ?"
            params.append(model)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def summary(self) -> dict:
        """
        Return aggregate stats across all logged calls.

        Returns dict with:
            total_calls, total_cost_usd,
            avg_duration_ms, avg_input_tokens,
            avg_output_tokens, total_errors
        """
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                    AS total_calls,
                    ROUND(SUM(cost_usd), 6)     AS total_cost_usd,
                    ROUND(AVG(duration_ms), 2)  AS avg_duration_ms,
                    ROUND(AVG(input_tokens), 1) AS avg_input_tokens,
                    ROUND(AVG(output_tokens),1) AS avg_output_tokens,
                    SUM(CASE WHEN error IS NOT NULL
                        THEN 1 ELSE 0 END)      AS total_errors
                FROM llm_calls
            """).fetchone()

            return {
                "total_calls":       row[0],
                "total_cost_usd":    row[1],
                "avg_duration_ms":   row[2],
                "avg_input_tokens":  row[3],
                "avg_output_tokens": row[4],
                "total_errors":      row[5],
            }

    def clear(self):
        """Delete all logs — useful for testing"""
        with self._connect() as conn:
            conn.execute("DELETE FROM llm_calls")