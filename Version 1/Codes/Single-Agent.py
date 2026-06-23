
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL = "gpt-4o"
_COST_PER_1K_INPUT_TOKENS = 0.0025
_COST_PER_1K_OUTPUT_TOKENS = 0.0100


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _safe_json_load(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SingleAgentNaiveBaseline
# ---------------------------------------------------------------------------
class SingleAgentNaiveBaseline:
    """
    A minimal one-pass single-agent baseline.

    Important: this class is intentionally naive.
    It does not use tool calling, schema search, function lookup,
    validation, or self-correction.
    """

    def __init__(
        self,
        openai_api_key: str,
        meta_file_path: str,
        postgis_functions_list: list,
        tables_json_abrvs: dict,
        columns_json_abrvs: dict,
        column_values_json: dict,
        ai_descriptions: dict,
        db_connection_string: str,
        model: str = _MODEL,
    ):
        self.openai = OpenAI(api_key=openai_api_key)
        self.meta_file_path = meta_file_path
        self.postgis_fns = postgis_functions_list
        self.tables_abrvs = tables_json_abrvs
        self.columns_abrvs = columns_json_abrvs
        self.column_values = column_values_json
        self.ai_descriptions = ai_descriptions
        self.db_conn_str = db_connection_string
        self.model = model

    @staticmethod
    def _compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000 * _COST_PER_1K_INPUT_TOKENS
            + completion_tokens / 1000 * _COST_PER_1K_OUTPUT_TOKENS
        )

    def _build_light_schema(self) -> Dict[str, Any]:
        """
        Build a small schema summary.

        This is intentionally light. It keeps the prompt short and fast,
        but it also increases ambiguity because the agent does not receive
        full grounding support.
        """
        schema = {}
        for table, cols in self.columns_abrvs.items():
            schema[table] = {
                "description": self.tables_abrvs.get(table, "No description."),
                "columns": list(cols.keys()),
            }
        return schema

    def _build_system_prompt(self) -> str:
        schema_summary = json.dumps(self._build_light_schema(), indent=2)
        return f"""
You are a single agent that converts natural language into one PostgreSQL/PostGIS SQL query.

Design goal:
- Be fast.
- Be simple.
- Use one pass only.
- Do not ask follow-up questions.
- Do not use any external tools.
- Do not validate or repair the SQL.

Behavior rules:
1. Read the user question.
2. Infer the table names, columns, filters, and spatial operations directly from the schema below.
3. Generate one SQL query.
4. Return JSON only.

Important limitations of this baseline:
- If the query is ambiguous, make your best guess.
- If multiple tables or columns look similar, choose the most likely one.
- Do not run EXPLAIN.
- Do not check CRS correctness.
- Do not verify joins or aggregations after generation.
- Do not add comments outside JSON.

Use PostGIS when the question clearly needs a spatial operation such as within, intersects, distance, area, or length.
If the question does not clearly need spatial logic, generate standard SQL.

AVAILABLE LIGHT SCHEMA
{schema_summary}

OUTPUT FORMAT
{{
  "cot_trace": {{
    "identified_tables": ["..."],
    "identified_columns": ["..."],
    "assumed_spatial_operation": "...",
    "plan": ["Step 1: ...", "Step 2: ..."]
  }},
  "sql_query": "SELECT ...",
  "manifest": {{
    "assumptions": ["..."],
    "known_single_agent_limitations": [
      "semantic errors may remain",
      "schema ambiguity may remain",
      "technical mistakes may remain",
      "joins or aggregation may be incomplete",
      "query was not validated"
    ]
  }}
}}
"""

    def run(self, user_query: str) -> dict:
        """
        Execute one LLM call only.

        This preserves the intended single-agent advantages:
        low latency, lower cost, and architectural simplicity.
        """
        print(f"\n=== [SingleAgentNaive] Query: {user_query!r} ===")
        wall_start = time.perf_counter()

        messages: List[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]

        try:
            t0 = time.perf_counter()
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            t1 = time.perf_counter()
        except OpenAIError as e:
            return {
                "status": "error",
                "final_sql": "",
                "cot_trace": None,
                "manifest": None,
                "telemetry": None,
                "error": str(e),
            }

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        call_latency = t1 - t0
        total_latency = time.perf_counter() - wall_start
        estimated_cost = self._compute_cost(prompt_tokens, completion_tokens)

        raw = response.choices[0].message.content or "{}"
        payload = _safe_json_load(raw)

        if not payload or "sql_query" not in payload:
            return {
                "status": "failed",
                "final_sql": "",
                "cot_trace": None,
                "manifest": None,
                "telemetry": {
                    "total_latency_s": round(total_latency, 3),
                    "llm_latency_s": round(call_latency, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "llm_calls": 1,
                    "calls_breakdown": [
                        {
                            "iteration": 0,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "cost_usd": round(estimated_cost, 6),
                            "latency_s": round(call_latency, 3),
                        }
                    ],
                },
                "error": "Model did not return the expected JSON format.",
            }

        sql = payload.get("sql_query", "")
        cot_trace = payload.get("cot_trace", {})
        manifest = payload.get("manifest", {})

        # Intentionally no validation here.
        # This preserves the baseline weakness: lack of validation.

        print(f"[SingleAgentNaive] Done. SQL preview: {sql[:100]}...")
        return {
            "status": "generated",
            "final_sql": sql,
            "cot_trace": cot_trace,
            "manifest": manifest,
            "telemetry": {
                "total_latency_s": round(total_latency, 3),
                "llm_latency_s": round(call_latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 6),
                "llm_calls": 1,
                "calls_breakdown": [
                    {
                        "iteration": 0,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "cost_usd": round(estimated_cost, 6),
                        "latency_s": round(call_latency, 3),
                    }
                ],
            },
        }


# ---------------------------------------------------------------------------
# Drop-in replacement
# ---------------------------------------------------------------------------
def run_single_agent(
    user_query: str,
    database_name: str,
    meta_file_path: str,
    postgis_functions: list,
    tables_abrvs: dict,
    columns_abrvs: dict,
    column_values: dict,
    ai_descriptions: dict,
    openai_api_key: str,
    gemini_api_key: str,
    db_connection_string: str = None,
) -> dict:
    """
    Drop-in replacement for the original function.

    Notes:
    - gemini_api_key is kept only for interface compatibility.
    - postgis_functions is kept only for interface compatibility.
    - No validator is used in this baseline.
    """
    if db_connection_string is None:
        db_connection_string = (
            f"postgresql+psycopg2://user:password@localhost:5432/{database_name}"
        )

    agent = SingleAgentNaiveBaseline(
        openai_api_key=openai_api_key,
        meta_file_path=meta_file_path,
        postgis_functions_list=postgis_functions,
        tables_json_abrvs=tables_abrvs,
        columns_json_abrvs=columns_abrvs,
        column_values_json=column_values,
        ai_descriptions=ai_descriptions,
        db_connection_string=db_connection_string,
    )

    result = agent.run(user_query)
    return {
        "status": result["status"],
        "final_sql": result["final_sql"],
        "intermediate": {
            "cot_trace": result.get("cot_trace"),
            "manifest": result.get("manifest"),
        },
        "telemetry": result.get("telemetry"),
    }
