import json
import time
import requests
import dateparser
from typing import Dict, Any, List, Optional
import numpy as np
import pickle
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI, OpenAIError

OpenAIModel = "gpt-4o"

# -----------------------------------------------------------------------------
# METRICS COLLECTOR
# -----------------------------------------------------------------------------
class Metrics:
    """Simple accumulator for OpenAI token usage and total time."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def total_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

# -----------------------------------------------------------------------------
# 1. ENTITY EXTRACTION AGENT (EEA)
# -----------------------------------------------------------------------------
class EntityExtractionAgent:
    """
    Extracts semantic entities from a natural language query.
    Types: location, timeframe, table_reference, filter_condition, general.
    """
    SYSTEM_PROMPT = """
You are an expert Data Extraction Agent. Parse the natural language query and
extract ONLY the key entities required to query a database. Place all entities
in a single 'entities' array.

TYPE DEFINITIONS:
- "table_reference": user explicitly names a database table
  (e.g., "roads", "counties", "poi", "states", "blockgroups", "ghcn", "tracts").

- "filter_condition": ANY value that will be used in a WHERE clause filter,
  including:
    * Comparative expressions  : "> 50", "< 100", ">= 2020"
    * Exact numeric IDs        : "360036", "42", "= 5"
    * Exact string matches     : "'Nevada'", "= 'primary'"
  Rule: if a number or string in the query identifies a specific row or sets
  a threshold, it is ALWAYS filter_condition, never "general".

- "location": a named place needed ONLY for SPATIAL GEOMETRY operations
  (e.g., finding all roads within a city boundary using ST_Intersects).
  Do NOT use for place names that are simply text filters against a table
  column (e.g., WHERE name = 'Nevada').
  CRITICAL: if a table_reference is already present, any proper noun
  identifying a row in that table is "general", NOT "location".

- "timeframe": a date or time range expression.

- "general": column concepts and keywords useful for schema search.
  This includes:
    * Column concepts   : "name", "geometry", "speed limit", "length"
    * Proper nouns that are row-filter values when a table_reference exists
    * Unit specifications: "meters", "km", "mph", "square kilometers"
      (include these ONLY when they provide essential SQL context, e.g., to
      choose between ST_Length vs ST_Length::geography; otherwise omit them)
  Do NOT include question words, filler, or implied operations.

EXAMPLES:

User: "What is the length of the road segment with the gid of 360036, in meters?"
Output: {
    "status": "success",
    "clarification_message": "",
    "entities": [
        {"value": "roads",   "type": "table_reference"},
        {"value": "gid",     "type": "general"},
        {"value": "360036",  "type": "filter_condition"},
        {"value": "length",  "type": "general"},
        {"value": "meters",  "type": "general"}
    ]
}

User: "List the names of all roads with a maximum speed limit greater than 50 mph."
Output: {
    "status": "success",
    "clarification_message": "",
    "entities": [
        {"value": "roads",       "type": "table_reference"},
        {"value": "name",        "type": "general"},
        {"value": "speed limit", "type": "general"},
        {"value": "> 50",        "type": "filter_condition"}
    ]
}

User: "Show the WGS 84 geometry for the state of Nevada."
Output: {
    "status": "success",
    "clarification_message": "",
    "entities": [
        {"value": "states",   "type": "table_reference"},
        {"value": "Nevada",   "type": "general"},
        {"value": "geometry", "type": "general"},
        {"value": "WGS 84",   "type": "general"}
    ]
}

User: "How many hospitals are in Boston, MA built after 2020?"
Output: {
    "status": "success",
    "clarification_message": "",
    "entities": [
        {"value": "hospitals",  "type": "general"},
        {"value": "Boston, MA", "type": "location"},
        {"value": "after 2020", "type": "timeframe"}
    ]
}

User: "Find all poi within Sequoia National Park."
Output: {
    "status": "success",
    "clarification_message": "",
    "entities": [
        {"value": "poi",                   "type": "table_reference"},
        {"value": "ne_protected_areas",    "type": "table_reference"},
        {"value": "Sequoia National Park", "type": "general"}
    ]
}

Return ONLY a valid JSON object matching this schema.
"""

    def __init__(self, openai_api_key: str, metrics: Metrics):
        self.client = OpenAI(api_key=openai_api_key)
        self.nominatim_headers = {
            "User-Agent": "SimpleSpatialAgent/1.0 (pipeline_builder@gmail.com)"
        }
        self.metrics = metrics

    def _validate_location(self, entity: str) -> Dict[str, Any]:
        """Validates locations using OpenStreetMap Nominatim."""
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={entity.replace(' ', '+')}&format=json"
        )
        try:
            resp = requests.get(url, headers=self.nominatim_headers, timeout=10)
            resp.raise_for_status()
            results = resp.json()
            time.sleep(1)
            if not results:
                return {"status": "invalid"}
            unique = list(dict.fromkeys(r.get("display_name") for r in results))
            if len(unique) > 1:
                top = results[0].get("importance", 0)
                sec = results[1].get("importance", 0)
                if top > 0.6 and (top - sec) > 0.2:
                    return {"status": "valid"}
                return {"status": "ambiguous", "options": unique[:3]}
            return {"status": "valid"}
        except requests.exceptions.RequestException as e:
            print(f"  [EEA Warning] Nominatim error: {e}. Failing open.")
            return {"status": "valid"}

    def _parse_timeframe(self, time_str: str) -> Optional[str]:
        """Validates and standardizes timeframes using dateparser."""
        settings = {"PREFER_DAY_OF_MONTH": "first", "PREFER_MONTH_OF_YEAR": "first"}
        parsed   = dateparser.parse(time_str, settings=settings)
        return parsed.isoformat() if parsed else None

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Extract entities and validate location/timeframe values."""
        try:
            response = self.client.chat.completions.create(
                model=OpenAIModel,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_query},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            # Record token usage
            usage = response.usage
            self.metrics.add_usage(usage.prompt_tokens, usage.completion_tokens)

            payload = json.loads(response.choices[0].message.content)
        except (OpenAIError, json.JSONDecodeError) as e:
            return {
                "status":                "needs_clarification",
                "clarification_message": f"System error: {e}.",
                "entities":              [],
            }

        issues, valid = [], []
        for ent in payload.get("entities", []):
            v, t = ent.get("value", ""), ent.get("type", "general")
            if t == "location":
                val = self._validate_location(v)
                if val["status"] == "ambiguous":
                    opts = " or ".join(f'"{o}"' for o in val["options"])
                    issues.append(f"Multiple places named '{v}'. Did you mean {opts}?")
                elif val["status"] == "invalid":
                    issues.append(f"Couldn't locate '{v}'. Could you clarify?")
                else:
                    valid.append({"value": v, "type": "location"})
            elif t == "timeframe":
                iso = self._parse_timeframe(v)
                if iso:
                    valid.append({"value": iso, "type": "timeframe"})
                else:
                    issues.append(
                        f"Couldn't parse timeframe '{v}'. Please give a specific date."
                    )
            elif t in ("table_reference", "filter_condition"):
                valid.append({"value": v, "type": t})
            else:
                valid.append({"value": v, "type": "general"})

        payload["status"]                = "needs_clarification" if issues else "success"
        payload["clarification_message"] = " ".join(issues)
        payload["entities"]              = valid
        return payload


# -----------------------------------------------------------------------------
# 2. METADATA RETRIEVAL AGENT (MRA)
# -----------------------------------------------------------------------------
class MetadataRetrievalAgent:
    """
    Retrieves and enriches database metadata.
    """
    GEOM_COLUMN_NAMES = {"geom", "geometry", "the_geom", "wkb_geometry", "shape"}

    def __init__(
        self,
        openai_api_key: str,
        meta_file_path: str,
        postgis_functions_list: list,
        tables_json_abrvs: dict,
        columns_json_abrvs: dict,
        column_values_json: dict,
        ai_descriptions: dict,
        metrics: Metrics,
    ):
        self.openai_client          = OpenAI(api_key=openai_api_key)
        self.meta_file_path         = meta_file_path
        self.postgis_functions_list = postgis_functions_list
        self.tables_json_abrvs      = tables_json_abrvs
        self.columns_json_abrvs     = columns_json_abrvs
        self.column_values_json     = column_values_json
        self.ai_descriptions        = ai_descriptions
        self.metrics                = metrics

    # ------------------------------------------------------------------ tools

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return a unit-norm embedding vector or None on error using OpenAI text-embedding-3-large."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            self.metrics.add_usage(response.usage.prompt_tokens, 0)
            
            vec = np.array(response.data[0].embedding, dtype=float)
            n   = np.linalg.norm(vec)
            return vec / n if n > 0 else None
        except Exception as e:
            print(f"  [MRA Warning] Embedding error for '{text}': {e}")
            return None

    def _search_columns(self, query_text: str, top_k: int = 10) -> str:
        """
        Global semantic search across all table columns.
        top_k=10 so forced-table columns compete against other tables' columns.
        """
        print(f"  [MRA Tool] Searching columns for: '{query_text}'")
        q_emb = self._embed(query_text)
        if q_emb is None:
            return json.dumps([])

        with open(self.meta_file_path, "rb") as f:
            meta = pickle.load(f)
        saved = meta.get("embeddings", {})

        results = []
        for table, cols in saved.items():
            for col, emb_vals in cols.items():
                c_emb = np.array(emb_vals, dtype=float)
                n     = np.linalg.norm(c_emb)
                if n == 0:
                    continue
                score = float(np.dot(q_emb, c_emb / n))
                results.append({"table": table, "column": col, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps(results[:top_k])

    def _search_spatial_functions(self, query_text: str, top_k: int = 3) -> str:
        """Semantic search for PostGIS functions."""
        print(f"  [MRA Tool] Searching PostGIS functions for: '{query_text}'")
        q_emb = self._embed(query_text)
        if q_emb is None:
            return json.dumps([])

        results = []
        for item in self.postgis_functions_list:
            emb_vals = item.get("embedding")
            if not emb_vals:
                continue
            c_emb = np.array(emb_vals, dtype=float)
            n     = np.linalg.norm(c_emb)
            if n == 0:
                continue
            score      = float(np.dot(q_emb, c_emb / n))
            r          = {k: v for k, v in item.items() if k != "embedding"}
            r["score"] = score
            results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps(results[:top_k])

    def _targeted_table_search(
        self, table: str, query_texts: List[str], top_k_per_query: int = 3
    ) -> List[dict]:
        """
        Search ALL columns of a specific table for each query_text.
        Used when the global search misses a forced table entirely, or when
        a required geometry column is absent (FIX MRA-2).
        """
        print(f"  [MRA Targeted] Searching table '{table}' for: {query_texts}")
        with open(self.meta_file_path, "rb") as f:
            meta = pickle.load(f)
        saved = meta.get("embeddings", {})

        if table not in saved:
            print(f"  [MRA Targeted] Table '{table}' has no embeddings.")
            return []

        table_cols        = saved[table]
        best: Dict[str, dict] = {}

        for query_text in query_texts:
            q_emb = self._embed(query_text)
            if q_emb is None:
                continue

            scores = []
            for col, emb_vals in table_cols.items():
                c_emb = np.array(emb_vals, dtype=float)
                n     = np.linalg.norm(c_emb)
                if n == 0:
                    continue
                score = float(np.dot(q_emb, c_emb / n))
                scores.append({"table": table, "column": col, "score": score})

            scores.sort(key=lambda x: x["score"], reverse=True)
            for hit in scores[:top_k_per_query]:
                col = hit["column"]
                if col not in best or hit["score"] > best[col]["score"]:
                    best[col] = hit

        results = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        print(
            f"  [MRA Targeted] Found {len(results)} candidate(s) in '{table}': "
            + ", ".join(f"{r['column']}({r['score']:.3f})" for r in results[:5])
        )
        return results

    # ---------------------------------------------------------------- helpers

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract the first balanced JSON object from arbitrary text."""
        brace_stack, start = [], -1
        for i, ch in enumerate(text):
            if ch == "{":
                if not brace_stack:
                    start = i
                brace_stack.append(ch)
            elif ch == "}" and brace_stack:
                brace_stack.pop()
                if not brace_stack and start != -1:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
        cleaned = text.strip().lstrip("`").rstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def _validate_and_correct_columns(self, mapped_json: dict) -> dict:
        """
        For each selected column:
          - Exists in real schema → keep.
          - Not found → embedding match (threshold 0.5) → replace.
          - No good match → mark uncertain with actual column list.
        """
        corrected                     = mapped_json.copy()
        corrected["selected_columns"] = []

        with open(self.meta_file_path, "rb") as f:
            meta = pickle.load(f)
        saved = meta.get("embeddings", {})

        for col_info in mapped_json.get("selected_columns", []):
            table  = col_info["table"]
            col    = col_info["column"]
            reason = col_info.get("reason", "")

            if (
                table in self.columns_json_abrvs
                and col in self.columns_json_abrvs[table]
            ):
                corrected["selected_columns"].append(col_info)
                continue

            print(
                f"  [MRA Validation] '{col}' not in '{table}'. "
                "Searching best match..."
            )
            best_match, best_score = None, -1.0
            actual_cols            = list(
                self.columns_json_abrvs.get(table, {}).keys()
            )

            if table in saved:
                q_emb = self._embed(col)
                if q_emb is not None:
                    for actual_col, emb_vals in saved[table].items():
                        c_emb = np.array(emb_vals, dtype=float)
                        n     = np.linalg.norm(c_emb)
                        if n == 0:
                            continue
                        score = float(np.dot(q_emb, c_emb / n))
                        if score > best_score:
                            best_score = score
                            best_match = actual_col

            if best_match and best_score > 0.5:
                print(f"      '{col}' → '{best_match}' (score={best_score:.3f})")
                corrected["selected_columns"].append({
                    "table":  table,
                    "column": best_match,
                    "reason": reason + f" (corrected from '{col}')",
                })
            else:
                actual_list = ", ".join(actual_cols) if actual_cols else "none"
                print(f"      No match for '{col}'. Marking uncertain.")
                corrected["selected_columns"].append({
                    "table":     table,
                    "column":    col,
                    "reason":    reason + f" (uncertain; actual cols: {actual_list})",
                    "uncertain": True,
                })

        return corrected

    def _build_fallback_columns(
        self,
        tool_search_results: list,
        selected_tables: list,
        entities: list,
    ) -> list:
        """Build column candidates from accumulated tool search results."""
        if not tool_search_results:
            return []

        tables_set    = set(selected_tables)
        keyword_hints = set()
        for e in entities:
            if e.get("type") in ("general", "filter_condition"):
                keyword_hints.update(e["value"].lower().split())

        best: Dict[tuple, dict] = {}
        for r in tool_search_results:
            if r["table"] not in tables_set or r["score"] < 0.3:
                continue
            key = (r["table"], r["column"])
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = r

        def sort_key(item):
            overlap = len(set(item["column"].lower().split("_")) & keyword_hints)
            return (overlap, item["score"])

        per_table: Dict[str, int] = {}
        fallback: list = []
        for r in sorted(best.values(), key=sort_key, reverse=True):
            t = r["table"]
            if per_table.get(t, 0) >= 5:
                continue
            per_table[t] = per_table.get(t, 0) + 1
            fallback.append({
                "table":  t,
                "column": r["column"],
                "reason": f"Fallback from global search (score={r['score']:.3f}).",
            })

        print(f"  [MRA Fallback] Global fallback produced {len(fallback)} column(s).")
        return fallback

    def _table_missing_geom(self, table: str, mapped_columns: list) -> bool:
        """
        FIX MRA-2 helper: returns True if the table has no geometry column
        among its currently mapped columns.
        """
        cols_for_table = {
            c["column"].lower()
            for c in mapped_columns
            if c.get("table") == table
        }
        return not cols_for_table.intersection(self.GEOM_COLUMN_NAMES)

    # ---------------------------------------------------------------- main loop

    def process(
        self, user_question: str, entities: list, spatial_function_req: bool
    ) -> dict:
        """Run the LLM agentic loop, then apply post-loop guards."""

        forced_tables = [
            e["value"] for e in entities
            if e.get("type") == "table_reference"
            and e["value"] in self.tables_json_abrvs
        ]
        filter_conditions = [
            e["value"] for e in entities if e.get("type") == "filter_condition"
        ]
        general_entities = [
            e["value"] for e in entities if e.get("type") == "general"
        ]

        # FIX MRA-1: exclude pure numeric filter values from search queries
        searchable_general = [
            v for v in general_entities
            if not v.strip().lstrip("-").replace(".", "", 1).isdigit()
        ]

        # Early-exit guard: check embedding coverage for forced tables
        with open(self.meta_file_path, "rb") as f:
            _meta_check = pickle.load(f)
        _saved_check      = _meta_check.get("embeddings", {})
        missing_from_index = [
            ft for ft in forced_tables if ft not in _saved_check
        ]
        if missing_from_index:
            print(
                f"  [MRA WARNING] Forced table(s) missing from embedding index: "
                f"{missing_from_index}. Rebuild embeddings and retry."
            )
            return {
                "error": (
                    f"Tables {missing_from_index} are not in the embedding index. "
                    "Run the embedding generation pipeline to add them, then retry."
                )
            }

        system_prompt = f"""
You are the Metadata Retrieval Agent for a Spatial SQL database.
Find the exact tables, columns, and (if requested) PostGIS functions needed to
answer the user's question.

User Question: "{user_question}"
Extracted Entities: {json.dumps(entities)}
Filter Conditions (WHERE clause values — do NOT search for these): {json.dumps(filter_conditions)}

FORCED TABLES (must appear in selected_tables): {json.dumps(forced_tables)}

AVAILABLE SCHEMA:
{json.dumps(self.tables_json_abrvs)}

INSTRUCTIONS:
1. Call 'search_columns' for each CONCEPT entity (general type only). Keep
   queries short (e.g., 'name', 'geometry', 'length', 'speed limit').
   Do NOT search for filter_condition values like numeric IDs (e.g., "360036").
2. Max 1-2 searches per concept. Do NOT over-search.
3. selected_columns MUST use ONLY column names returned by the tools.
4. ALL forced tables MUST appear in selected_tables.
"""
        if spatial_function_req:
            system_prompt += (
                "\n5. Call 'search_spatial_functions' ONCE for the PostGIS function."
            )

        system_prompt += f"""

OUTPUT — raw JSON only, no markdown:
{{
    "selected_tables": {json.dumps(forced_tables) if forced_tables else '["table1"]'},
    "selected_columns": [
        {{"table": "t", "column": "exact_col_from_tools", "reason": "why"}}
    ],
    "selected_functions": []
}}
REMINDER: selected_tables MUST include {json.dumps(forced_tables)}.
REMINDER: only use column names that appeared in tool results.
REMINDER: do NOT search for numeric IDs or filter values.
"""

        messages = [{"role": "system", "content": system_prompt}]
        tools    = [
            {
                "type": "function",
                "function": {
                    "name":        "search_columns",
                    "description": "Semantic search to find real database columns by concept.",
                    "parameters": {
                        "type":       "object",
                        "properties": {"query_text": {"type": "string"}},
                        "required":   ["query_text"],
                    },
                },
            }
        ]
        if spatial_function_req:
            tools.append({
                "type": "function",
                "function": {
                    "name":        "search_spatial_functions",
                    "description": "Semantic search for PostGIS spatial functions.",
                    "parameters": {
                        "type":       "object",
                        "properties": {"query_text": {"type": "string"}},
                        "required":   ["query_text"],
                    },
                },
            })

        tool_search_results: list      = []
        final_llm_json: Optional[dict] = None

        for _ in range(8):
            response         = self.openai_client.chat.completions.create(
                model=OpenAIModel, messages=messages, tools=tools, tool_choice="auto"
            )
            # Record token usage
            usage = response.usage
            self.metrics.add_usage(usage.prompt_tokens, usage.completion_tokens)

            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.tool_calls:
                for tc in response_message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_columns":
                        result = self._search_columns(args["query_text"])
                        try:
                            tool_search_results.extend(json.loads(result))
                        except json.JSONDecodeError:
                            pass
                    elif tc.function.name == "search_spatial_functions":
                        result = self._search_spatial_functions(args["query_text"])
                    else:
                        result = json.dumps({"error": "Unknown tool"})
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         tc.function.name,
                        "content":      result,
                    })
            else:
                raw            = response_message.content or ""
                final_llm_json = self._extract_json(raw)
                if final_llm_json is None:
                    print(f"  [MRA Error] Could not extract JSON:\n{raw}")
                break

        # ---- post-loop guards ------------------------------------------------

        if not final_llm_json:
            final_llm_json = {
                "selected_tables":    [],
                "selected_columns":   [],
                "selected_functions": [],
            }

        # FIX 2A: enforce forced tables
        existing = set(final_llm_json.get("selected_tables", []))
        for ft in forced_tables:
            if ft not in existing:
                print(f"  [MRA Guard] Adding missing forced table '{ft}'.")
                final_llm_json.setdefault("selected_tables", []).append(ft)

        # FIX 2B: infer tables from columns if still empty
        if not final_llm_json.get("selected_tables"):
            inferred = list({
                c["table"]
                for c in final_llm_json.get("selected_columns", [])
                if "table" in c
            })
            if inferred:
                print(f"  [MRA Guard] Inferred tables from columns: {inferred}")
                final_llm_json["selected_tables"] = inferred

        # FIX 2C: global fallback when no columns at all
        if not final_llm_json.get("selected_columns") and tool_search_results:
            print("  [MRA Guard] Trying global tool-search fallback...")
            final_llm_json["selected_columns"] = self._build_fallback_columns(
                tool_search_results,
                final_llm_json.get("selected_tables", []),
                entities,
            )

        # FIX 2D + FIX MRA-2: targeted per-table search when:
        #   (a) a forced table has zero columns, OR
        #   (b) a forced table is involved in a spatial query but has no geom column
        current_columns = final_llm_json.get("selected_columns", [])
        tables_covered  = {c["table"] for c in current_columns}

        for ft in forced_tables:
            needs_targeted = False
            reason_tag     = ""

            if ft not in tables_covered:
                # Case (a): table has no columns at all
                needs_targeted = True
                reason_tag     = "zero columns"
            elif spatial_function_req and self._table_missing_geom(ft, current_columns):
                # FIX MRA-2 — Case (b): spatial query but geom column missing
                needs_targeted = True
                reason_tag     = "missing geometry column on spatial query"

            if needs_targeted:
                print(
                    f"  [MRA Guard] Targeted search for '{ft}' ({reason_tag}). "
                    "Running targeted per-table search."
                )
                # For case (b), also include geometry-specific queries
                query_texts = list(searchable_general) + filter_conditions
                if spatial_function_req:
                    query_texts = ["geometry", "geom"] + query_texts
                if not query_texts:
                    query_texts = [user_question]

                targeted = self._targeted_table_search(
                    ft, query_texts, top_k_per_query=3
                )
                added = 0
                for hit in targeted:
                    if hit["score"] < 0.2 or added >= 5:
                        break
                    final_llm_json.setdefault("selected_columns", []).append({
                        "table":  ft,
                        "column": hit["column"],
                        "reason": (
                            f"Targeted search fallback [{reason_tag}] "
                            f"(score={hit['score']:.3f})."
                        ),
                    })
                    added += 1

        # Hard stop if nothing was mapped
        if (
            not final_llm_json.get("selected_tables")
            and not final_llm_json.get("selected_columns")
        ):
            return {
                "error": (
                    "MRA failed to map any tables or columns. "
                    "Cannot proceed without schema grounding."
                )
            }

        corrected = self._validate_and_correct_columns(final_llm_json)
        return self._enrich_metadata(corrected)

    # ---------------------------------------------------------------- enrich

    def _enrich_metadata(self, mapped_json: dict) -> dict:
        """Attach descriptions and sample values to the selected schema."""
        enriched = {
            "tables":    {},
            "functions": mapped_json.get("selected_functions", []),
        }

        for table in mapped_json.get("selected_tables", []):
            enriched["tables"][table] = {
                "description": self.tables_json_abrvs.get(table, "No description."),
                "columns":     {},
            }

        for col_info in mapped_json.get("selected_columns", []):
            t, c       = col_info["table"], col_info["column"]
            uncertain  = col_info.get("uncertain", False)

            if t not in enriched["tables"]:
                enriched["tables"][t] = {"description": "", "columns": {}}

            if uncertain:
                desc    = f"UNCERTAIN COLUMN '{c}'. {col_info.get('reason', '')}"
                samples = []
            else:
                desc    = self.ai_descriptions.get(t, {}).get(
                    c, self.columns_json_abrvs.get(t, {}).get(c, "No description.")
                )
                samples = self.column_values_json.get(t, {}).get(c, [])

            enriched["tables"][t]["columns"][c] = {
                "reason_selected": col_info.get("reason", ""),
                "description":     desc,
                "sample_values":   (
                    samples[:5] if isinstance(samples, list) else samples
                ),
            }

        return enriched


# -----------------------------------------------------------------------------
# 3. QUERY LOGIC AGENT (QLA)
# -----------------------------------------------------------------------------
class QueryLogicAgent:
    """
    Generates a high-level logical execution plan.
    """

    def __init__(self, openai_api_key: str, metrics: Metrics):
        self.client = OpenAI(api_key=openai_api_key)
        self.metrics = metrics

    def _verify_spatial_function(self, function_name: str, geom_types: list) -> str:
        print(f"  [QLA Tool] Verifying {function_name} with {geom_types}...")
        fname   = function_name.upper()
        geom_up = [g.upper() for g in geom_types]
        if fname == "ST_AREA" and any(
            g in ["POINT", "MULTILINESTRING", "LINESTRING"] for g in geom_up
        ):
            return json.dumps({
                "valid": False,
                "error": f"{fname} cannot be used on 0D/1D geometries.",
            })
        if fname in ("ST_INTERSECTS", "ST_INTERSECTION"):
            return json.dumps({"valid": True, "message": f"{fname} is valid."})
        return json.dumps({"valid": True, "message": "Function signature appears valid."})

    def _check_join_path(self, table_a: str, table_b: str) -> str:
        print(f"  [QLA Tool] Join path: {table_a} <-> {table_b}...")
        return json.dumps({
            "join_type":      "spatial",
            "recommendation": (
                f"No relational FK between {table_a} and {table_b}. "
                "Use ST_Intersects on geom columns."
            ),
        })

    def _extract_json(self, text: str) -> Optional[dict]:
        brace_stack, start = [], -1
        for i, ch in enumerate(text):
            if ch == "{":
                if not brace_stack:
                    start = i
                brace_stack.append(ch)
            elif ch == "}" and brace_stack:
                brace_stack.pop()
                if not brace_stack and start != -1:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
        cleaned = text.strip().lstrip("`").rstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def generate_plan(self, question: str, mra_schema: dict) -> dict:
        """Draft and verify a step-by-step execution plan."""

        if not mra_schema.get("tables"):
            return {
                "error": (
                    "No schema from MRA. "
                    "Cannot generate a plan without table/column mappings."
                )
            }

        tables_without_columns = [
            t for t, info in mra_schema["tables"].items()
            if not info.get("columns")
        ]
        column_constraint_note = ""
        if tables_without_columns:
            print(
                f"  [QLA Warning] Tables with no mapped columns: "
                f"{tables_without_columns}. Injecting column constraint."
            )
            column_constraint_note = f"""
STRICT COLUMN CONSTRAINT:
Tables with NO columns in the schema: {tables_without_columns}.
For these tables use descriptive placeholders:
  COLUMN_UNKNOWN_name, COLUMN_UNKNOWN_geometry, COLUMN_UNKNOWN_speed, etc.
List each needed column separately. The SQL agent will resolve them.
"""

        system_prompt = f"""
You are the Query Logic Agent for a Spatial SQL database.
Generate a step-by-step execution plan. DO NOT write SQL.

User Question: "{question}"

Available Schema:
{json.dumps(mra_schema, indent=2)}
{column_constraint_note}

INSTRUCTIONS:
1. Use ONLY columns listed in each table's 'columns' dict.
2. For spatial functions call 'verify_spatial_function'.
3. For table joins call 'check_join_path'.
4. Stop using tools once validated; output final JSON.

FINAL OUTPUT — JSON only, no markdown:
{{
    "step_by_step_plan": ["Step 1: ...", "Step 2: ..."],
    "trimmed_schema_required": {{"table1": ["col1", "COLUMN_UNKNOWN_speed"]}}
}}
"""

        messages = [{"role": "system", "content": system_prompt}]
        tools    = [
            {
                "type": "function",
                "function": {
                    "name": "verify_spatial_function",
                    "description": "Checks PostGIS function compatibility with geometry types.",
                    "parameters": {
                        "type":       "object",
                        "properties": {
                            "function_name": {"type": "string"},
                            "geom_types":    {
                                "type":  "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["function_name", "geom_types"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_join_path",
                    "description": "Determines join type between two tables.",
                    "parameters": {
                        "type":       "object",
                        "properties": {
                            "table_a": {"type": "string"},
                            "table_b": {"type": "string"},
                        },
                        "required": ["table_a", "table_b"],
                    },
                },
            },
        ]

        for iteration in range(8):
            response = self.client.chat.completions.create(
                model=OpenAIModel, messages=messages, tools=tools, tool_choice="auto"
            )
            # Record token usage
            usage = response.usage
            self.metrics.add_usage(usage.prompt_tokens, usage.completion_tokens)

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "verify_spatial_function":
                        result = self._verify_spatial_function(
                            args["function_name"], args["geom_types"]
                        )
                    elif tc.function.name == "check_join_path":
                        result = self._check_join_path(args["table_a"], args["table_b"])
                    else:
                        result = json.dumps({"error": "Unknown tool"})
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         tc.function.name,
                        "content":      result,
                    })
            else:
                parsed = self._extract_json(msg.content or "")
                if parsed is not None:
                    return parsed
                print(
                    f"  [QLA] Iteration {iteration}: not JSON — asking for correction."
                )
                messages.append({
                    "role":    "user",
                    "content": "Output only the JSON object, no extra text or markdown.",
                })

        return {"error": "QLA failed to converge on a valid plan."}


# -----------------------------------------------------------------------------
# 4. SQL GENERATION AGENT (SGA)
# -----------------------------------------------------------------------------
class SQLGenerationAgent:
    """
    Translates a logical plan into executable PostGIS SQL.
    Resolves COLUMN_UNKNOWN_<concept> placeholders via embedding lookup
    before passing the schema to the LLM.
    """

    def __init__(
        self,
        openai_api_key: str,
        meta_file_path: str,
        metrics: Metrics,
    ):
        self.client         = OpenAI(api_key=openai_api_key)
        self.meta_file_path = meta_file_path
        self.metrics        = metrics

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return a unit-norm embedding vector or None on error using OpenAI text-embedding-3-large."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            self.metrics.add_usage(response.usage.prompt_tokens, 0)
            
            vec = np.array(response.data[0].embedding, dtype=float)
            n   = np.linalg.norm(vec)
            return vec / n if n > 0 else None
        except Exception as e:
            print(f"  [SGA Warning] Embedding error: {e}")
            return None

    def _resolve_unknown_column(self, table: str, concept: str) -> Optional[str]:
        """Resolve COLUMN_UNKNOWN_<concept> to a real column via embedding."""
        print(f"  [SGA Resolve] Resolving '{concept}' in table '{table}'...")
        with open(self.meta_file_path, "rb") as f:
            meta = pickle.load(f)
        saved = meta.get("embeddings", {})

        if table not in saved:
            print(f"  [SGA Resolve] No embeddings for table '{table}'.")
            return None

        q_emb = self._embed(concept)
        if q_emb is None:
            return None

        best_col, best_score = None, -1.0
        for col, emb_vals in saved[table].items():
            c_emb = np.array(emb_vals, dtype=float)
            n     = np.linalg.norm(c_emb)
            if n == 0:
                continue
            score = float(np.dot(q_emb, c_emb / n))
            if score > best_score:
                best_score = score
                best_col   = col

        if best_col and best_score > 0.3:
            print(f"  [SGA Resolve] '{concept}' → '{best_col}' (score={best_score:.3f})")
            return best_col
        print(f"  [SGA Resolve] No confident match for '{concept}' (best={best_score:.3f}).")
        return None

    def _resolve_placeholders_in_schema(self, trimmed_schema: dict) -> dict:
        """Replace COLUMN_UNKNOWN_<concept> keys with real column names."""
        resolved = {}
        for table, cols in trimmed_schema.items():
            if not isinstance(cols, dict):
                resolved[table] = cols
                continue
            resolved[table] = {}
            for col_key, col_info in cols.items():
                if col_key.upper().startswith("COLUMN_UNKNOWN_"):
                    concept  = col_key[len("COLUMN_UNKNOWN_"):].replace("_", " ")
                    real_col = self._resolve_unknown_column(table, concept)
                    final    = real_col if real_col else col_key
                    resolved[table][final] = col_info
                    if real_col:
                        resolved[table][final]["description"] = (
                            f"Resolved from placeholder '{col_key}'."
                        )
                else:
                    resolved[table][col_key] = col_info
        return resolved

    def _probe_exact_string_match(self, table: str, column: str, search_term: str) -> str:
        print(f"  [SGA Tool] Probing '{search_term}' in {table}.{column}...")
        return json.dumps({
            "exact_match_found": None,
            "suggestion":        "Use ILIKE for fuzzy matching",
        })

    def _get_function_signature(self, function_name: str) -> str:
        print(f"  [SGA Tool] Fetching signature for {function_name}...")
        fname = function_name.upper()
        if fname == "ST_DWITHIN":
            return json.dumps({
                "signature": "boolean ST_DWithin(geometry g1, geometry g2, double precision d);",
                "warning":   "SRID 4326: distance in DEGREES. Cast to geography for metres.",
            })
        if fname == "ST_INTERSECTS":
            return json.dumps({
                "signature": "boolean ST_Intersects(geometry a, geometry b);"
            })
        if fname == "ST_AREA":
            return json.dumps({
                "signature": "float ST_Area(geometry); float ST_Area(geography);",
                "warning":   "SRID 4326 → use ST_Area(geom::geography) for square metres.",
            })
        if fname == "ST_LENGTH":
            return json.dumps({
                "signature": "float ST_Length(geometry); float ST_Length(geography);",
                "warning": (
                    "ST_Length(geometry) returns degrees for SRID 4326. "
                    "Use ST_Length(geom::geography) for metres, or "
                    "ST_Length(ST_Transform(geom, 3857)) for projected metres."
                ),
            })
        if fname == "ST_TRANSFORM":
            return json.dumps({
                "signature": "geometry ST_Transform(geometry g, integer srid);",
                "note":      "Reprojects geometry to the target SRID.",
            })
        if fname == "ST_ASTEXT":
            return json.dumps({
                "signature": "text ST_AsText(geometry);",
                "note":      "Returns WKT representation of the geometry.",
            })
        return json.dumps({"signature": f"{function_name}(geometry, geometry)"})

    def generate_sql(self, logical_plan: list, trimmed_schema: dict) -> dict:
        """Resolve COLUMN_UNKNOWN placeholders then generate SQL via ReAct loop."""
        resolved_schema = self._resolve_placeholders_in_schema(trimmed_schema)

        system_prompt = f"""
You are the SQL Generation Agent for a PostGIS database.
Translate the Logical Plan into an executable SQL query.

Trimmed Schema (placeholders already resolved):
{json.dumps(resolved_schema, indent=2)}

Logical Plan:
{json.dumps(logical_plan, indent=2)}

INSTRUCTIONS:
1. Use ONLY column names that appear as keys in the trimmed schema above.
2. For spatial functions call get_function_signature for correct syntax/CRS notes.
3. For length/area in metres on SRID 4326 data: use ::geography cast
   (e.g., ST_Length(geom::geography)) — simpler than ST_Transform.
4. Do NOT execute SQL.
5. Output valid JSON only.

OUTPUT FORMAT:
{{
    "sql_query": "SELECT ...",
    "manifest": {{
        "crs_assumptions":   "...",
        "value_assumptions": "..."
    }}
}}
"""
        messages = [{"role": "system", "content": system_prompt}]
        tools    = [
            {
                "type": "function",
                "function": {
                    "name":        "probe_exact_string_match",
                    "description": "Finds exact column name spelling in a database table.",
                    "parameters": {
                        "type":       "object",
                        "properties": {
                            "table":       {"type": "string"},
                            "column":      {"type": "string"},
                            "search_term": {"type": "string"},
                        },
                        "required": ["table", "column", "search_term"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name":        "get_function_signature",
                    "description": "Returns PostGIS function syntax and CRS notes.",
                    "parameters": {
                        "type":       "object",
                        "properties": {"function_name": {"type": "string"}},
                        "required":   ["function_name"],
                    },
                },
            },
        ]

        for _ in range(5):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                response_format={"type": "json_object"},
            )
            # Record token usage
            usage = response.usage
            self.metrics.add_usage(usage.prompt_tokens, usage.completion_tokens)

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "probe_exact_string_match":
                        result = self._probe_exact_string_match(
                            args["table"], args["column"], args["search_term"]
                        )
                    elif tc.function.name == "get_function_signature":
                        result = self._get_function_signature(args["function_name"])
                    else:
                        result = json.dumps({"error": "Unknown tool"})
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         tc.function.name,
                        "content":      result,
                    })
            else:
                raw = (msg.content or "").strip().lstrip("`").rstrip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    continue

        return {"error": "SGA failed to generate SQL within iteration limit."}


# -----------------------------------------------------------------------------
# 5. REVIEWER AGENT (RA)
# -----------------------------------------------------------------------------
class ReviewerAgent:
    """
    Validates and iteratively fixes SQL using EXPLAIN.
    """

    UNABLE_TO_FIX_PREFIX = "SELECT 'UNABLE_TO_FIX"

    def __init__(self, openai_api_key: str, db_connection_string: str, metrics: Metrics):
        self.client               = OpenAI(api_key=openai_api_key)
        self.db_connection_string = db_connection_string
        self.metrics              = metrics

    def validate_query(self, query: str) -> dict:
        engine = create_engine(self.db_connection_string)
        try:
            with engine.connect() as conn:
                conn.execute(text("SET statement_timeout = 3000"))
                conn.execute(text(f"EXPLAIN {query}"))
                return {"status": "Valid", "Output": "Syntactically correct."}
        except SQLAlchemyError as e:
            return {"status": "Error", "Output": str(e.__cause__ or e)}

    def run_query(self, sql: str) -> dict:
        engine = create_engine(self.db_connection_string)
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(sql)).mappings().all()
                return {"status": "Success", "Output": [dict(r) for r in rows]}
        except SQLAlchemyError as e:
            return {"status": "Error", "Output": str(e.__cause__ or e)}
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"status": "Error", "Output": str(e)}

    def _is_sentinel(self, sql: str) -> bool:
        """True if the SQL is the UNABLE_TO_FIX sentinel — stop iterating."""
        return sql.strip().upper().startswith(self.UNABLE_TO_FIX_PREFIX.upper())

    def _is_semantically_valid(self, sql: str) -> bool:
        """True only if the SQL queries application tables, not system tables."""
        sql_lower          = sql.lower()
        forbidden_patterns = [
            "information_schema",
            "pg_catalog",
            "pg_class",
            "pg_tables",
            "pg_attribute",
            "pg_namespace",
            "select column_name from",
            "select table_name from",
        ]
        for p in forbidden_patterns:
            if p in sql_lower:
                print(f"  [RA Semantic] Forbidden pattern: '{p}'")
                return False
        return True

    def review_and_fix(
        self,
        sql: str,
        question: str,
        mra_schema: dict,
        max_iterations: int = 4,
    ) -> dict:
        """Iteratively validate and fix SQL; return {"status", "sql"}."""
        schema_col_ref: Dict[str, list] = {
            t: list(info.get("columns", {}).keys())
            for t, info in mra_schema.get("tables", {}).items()
        }

        current_sql = sql
        print("RA:")

        for i in range(max_iterations):

            # Sentinel early-exit
            if self._is_sentinel(current_sql):
                print(
                    f"  [RA] UNABLE_TO_FIX sentinel at iteration {i}. "
                    "Stopping — upstream schema had insufficient column data."
                )
                return {
                    "status": "failed",
                    "sql":    "",
                    "reason": (
                        "UNABLE_TO_FIX: insufficient schema column data. "
                        "Check MRA output — columns may not have been resolved."
                    ),
                }

            validation = self.validate_query(current_sql)

            if validation["status"] == "Valid":
                if not self._is_semantically_valid(current_sql):
                    print("  [RA] Passed EXPLAIN but failed semantic check.")
                    validation = {
                        "status": "Error",
                        "Output": (
                            "Query targets a system table or returns a sentinel. "
                            "Rewrite using only application tables from the schema."
                        ),
                    }
                else:
                    print(f"  [RA] SQL valid after {i} iteration(s).")
                    return {"status": "valid", "sql": current_sql}

            print(f"  [RA] Iteration {i}: fixing error...")
            print(f"  Query : {current_sql}")
            print(f"  Error : {validation['Output']}")

            fix_prompt = f"""Fix this PostgreSQL/PostGIS query that failed.

Original question : '{question}'
Failed query      : {current_sql}
Error             : {validation["Output"]}

AVAILABLE SCHEMA (the only tables/columns you may use):
{json.dumps(schema_col_ref, indent=2)}

STRATEGY (follow in order):
1. "column X does not exist":
   a. Find the table X should belong to.
   b. Pick the column whose meaning best matches X from that table's list.
   c. Replace X with that exact column name. No guessing outside the list.
2. Type mismatch / syntax error: fix syntax only; preserve query intent.
3. Missing table: use only tables in the schema.

RULES:
- Never query information_schema, pg_catalog, or system tables.
- Never change what the query answers.
- Only use column names from the schema above.
- If truly unfixable: {{"query": "SELECT 'UNABLE_TO_FIX: <reason>' AS error_message"}}

Respond with JSON only: {{"query": "FIXED_SQL"}}"""

            try:
                resp  = self.client.chat.completions.create(
                    model=OpenAIModel,
                    messages=[{"role": "user", "content": fix_prompt}],
                    response_format={"type": "json_object"},
                )
                # Record token usage
                usage = resp.usage
                self.metrics.add_usage(usage.prompt_tokens, usage.completion_tokens)

                fixed = json.loads(resp.choices[0].message.content).get("query", "")
                if fixed:
                    current_sql = fixed
                    print(f"  Refined: {current_sql}")
                else:
                    print("  [RA] Empty query returned. Stopping.")
                    break
            except Exception as e:
                print(f"  [RA] Fix call failed: {e}")
                break

        return {"status": "failed", "sql": ""}


# -----------------------------------------------------------------------------
# 6. PIPELINE ORCHESTRATOR
# -----------------------------------------------------------------------------
def detect_spatial_intent(query: str, entities: list) -> bool:
    """
    Returns True if the query likely needs PostGIS spatial functions.

    FIX: Added measurement/computation keywords so that ST_Length, ST_Area,
    and similar operations are correctly identified as spatial — previously
    only relational keywords like 'intersect' and 'within' were covered.
    """
    spatial_keywords = [
        # Relational / topological
        "intersect", "within", "near", "buffer", "distance",
        "closest", "furthest", "contain", "cross", "touch",
        "overlap", "adjacent", "boundary", "inside", "outside",
        # Measurement / computation  ← NEW
        "length", "area", "perimeter", "size", "extent",
        "measure", "calculate", "compute", "st_",
        "square meter", "square kilomet", "kilometer", "kilometre",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in spatial_keywords)


def run_pipeline(
    user_query: str,
    database_name: str,
    meta_file_path: str,
    postgis_functions: list,
    tables_abrvs: dict,
    columns_abrvs: dict,
    column_values: dict,
    ai_descriptions: dict,
    openai_api_key: str,
    db_connection_string: str = None,
) -> dict:
    """
    Execute the full agent pipeline: query → entities → schema → plan → SQL → review.
    Returns {"status", "final_sql", "intermediate", "metrics"}.
    """
    metrics = Metrics()
    metrics.start()

    # 1. Entity Extraction
    print("\n=== [1/5] Entity Extraction Agent ===")
    eea = EntityExtractionAgent(openai_api_key=openai_api_key, metrics=metrics)
    ep  = eea.process_query(user_query)
    if ep["status"] != "success":
        metrics.stop()
        return {
            "status":   "clarification_needed",
            "message":  ep["clarification_message"],
            "entities": ep["entities"],
            "metrics":  {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_time_sec": metrics.total_time()
            }
        }
    print(f"  Entities: {json.dumps(ep['entities'], indent=2)}")

    # 2. Spatial intent
    spatial_needed = detect_spatial_intent(user_query, ep["entities"])
    print(f"\n=== [2/5] Spatial Intent: {spatial_needed} ===")

    # 3. Metadata Retrieval
    print("\n=== [3/5] Metadata Retrieval Agent ===")
    mra = MetadataRetrievalAgent(
        openai_api_key=openai_api_key,
        meta_file_path=meta_file_path,
        postgis_functions_list=postgis_functions,
        tables_json_abrvs=tables_abrvs,
        columns_json_abrvs=columns_abrvs,
        column_values_json=column_values,
        ai_descriptions=ai_descriptions,
        metrics=metrics,
    )
    enriched_schema = mra.process(
        user_question=user_query,
        entities=ep["entities"],
        spatial_function_req=spatial_needed,
    )
    if "error" in enriched_schema:
        metrics.stop()
        return {
            "status": "error",
            "message": enriched_schema["error"],
            "metrics": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_time_sec": metrics.total_time()
            }
        }
    print(f"  Tables mapped: {list(enriched_schema.get('tables', {}).keys())}")
    for t, info in enriched_schema.get("tables", {}).items():
        print(f"    {t}: {list(info.get('columns', {}).keys())}")

    # 4. Query Logic
    print("\n=== [4/5] Query Logic Agent ===")
    qla          = QueryLogicAgent(openai_api_key=openai_api_key, metrics=metrics)
    logical_plan = qla.generate_plan(question=user_query, mra_schema=enriched_schema)
    if "error" in logical_plan:
        metrics.stop()
        return {
            "status": "error",
            "message": logical_plan["error"],
            "metrics": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_time_sec": metrics.total_time()
            }
        }
    print(f"  Plan steps     : {len(logical_plan.get('step_by_step_plan', []))}")
    print(f"  Trimmed schema : {logical_plan.get('trimmed_schema_required', {})}")

    # 5. Build trimmed schema with full column details
    trimmed_full = {}
    for table, cols in logical_plan.get("trimmed_schema_required", {}).items():
        if table in enriched_schema["tables"]:
            tinfo               = enriched_schema["tables"][table]
            trimmed_full[table] = {}
            for col in cols:
                trimmed_full[table][col] = tinfo["columns"].get(
                    col, {"description": "Column not in enriched schema"}
                )
        else:
            trimmed_full[table] = {
                "warning": f"Table '{table}' not in enriched schema"
            }

    # 6. SQL Generation
    print("\n=== [5a/5] SQL Generation Agent ===")
    sga = SQLGenerationAgent(
        openai_api_key=openai_api_key,
        meta_file_path=meta_file_path,
        metrics=metrics,
    )
    sql_payload = sga.generate_sql(
        logical_plan=logical_plan["step_by_step_plan"],
        trimmed_schema=trimmed_full,
    )
    if "error" in sql_payload:
        metrics.stop()
        return {
            "status": "error",
            "message": sql_payload["error"],
            "metrics": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_time_sec": metrics.total_time()
            }
        }
    sql_query = sql_payload["sql_query"]
    print(f"  Generated SQL: {sql_query}")

    # 7. Review & Fix
    print("\n=== [5b/5] Reviewer Agent ===")
    if db_connection_string is None:
        db_connection_string = (
            f"postgresql+psycopg2://{db_user}:{db_password}@localhost:5432/{database_name}"
        )
    reviewer      = ReviewerAgent(
        openai_api_key=openai_api_key,
        db_connection_string=db_connection_string,
        metrics=metrics,
    )
    review_result = reviewer.review_and_fix(
        sql=sql_query,
        question=user_query,
        mra_schema=enriched_schema,
    )

    metrics.stop()

    # Print metrics summary
    print("\n=== METRICS ===")
    print(f"Total OpenAI prompt tokens: {metrics.prompt_tokens}")
    print(f"Total OpenAI completion tokens: {metrics.completion_tokens}")
    print(f"Total framework time: {metrics.total_time():.2f} seconds")

    return {
        "status":    review_result["status"],
        "final_sql": review_result.get("sql", ""),
        "intermediate": {
            "entities":        ep,
            "enriched_schema": enriched_schema,
            "logical_plan":    logical_plan,
            "sql_payload":     sql_payload,
        },
        "metrics": {
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_time_sec": metrics.total_time()
        }
    }