"""
Intent Classifier module for the AI Data Analyst Agent.
Classifies user requests using Gemini to determine the appropriate analysis path.
Phase 4: Agent Brain
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from config import generate_content, settings


class IntentType(Enum):
    """
    Types of user intents the agent can handle.
    """
    DATA_QUERY       = "data_query"       # Needs SQL or Python analysis
    VISUALISATION    = "visualisation"    # Explicitly asks for a chart
    FOLLOW_UP        = "follow_up"        # References previous answer
    CLARIFICATION    = "clarification"    # Answering a clarifying question
    EXPORT           = "export"           # Wants to download/export report
    GREETING         = "greeting"         # Hello, thanks, etc.
    OUT_OF_SCOPE     = "out_of_scope"     # Nothing to do with the dataset
    AMBIGUOUS        = "ambiguous"        # Cannot determine without more info


@dataclass
class IntentResult:
    """
    Result of the intent classification process.
    """
    intent: IntentType
    confidence: float
    requires_clarification: bool
    clarification_question: Optional[str]
    detected_columns: List[str]
    detected_time_range: Optional[str]
    detected_metric: Optional[str]


def _extract_columns_from_session(session_state) -> list[str]:
    """
    Extracts the actual column names from the loaded dataset.
    Tries multiple sources in priority order so it works regardless of
    how the session state was populated.
    """
    try:
        df = session_state.active_dataframe
        if df is not None and hasattr(df, "columns"):
            return [str(column) for column in df.columns.tolist()]
    except Exception:
        pass

    try:
        profile = session_state.schema_profile
        if profile is not None and hasattr(profile, "columns"):
            columns = []
            for column in profile.columns:
                if column is None:
                    continue
                if hasattr(column, "name"):
                    columns.append(str(column.name))
                elif isinstance(column, str):
                    columns.append(column)
            if columns:
                return columns
    except Exception:
        pass

    try:
        import re

        schema_text = session_state.schema_context or ""
        matches = re.findall(r"-\s+(\w+)\s+\(", schema_text)
        if matches:
            return matches
    except Exception:
        pass

    return []


def _build_column_context(columns: list[str], session_state) -> str:
    """
    Builds a structured column context string for injection into the LLM prompt.
    """
    if not columns:
        return "No dataset loaded."

    lines = [f"AVAILABLE COLUMNS ({len(columns)} total):"]

    try:
        df = session_state.active_dataframe
        profile = session_state.schema_profile

        for column in columns:
            column_type = "unknown"
            sample = ""

            if profile is not None and hasattr(profile, "columns"):
                col_profile = next(
                    (candidate for candidate in profile.columns if getattr(candidate, "name", None) == column),
                    None,
                )
                if col_profile is not None:
                    column_type = getattr(col_profile, "inferred_meaning", "unknown") or "unknown"
                    sample_values = getattr(col_profile, "sample_values", None) or []
                    if sample_values:
                        sample = f" | e.g. {list(sample_values[:3])}"

            elif df is not None and column in df.columns:
                dtype = str(df[column].dtype)
                if "int" in dtype or "float" in dtype:
                    column_type = "numeric"
                elif "datetime" in dtype:
                    column_type = "datetime"
                elif "bool" in dtype:
                    column_type = "boolean"
                else:
                    column_type = "categorical/text"

                try:
                    values = df[column].dropna().unique()[:3]
                    sample = f" | e.g. {list(values)}"
                except Exception:
                    pass

            lines.append(f"  - {column} ({column_type}){sample}")

    except Exception:
        for column in columns:
            lines.append(f"  - {column}")

    return "\n".join(lines)


def classify_intent(question: str, session_state) -> IntentResult:
    """
    Classifies the user's intent using the actual schema of the loaded dataset.
    Works with any dataset without hardcoded column assumptions.
    """
    start = time.time()

    actual_columns = _extract_columns_from_session(session_state)
    column_context = _build_column_context(actual_columns, session_state)

    dataset_name = session_state.active_filename or "No dataset loaded"
    turns_done = len(getattr(session_state, "conversation_history", []) or [])

    prompt = f"""
You are an intent classifier for an AI data analyst agent.

LOADED DATASET: {dataset_name}

{column_context}

CONVERSATION TURNS SO FAR: {turns_done}

USER QUESTION: "{question}"

YOUR JOB:
Classify the user's intent and respond ONLY with valid JSON.

COLUMN MATCHING RULES:
- The column names above are the exact real column names from the user's dataset.
- If the user's question contains a word that matches or closely matches one of these column names, that column is the one they mean.
- Treat singular/plural variants, underscores, and partial matches as valid if they clearly point to the same field.
- Never ask which column when there is an obvious match.

CLARIFICATION RULES:
Set requires_clarification=true only when all of these are true simultaneously:
  1. The question refers to a concept with no matching column in the dataset at all.
  2. There are multiple equally valid interpretations.
  3. The analysis would produce wrong results without clarification.

Do not ask for clarification when:
- The column name appears verbatim in the question.
- A synonym clearly maps to one column.
- The question says "show me X" and X is a column name.
- The question says "by X" and X is a column name.
- There is only one column of the relevant type.

INTENT RULES:
- Questions about data values, rankings, totals, averages, counts, or comparisons → data_query
- Mentions chart, graph, plot, visualise, or show → visualisation
- References that, it, the previous, or compare to → follow_up
- Mentions export, download, report, or PDF → export
- Hello, thanks, hi, how are you → greeting
- Completely unrelated to the dataset → out_of_scope
- Genuinely impossible to answer without more info → ambiguous

DETECTED COLUMNS:
List every column name from the dataset that appears in or is clearly implied by the question. Use the exact column names from the list above.

RESPOND WITH ONLY THIS JSON — no explanation, no markdown:
{{
  "intent": "data_query|visualisation|follow_up|clarification|export|greeting|out_of_scope|ambiguous",
  "confidence": 0.0-1.0,
  "requires_clarification": false,
  "clarification_question": null,
  "detected_columns": ["exact_col_name_1", "exact_col_name_2"],
  "detected_time_range": null,
  "detected_metric": "the main numeric concept in the question or null"
}}
"""

    try:
        response_text = generate_content(prompt)

        clean = response_text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]).strip()

        result = json.loads(clean)

    except Exception as e:
        print(f"[DEBUG] Intent classification failed: {e}")
        result = {
            "intent": "data_query",
            "confidence": 0.5,
            "requires_clarification": False,
            "clarification_question": None,
            "detected_columns": [],
            "detected_time_range": None,
            "detected_metric": None,
        }

    if result.get("requires_clarification") and actual_columns:
        question_lower = question.lower()
        direct_matches = []

        for column in actual_columns:
            column_lower = str(column).lower()
            if column_lower in question_lower:
                direct_matches.append(column)
                continue

            parts = [part for part in column_lower.replace("-", "_").split("_") if len(part) >= 4]
            if any(part in question_lower for part in parts):
                direct_matches.append(column)

        if direct_matches:
            if settings.debug:
                print(f"[DEBUG] Clarification override — matched real columns: {direct_matches}")
            result["requires_clarification"] = False
            result["clarification_question"] = None

            existing = result.get("detected_columns", []) or []
            for column in direct_matches:
                if column not in existing:
                    existing.append(column)
            result["detected_columns"] = existing

    intent_map = {
        "data_query": IntentType.DATA_QUERY,
        "visualisation": IntentType.VISUALISATION,
        "visualization": IntentType.VISUALISATION,
        "follow_up": IntentType.FOLLOW_UP,
        "clarification": IntentType.CLARIFICATION,
        "export": IntentType.EXPORT,
        "greeting": IntentType.GREETING,
        "out_of_scope": IntentType.OUT_OF_SCOPE,
        "ambiguous": IntentType.AMBIGUOUS,
    }

    intent_enum = intent_map.get(result.get("intent", "data_query"), IntentType.DATA_QUERY)

    elapsed = (time.time() - start) * 1000
    try:
        session_state.log_tool_call("intent_classifier", True, elapsed, None)
    except Exception:
        pass

    return IntentResult(
        intent=intent_enum,
        confidence=float(result.get("confidence", 0.8)),
        requires_clarification=bool(result.get("requires_clarification", False)),
        clarification_question=result.get("clarification_question"),
        detected_columns=result.get("detected_columns", []),
        detected_time_range=result.get("detected_time_range"),
        detected_metric=result.get("detected_metric"),
    )
