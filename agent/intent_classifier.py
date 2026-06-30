"""
Intent Classifier module for the AI Data Analyst Agent.
Classifies user requests using Gemini to determine the appropriate analysis path.
Phase 4: Agent Brain
"""

import json
import time
from enum import Enum
from dataclasses import dataclass
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

def classify_intent(question: str, session_state) -> IntentResult:
    """
    Uses Gemini to classify the user's intent.
    """
    start_time = time.time()
    
    filename = session_state.active_filename or 'None'
    schema_context = session_state.schema_context[:400] if session_state.schema_context else 'None'
    history_count = len(session_state.conversation_history)
    
    prompt = f"""
You are an intent classifier for a data analysis agent.

Dataset loaded: {filename}
Available columns: {schema_context}
Conversation turns so far: {history_count}

User question: "{question}"

Classify this question and respond ONLY with valid JSON in this exact format:
{{
  "intent": "data_query|visualisation|follow_up|clarification|export|greeting|out_of_scope|ambiguous",
  "confidence": 0.0-1.0,
  "requires_clarification": true|false,
  "clarification_question": "question to ask user or null",
  "detected_columns": ["col1", "col2"],
  "detected_time_range": "string or null",
  "detected_metric": "string or null"
}}

CRITICAL RULE — CHECK SCHEMA MATCHES FIRST:
Before setting requires_clarification to true, check if the question already contains a word that exactly matches or closely matches an available column name (case-insensitive, ignoring plurals and underscores).

Examples:
- Question mentions 'product' AND a column named 'product' exists → this is NOT ambiguous, set requires_clarification=false
- Question mentions 'region' AND a column named 'region' exists → NOT ambiguous
- Question mentions 'sales' AND a column named 'sales_usd' exists → NOT ambiguous, the metric is clear
- Question mentions 'time' or 'trend' AND a column named 'month' or 'date' exists → NOT ambiguous

Only set requires_clarification=true when:
- The question uses a vague term with NO matching column at all (e.g. 'show me performance' with no column that means performance)
- There are MULTIPLE columns that could equally match (e.g. two date columns: 'order_date' and 'ship_date', and the question just says 'by date')
- A time period is needed but not specified AND there is no reasonable default (e.g. 'compare to last quarter' with no way to know what 'last' means)

When in doubt, prefer requires_clarification=false and let the agent attempt the query — a wrong first attempt that gets corrected is better than blocking the user with an unnecessary question.

Rules:
- If no dataset is loaded and question is about data → ambiguous
- If question references "that", "it", "the previous" → follow_up
- If question mentions "chart", "graph", "plot", "visualise" → visualisation
- If question mentions "export", "download", "report", "PDF" → export
- requires_clarification=true only if the question cannot be executed without more information (e.g. "show me sales" with no time period AND no default exists)
- detected_columns must include EVERY column name from the schema that is mentioned or implied by the question, even loosely. If the question says 'product' and a column 'product' exists, you MUST include 'product' in detected_columns. A non-empty detected_columns list for the main subject of the question is strong evidence against needing clarification.
"""

    try:
        response_text = generate_content(prompt)

        # Clean up response text in case of markdown blocks
        clean_json = response_text.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[1].rsplit("\n", 1)[0]
        if clean_json.startswith("json"):
            clean_json = clean_json.replace("json", "", 1).strip()
            
        data = json.loads(clean_json)
        
        # Convert string intent to Enum
        try:
            intent_enum = IntentType(data.get("intent", "data_query"))
        except ValueError:
            intent_enum = IntentType.DATA_QUERY
            
        result = {
            "intent": intent_enum,
            "confidence": data.get("confidence", 0.5),
            "requires_clarification": data.get("requires_clarification", False),
            "clarification_question": data.get("clarification_question"),
            "detected_columns": data.get("detected_columns", []),
            "detected_time_range": data.get("detected_time_range"),
            "detected_metric": data.get("detected_metric")
        }

        # Safety net: if detected_columns has at least one column name that exists in the schema,
        # override unnecessary clarification requests.
        if result.get("requires_clarification") and result.get("detected_columns"):
            schema_text = (session_state.schema_context or "").lower()
            valid_columns = [
                col for col in result["detected_columns"]
                if col and col.lower() in schema_text
            ]
            if valid_columns:
                if settings.debug:
                    print(
                        f"[DEBUG] Overriding clarification request — found valid column match(es): {valid_columns}"
                    )
                result["requires_clarification"] = False
                result["clarification_question"] = None

        result = IntentResult(
            intent=result["intent"],
            confidence=result["confidence"],
            requires_clarification=result["requires_clarification"],
            clarification_question=result["clarification_question"],
            detected_columns=result["detected_columns"],
            detected_time_range=result["detected_time_range"],
            detected_metric=result["detected_metric"]
        )
        
    except Exception as e:
        # Fallback to default on failure
        result = IntentResult(
            intent=IntentType.DATA_QUERY,
            confidence=0.5,
            requires_clarification=False,
            clarification_question=None,
            detected_columns=[],
            detected_time_range=None,
            detected_metric=None
        )
        
    execution_time = (time.time() - start_time) * 1000
    session_state.log_tool_call("intent_classifier", True, execution_time)
    
    return result
