"""
Planner module for the AI Data Analyst Agent.
Maps classified intents to a sequence of tool execution steps.
Phase 4: Agent Brain
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from agent.intent_classifier import IntentResult, IntentType

@dataclass
class ExecutionStep:
    """
    A single step in the agent's execution plan.
    """
    step_number: int
    tool_name: str
    tool_params: Dict       # params to pass to the tool's run()
    description: str        # human-readable: "Running SQL query..."
    is_required: bool       # if False, skip on error and continue
    depends_on_step: Optional[int]   # step that must succeed first (None = no dependency)

@dataclass
class ExecutionPlan:
    """
    The full sequence of steps to execute.
    """
    steps: List[ExecutionStep]
    estimated_steps: int
    plan_summary: str
    needs_clarification: bool
    clarification_question: Optional[str]

def create_plan(intent_result: IntentResult, question: str, session_state) -> ExecutionPlan:
    """
    Maps intent to a concrete sequence of ExecutionSteps.
    """
    steps = []
    needs_clarification = intent_result.requires_clarification
    clarification_question = intent_result.clarification_question
    plan_summary = ""

    # GREETING
    if intent_result.intent == IntentType.GREETING:
        plan_summary = "Respond to greeting directly"
        
    # OUT_OF_SCOPE
    elif intent_result.intent == IntentType.OUT_OF_SCOPE:
        plan_summary = "Inform user this is outside data scope"
        
    # EXPORT
    elif intent_result.intent == IntentType.EXPORT:
        steps.append(ExecutionStep(
            step_number=1,
            tool_name="report_builder",
            tool_params={},
            description="Compiling session into PDF report",
            is_required=True,
            depends_on_step=None
        ))
        plan_summary = "Compile session into PDF report"
        
    # AMBIGUOUS
    elif intent_result.intent == IntentType.AMBIGUOUS:
        needs_clarification = True
        plan_summary = "Request clarification from user"

    # DATA_QUERY or VISUALISATION or FOLLOW_UP
    elif intent_result.intent in [IntentType.DATA_QUERY, IntentType.VISUALISATION, IntentType.FOLLOW_UP]:
        
        # Step 1: memory_retriever
        steps.append(ExecutionStep(
            step_number=1,
            tool_name="memory_retriever",
            tool_params={"query": question},
            description="Checking prior analysis context..." if intent_result.intent != IntentType.FOLLOW_UP else "Loading prior context...",
            is_required=True if intent_result.intent == IntentType.FOLLOW_UP else False,
            depends_on_step=None
        ))
        
        # Step 2: sql_executor
        steps.append(ExecutionStep(
            step_number=2,
            tool_name="sql_executor",
            tool_params={"query": "[TO BE FILLED BY LOOP]", "explanation": question},
            description="Running data query..." if intent_result.intent != IntentType.FOLLOW_UP else "Running follow-up query...",
            is_required=True,
            depends_on_step=1 if intent_result.intent == IntentType.FOLLOW_UP else None
        ))
        
        # Step 3: chart_generator
        steps.append(ExecutionStep(
            step_number=3,
            tool_name="chart_generator",
            tool_params={
                "chart_type": "[TO BE FILLED BY LOOP]",
                "x_column": "[TO BE FILLED BY LOOP]",
                "y_column": "[TO BE FILLED BY LOOP]",
                "title": question[:60],
                "chart_reason": "[TO BE FILLED BY LOOP]"
            },
            description="Generating visualisation...",
            is_required=True if intent_result.intent == IntentType.VISUALISATION else False,
            depends_on_step=2
        ))
        
        # Step 4: insight_narrator
        steps.append(ExecutionStep(
            step_number=4,
            tool_name="insight_narrator",
            tool_params={"result_summary": "[TO BE FILLED BY LOOP]", "question_asked": question},
            description="Writing plain English summary...",
            is_required=True,
            depends_on_step=2
        ))
        
        plan_summary = f"Executing {intent_result.intent.value} analysis steps"

    return ExecutionPlan(
        steps=steps,
        estimated_steps=len(steps),
        plan_summary=plan_summary,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question
    )
