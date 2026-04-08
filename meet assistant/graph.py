"""
graph.py
--------
LangGraph graph definition for the Meeting Intelligence System.

Control flow (deterministic — driven ONLY by state.validation_status):
  START
    ↓
  input_validation_agent
    ├─ "greeting"        → END   (response already in state)
    ├─ "invalid"         → END   (response already in state)
    ├─ "duplicate"       → qa_agent → END
    ├─ "valid_existing"  → qa_agent → END
    └─ "valid_new"       → transcript_processing_agent → qa_agent → END

Routing is a pure Python function reading state.validation_status.
The LLM never decides routing implicitly.
"""

from langgraph.graph import StateGraph, END

from state import MeetingState
from agents import (
    input_validation_agent,
    transcript_processing_agent,
    qa_agent,
)


# ─────────────────────────────────────────────
# Deterministic router — reads state only
# ─────────────────────────────────────────────
def route_after_validation(state: MeetingState) -> str:
    """
    Pure routing function.  No LLM.  No side effects.
    Returns the name of the next node (or END sentinel).
    """
    status = state.get("validation_status", "invalid")

    if status in ("greeting", "invalid"):
        return "end"                 # terminal — response already written

    if status == "valid_new":
        return "transcript_processing"

    if status in ("valid_existing", "duplicate"):
        return "qa"

    # Catch-all safety net
    return "end"


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(MeetingState)

    # Register nodes
    graph.add_node("input_validation",       input_validation_agent)
    graph.add_node("transcript_processing",  transcript_processing_agent)
    graph.add_node("qa",                     qa_agent)

    # Entry point
    graph.set_entry_point("input_validation")

    # Conditional edge from Input Validation Agent
    graph.add_conditional_edges(
        "input_validation",
        route_after_validation,
        {
            "transcript_processing": "transcript_processing",
            "qa":                    "qa",
            "end":                   END,
        },
    )

    # After Transcript Processing → always go to QA
    graph.add_edge("transcript_processing", "qa")

    # QA always terminates
    graph.add_edge("qa", END)

    return graph.compile()


# Compiled graph — imported by app.py
meeting_graph = build_graph()
