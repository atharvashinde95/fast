"""
graph.py — LangGraph Graph Definition (compatible with langgraph >= 1.1.x)

KEY FIX vs original:
  StateGraph(dict)        ← WRONG in langgraph 1.x: reducers never register
  StateGraph(MeetingState) ← CORRECT: LangGraph reads the TypedDict annotations
                             and wires the add_messages reducer for state["messages"]

Graph Flow:

  EXTRACTION GRAPH (run once):
    START → [extractor] → END

  QA GRAPH (run per user question):
    START → [qa_agent] → END
    (caller in main.py passes full accumulated state each time = multi-turn memory)
"""

from langgraph.graph import StateGraph, START, END
from agents.extractor import run_extractor
from agents.qa_agent import run_qa_agent
from state import MeetingState          # TypedDict — required for reducer detection


# ── Conditional edge ──────────────────────────────────────────────────────────

def check_extraction(state: dict) -> str:
    if state.get("error"):
        print(f"\n[Graph] Extraction error: {state['error']}")
    return END


# ── Graph 1: Extraction Pipeline ──────────────────────────────────────────────

def build_extraction_graph():
    builder = StateGraph(MeetingState)          # ← MeetingState, not dict
    builder.add_node("extractor", run_extractor)
    builder.add_edge(START, "extractor")
    builder.add_conditional_edges("extractor", check_extraction)
    return builder.compile()


# ── Graph 2: QA Conversation ──────────────────────────────────────────────────

def build_qa_graph():
    builder = StateGraph(MeetingState)          # ← MeetingState, not dict
    builder.add_node("qa_agent", run_qa_agent)
    builder.add_edge(START, "qa_agent")
    builder.add_edge("qa_agent", END)
    return builder.compile()


# Pre-compiled instances — import these in main.py
extraction_graph = build_extraction_graph()
qa_graph         = build_qa_graph()
