"""
graph_builder.py — Assembles and compiles the LangGraph state machine.

Graph topology (matches the diagram):

  __start__
      │
  initialize_state
      │
  router_decide_input ──── process_transcript ──┐
      │                                          │
      ├── load_session ──────────────────────────┤
      │                                          ▼
      ├── ask_for_transcript ←──────── follow_up_mode ──┐
      │         │                           │           │ (loop)
      │         └──── router_decide_input   │           │
      │                                     ▼           │
      └── end_session         router_decide_next_action ┘
                                   │            │
                             ask_for_transcript  end_session
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.nodes import (
    node_initialize_state,
    node_process_transcript,
    node_load_session,
    node_follow_up,
    node_ask_for_transcript,
    node_end_session,
    router_decide_input,
    router_decide_next_action,
)


def build_graph():
    builder = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    builder.add_node("initialize_state",           node_initialize_state)
    builder.add_node("process_transcript",         node_process_transcript)
    builder.add_node("load_session",               node_load_session)
    builder.add_node("follow_up_mode",             node_follow_up)
    builder.add_node("ask_for_transcript",         node_ask_for_transcript)
    builder.add_node("end_session",                node_end_session)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("initialize_state")

    # ── initialize_state → router_decide_input (conditional) ────────────────
    builder.add_edge("initialize_state", "router_decide_input_node")

    # We implement routers as conditional edges, not full nodes
    # LangGraph supports this via add_conditional_edges

    # Dummy passthrough node so we can wire conditional edges from it
    # Actually in LangGraph we wire conditional edges directly from a node.
    # Let's use initialize_state → conditional edge.

    # ── ROUTER 1: after initialize_state ────────────────────────────────────
    builder.add_conditional_edges(
        "initialize_state",
        router_decide_input,
        {
            "process_transcript": "process_transcript",
            "load_session":       "load_session",
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # ── process_transcript → follow_up_mode ─────────────────────────────────
    builder.add_edge("process_transcript", "follow_up_mode")

    # ── load_session → follow_up_mode ───────────────────────────────────────
    builder.add_edge("load_session", "follow_up_mode")

    # ── follow_up_mode → conditional ────────────────────────────────────────
    builder.add_conditional_edges(
        "follow_up_mode",
        _follow_up_router,
        {
            "loop":         "follow_up_mode",
            "next_action":  "router_decide_next_action_node",
        },
    )

    # ── ROUTER 2: after follow_up decides to leave ───────────────────────────
    builder.add_conditional_edges(
        "router_decide_next_action_node",
        router_decide_next_action,
        {
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # ── ask_for_transcript → back to router 1 ───────────────────────────────
    builder.add_conditional_edges(
        "ask_for_transcript",
        router_decide_input,
        {
            "process_transcript": "process_transcript",
            "load_session":       "load_session",
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # ── end_session → END ───────────────────────────────────────────────────
    builder.add_edge("end_session", END)

    # ── Compile with memory ─────────────────────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Helper router functions for conditional edges
# ─────────────────────────────────────────────────────────────────────────────

def _follow_up_router(state: AgentState) -> str:
    intent = state.get("intent", "follow_up")
    if intent in ("new_meeting", "end"):
        return "next_action"
    return "loop"


# We need a passthrough node for router 2 since LangGraph conditional edges
# must originate from a real node, not a function.
# Add it as a no-op node in the builder above.

def _passthrough(state: AgentState) -> AgentState:
    return state


# Patch the builder to add the passthrough nodes
_original_build = build_graph


def build_graph():  # noqa: F811  (intentional override)
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("initialize_state",              node_initialize_state)
    builder.add_node("process_transcript",            node_process_transcript)
    builder.add_node("load_session",                  node_load_session)
    builder.add_node("follow_up_mode",                node_follow_up)
    builder.add_node("ask_for_transcript",            node_ask_for_transcript)
    builder.add_node("end_session",                   node_end_session)
    # Passthrough node for router 2
    builder.add_node("router_decide_next_action_node", _passthrough)

    # Entry
    builder.set_entry_point("initialize_state")

    # Router 1 — after initialize_state
    builder.add_conditional_edges(
        "initialize_state",
        router_decide_input,
        {
            "process_transcript": "process_transcript",
            "load_session":       "load_session",
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # Linear edges after processing
    builder.add_edge("process_transcript", "follow_up_mode")
    builder.add_edge("load_session",       "follow_up_mode")

    # follow_up_mode — loop or escalate
    builder.add_conditional_edges(
        "follow_up_mode",
        _follow_up_router,
        {
            "loop":        "follow_up_mode",
            "next_action": "router_decide_next_action_node",
        },
    )

    # Router 2 — new meeting or end
    builder.add_conditional_edges(
        "router_decide_next_action_node",
        router_decide_next_action,
        {
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # ask_for_transcript — re-enters router 1
    builder.add_conditional_edges(
        "ask_for_transcript",
        router_decide_input,
        {
            "process_transcript": "process_transcript",
            "load_session":       "load_session",
            "ask_for_transcript": "ask_for_transcript",
            "end_session":        "end_session",
        },
    )

    # Terminal
    builder.add_edge("end_session", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
