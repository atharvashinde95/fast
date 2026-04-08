"""
state.py — Shared State for the Meeting Agent Graph

This is the "memory" that every node reads from and writes to.
LangGraph passes this TypedDict through the entire graph execution.
"""

from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict          # use typing_extensions for Python 3.10 compat
from langgraph.graph.message import add_messages  # still valid in langgraph 1.x
from pydantic import BaseModel, Field


# ── Pydantic schemas for structured LLM output ──────────────────────────────

class ActionItem(BaseModel):
    task: str = Field(description="The task to be completed")
    owner: str = Field(description="Person responsible, or 'Unknown'")
    deadline: str = Field(description="Deadline if mentioned, or 'Not specified'")
    priority: str = Field(description="High / Medium / Low")


class Decision(BaseModel):
    decision: str = Field(description="The decision that was made")
    made_by: str = Field(description="Who made or drove the decision")
    rationale: str = Field(description="Reason behind the decision, if mentioned")


class Participant(BaseModel):
    name: str = Field(description="Participant's name")
    role: str = Field(description="Their role or title if mentioned")
    key_contributions: list[str] = Field(description="Main points they raised")


class ExtractionOutput(BaseModel):
    summary: str = Field(description="Concise meeting summary (3-5 sentences)")
    action_items: list[ActionItem] = Field(description="All action items from the meeting")
    decisions: list[Decision] = Field(description="All decisions made during the meeting")
    participants: list[Participant] = Field(description="All participants identified")
    key_topics: list[str] = Field(description="Top 5-7 topics discussed")


# ── LangGraph State ──────────────────────────────────────────────────────────
# MUST be TypedDict (not a plain dict subclass) so LangGraph 1.x can:
#   • detect Annotated reducers (add_messages)
#   • properly merge partial updates returned by nodes

class MeetingState(TypedDict, total=False):
    """
    The central state object passed between every node in the graph.

    Fields:
        transcript          — raw input transcript text
        summary             — meeting summary (set by extractor node)
        action_items        — list of ActionItem dicts
        decisions           — list of Decision dicts
        participants        — list of Participant dicts
        key_topics          — list of topic strings
        extraction_complete — flag: True once extraction finishes
        messages            — chat history; add_messages reducer APPENDS, not replaces
        error               — any error message to surface to the user
    """
    transcript:          str
    summary:             Optional[str]
    action_items:        Optional[list]
    decisions:           Optional[list]
    participants:        Optional[list]
    key_topics:          Optional[list]
    extraction_complete: bool
    messages:            Annotated[list, add_messages]   # ← reducer registered here
    error:               Optional[str]


def initial_state(transcript: str) -> dict:
    """Factory: creates a fresh state from a raw transcript."""
    return {
        "transcript": transcript,
        "summary": None,
        "action_items": [],
        "decisions": [],
        "participants": [],
        "key_topics": [],
        "extraction_complete": False,
        "messages": [],
        "error": None,
    }
