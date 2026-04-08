"""
state.py
--------
LangGraph state schema for the Meeting Intelligence System.
State is the ONLY communication channel between agents.
No global variables. No hidden memory.
"""

from typing import Optional, List, Any
from typing_extensions import TypedDict


class MeetingState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────
    user_input: str                      # Raw input from the user

    # ── Validation ─────────────────────────────────────────
    validation_status: str               # "valid_new" | "valid_existing" | "invalid" | "greeting" | "duplicate"
    validation_reason: str               # Human-readable explanation of the validation outcome

    # ── Transcript ─────────────────────────────────────────
    transcript: Optional[str]            # Normalised transcript text (None if not yet provided)
    transcript_hash: Optional[str]       # SHA-256 hash for duplicate detection

    # ── Extraction ─────────────────────────────────────────
    summary: Optional[str]
    action_items: Optional[List[Any]]    # List of {owner, task, due_date} dicts
    decisions: Optional[List[str]]
    key_topics: Optional[List[str]]
    extraction_complete: bool            # True once Transcript Processing Agent has run

    # ── Response ───────────────────────────────────────────
    response: Optional[str]             # Final user-facing response (written by QA or Input Validation)
