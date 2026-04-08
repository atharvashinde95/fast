"""
agents.py
---------
The three agents of the Meeting Intelligence System.

Design rules (NON-NEGOTIABLE):
  • Every agent uses BOTH an LLM AND tools.
  • Agents only read/write state — no agent calls another agent.
  • No tool controls routing.
  • Routing decisions are made by the LLM inside Input Validation Agent,
    then deterministically encoded in state.validation_status.
  • Graph routing reads that status field — the LLM never routes implicitly.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from state import MeetingState
from tools import (
    word_count_tool,
    text_hash_tool,
    transcript_normalization_tool,
    extraction_tool,
    json_parse_tool,
    state_builder_tool,
    context_construction_tool,
    response_generation_tool,
)


# ══════════════════════════════════════════════════════════════════
# 1. INPUT VALIDATION AGENT
# ══════════════════════════════════════════════════════════════════
def input_validation_agent(state: MeetingState) -> MeetingState:
    """
    Entry-point decision-maker.

    Responsibilities:
      - Classify user input (greeting / invalid / new transcript / existing session Q)
      - Compute word count + hash for signals
      - Detect duplicates against session hash
      - Use LLM for semantic classification of ambiguous input
      - Produce response (via response_generation_tool) for terminal cases
      - Write validation_status, validation_reason, transcript, transcript_hash, response
    """

    user_input    = state.get("user_input", "")
    existing_hash = state.get("transcript_hash")        # present if transcript already processed
    extraction_ok = state.get("extraction_complete", False)
    llm = get_llm()

    # ── Tool: word count ──────────────────────────────────────────
    word_count: int = word_count_tool.invoke({"text": user_input})

    # ── Tool: hash ───────────────────────────────────────────────
    input_hash: str = text_hash_tool.invoke({"text": user_input})

    # ── LLM: semantic classification ─────────────────────────────
    system_prompt = """You are the Input Validation Agent for a Meeting Intelligence System.

Classify the user input into EXACTLY ONE of these categories and respond with ONLY the JSON shown:

1. greeting      — social greeting, small talk, or test message
2. invalid       — clearly not a transcript and not a question; random text, junk, code snippets
3. new_transcript — looks like a meeting transcript (speaker turns, discussion, dates, etc.)
4. question      — a question or follow-up about a meeting (no raw transcript in input)

Respond with ONLY this JSON (no markdown, no explanation):
{
  "classification": "<greeting|invalid|new_transcript|question>",
  "reason": "<one sentence explaining why>"
}"""

    classification_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"WORD COUNT: {word_count}\n\nUSER INPUT:\n{user_input}"),
    ])

    # Parse LLM classification
    try:
        raw = classification_response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        classification_data = json.loads(raw)
        classification = classification_data.get("classification", "invalid")
        reason          = classification_data.get("reason", "Unable to classify.")
    except (json.JSONDecodeError, AttributeError):
        classification = "invalid"
        reason          = "Classification parsing failed; treating as invalid."

    # ── Route: greeting / invalid (terminal — produce response) ──
    if classification in ("greeting", "invalid"):
        response_text: str = response_generation_tool.invoke({
            "reason":     reason,
            "input_type": classification,
        })
        return {
            **state,
            "validation_status": classification,
            "validation_reason": reason,
            "response": response_text,
        }

    # ── Route: question + transcript already processed ───────────
    if classification == "question" and extraction_ok:
        return {
            **state,
            "validation_status": "valid_existing",
            "validation_reason": reason,
        }

    # ── Route: question but NO transcript yet ────────────────────
    if classification == "question" and not extraction_ok:
        response_text = response_generation_tool.invoke({
            "reason":     "User asked a question but no transcript has been processed yet.",
            "input_type": "invalid",
        })
        return {
            **state,
            "validation_status": "invalid",
            "validation_reason": "No transcript available to answer from.",
            "response": response_text,
        }

    # ── Route: new_transcript — duplicate check ───────────────────
    if classification == "new_transcript":
        if existing_hash and existing_hash == input_hash:
            response_text = response_generation_tool.invoke({
                "reason":     "The submitted transcript matches the one already in session.",
                "input_type": "duplicate",
            })
            return {
                **state,
                "validation_status": "duplicate",
                "validation_reason": "Duplicate transcript — extraction already complete.",
                "response": response_text,
            }

        # New, unique transcript — normalise and store
        normalised: str = transcript_normalization_tool.invoke({"raw_text": user_input})

        return {
            **state,
            "validation_status": "valid_new",
            "validation_reason": reason,
            "transcript":         normalised,
            "transcript_hash":    input_hash,
            # Reset extraction state so processing runs fresh
            "summary":            None,
            "action_items":       None,
            "decisions":          None,
            "key_topics":         None,
            "extraction_complete": False,
            "response":           None,
        }

    # Fallback (should never reach here)
    return {
        **state,
        "validation_status": "invalid",
        "validation_reason": "Unhandled classification path.",
        "response": "I'm not sure what to do with that input. Please paste a meeting transcript.",
    }


# ══════════════════════════════════════════════════════════════════
# 2. TRANSCRIPT PROCESSING AGENT
# ══════════════════════════════════════════════════════════════════
def transcript_processing_agent(state: MeetingState) -> MeetingState:
    """
    Knowledge extraction & preparation agent.

    Responsibilities:
      - Receive normalised transcript from state
      - Use extraction_tool (LLM-backed) to extract structured knowledge
      - Use json_parse_tool to validate/enforce schema
      - Use state_builder_tool to prepare state-ready payload
      - Write summary, action_items, decisions, key_topics, extraction_complete
    """

    transcript = state.get("transcript", "")
    llm = get_llm()

    # ── Tool: LLM-based structured extraction ────────────────────
    raw_json: str = extraction_tool.invoke({"transcript": transcript})

    # ── Tool: parse + enforce schema ─────────────────────────────
    validated_json: str = json_parse_tool.invoke({"raw_json": raw_json})

    # ── Tool: build state payload ─────────────────────────────────
    state_payload_str: str = state_builder_tool.invoke({"validated_json": validated_json})

    # ── LLM: quality check / audit note (agent must use LLM) ─────
    try:
        payload = json.loads(state_payload_str)
    except json.JSONDecodeError:
        payload = {"extraction_complete": False, "error": "State builder returned invalid JSON."}

    if payload.get("extraction_complete"):
        audit_prompt = f"""You are auditing a meeting extraction result.
Briefly confirm (1 sentence) that the following extraction looks complete and coherent.
Extraction: {state_payload_str[:800]}"""
        audit_note = llm.invoke(audit_prompt).content.strip()
    else:
        audit_note = f"Extraction failed: {payload.get('error', 'unknown error')}"

    # ── Merge extracted fields into state ────────────────────────
    return {
        **state,
        "summary":             payload.get("summary"),
        "action_items":        payload.get("action_items"),
        "decisions":           payload.get("decisions"),
        "key_topics":          payload.get("key_topics"),
        "extraction_complete": payload.get("extraction_complete", False),
        "validation_reason":   state.get("validation_reason", "") + f" | Audit: {audit_note}",
    }


# ══════════════════════════════════════════════════════════════════
# 3. QA AGENT
# ══════════════════════════════════════════════════════════════════
def qa_agent(state: MeetingState) -> MeetingState:
    """
    Knowledge-grounded reasoning agent.

    Responsibilities:
      - Use context_construction_tool to build grounding context from state
      - Use LLM to answer user's question strictly from that context
      - If answer is not in context, explicitly say so
      - Write response to state
    """

    user_input   = state.get("user_input", "")
    summary      = state.get("summary") or ""
    action_items = json.dumps(state.get("action_items") or [])
    decisions    = json.dumps(state.get("decisions")    or [])
    key_topics   = json.dumps(state.get("key_topics")   or [])
    llm = get_llm()

    # ── Tool: build grounding context ────────────────────────────
    context: str = context_construction_tool.invoke({
        "summary":      summary,
        "action_items": action_items,
        "decisions":    decisions,
        "key_topics":   key_topics,
    })

    # ── LLM: grounded answer ─────────────────────────────────────
    system_prompt = """You are the QA Agent for a Meeting Intelligence System.

RULES (strictly enforced):
1. Answer ONLY from the MEETING CONTEXT provided below.
2. Do NOT infer, guess, or use outside knowledge.
3. If the answer is not in the context, say: "This information is not available in the processed transcript."
4. Be concise, factual, and cite specific context elements when possible.
5. Do NOT answer questions unrelated to the meeting."""

    user_message = f"""{context}

USER QUESTION:
{user_input}"""

    answer = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    return {
        **state,
        "response": answer.content.strip(),
    }
