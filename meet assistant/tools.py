"""
tools.py
--------
All tools used by agents in the Meeting Intelligence System.
Tools are stateless, side-effect-free, and reusable.
No tool controls routing — tools only perform actions.
"""

import hashlib
import json
import re
from typing import Any

from langchain_core.tools import tool

from config import get_llm


# ─────────────────────────────────────────────
# 1. LLM Tool — centralised LLM access
# ─────────────────────────────────────────────
@tool
def llm_tool(prompt: str) -> str:
    """
    Centralised LLM call.
    All agents must use this tool for any LLM inference.
    Returns the raw string response from the model.
    """
    llm = get_llm()
    result = llm.invoke(prompt)
    return result.content


# ─────────────────────────────────────────────
# 2. Word Count Tool
# ─────────────────────────────────────────────
@tool
def word_count_tool(text: str) -> int:
    """
    Returns the number of words in the given text.
    Used as an auxiliary signal during input validation.
    """
    return len(text.split())


# ─────────────────────────────────────────────
# 3. Text Hash Tool
# ─────────────────────────────────────────────
@tool
def text_hash_tool(text: str) -> str:
    """
    Returns a SHA-256 hex digest of the provided text.
    Used for duplicate transcript detection.
    """
    return hashlib.sha256(text.strip().encode()).hexdigest()


# ─────────────────────────────────────────────
# 4. Transcript Normalization Tool
# ─────────────────────────────────────────────
@tool
def transcript_normalization_tool(raw_text: str) -> str:
    """
    Cleans and normalises raw transcript text.
    - Strips leading/trailing whitespace
    - Collapses multiple blank lines to a single blank line
    - Normalises Windows line endings
    - Removes non-printable characters (except standard whitespace)
    Returns the cleaned transcript string.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


# ─────────────────────────────────────────────
# 5. Extraction Tool
# ─────────────────────────────────────────────
@tool
def extraction_tool(transcript: str) -> str:
    """
    Performs LLM-based structured extraction from a meeting transcript.
    Returns a JSON string with keys:
      - summary        (str)
      - action_items   (list of dicts: {owner, task, due_date})
      - decisions      (list of str)
      - key_topics     (list of str)
    Only returns JSON — no extra prose.
    """
    llm = get_llm()

    prompt = f"""You are a meeting analyst. Extract structured information from the transcript below.

Return ONLY a valid JSON object with EXACTLY these keys:
{{
  "summary": "<2-4 sentence overview of the meeting>",
  "action_items": [
    {{"owner": "<person>", "task": "<task description>", "due_date": "<date or 'Not specified'>"}}
  ],
  "decisions": ["<decision 1>", "<decision 2>"],
  "key_topics": ["<topic 1>", "<topic 2>"]
}}

Do NOT include markdown code fences, explanations, or any text outside the JSON object.

TRANSCRIPT:
{transcript}
"""
    result = llm.invoke(prompt)
    return result.content


# ─────────────────────────────────────────────
# 6. JSON Parse / Schema Enforcement Tool
# ─────────────────────────────────────────────
@tool
def json_parse_tool(raw_json: str) -> str:
    """
    Parses a raw JSON string and enforces the expected extraction schema.
    Returns a validated, re-serialised JSON string.
    If parsing fails, returns a JSON error envelope.
    """
    try:
        # Strip code fences if the LLM accidentally added them
        cleaned = re.sub(r"```(?:json)?|```", "", raw_json).strip()
        data = json.loads(cleaned)

        # Enforce required keys with defaults
        validated = {
            "summary": data.get("summary", ""),
            "action_items": data.get("action_items", []),
            "decisions": data.get("decisions", []),
            "key_topics": data.get("key_topics", []),
        }

        # Normalise action_items schema
        normalised_items = []
        for item in validated["action_items"]:
            if isinstance(item, dict):
                normalised_items.append({
                    "owner": item.get("owner", "Unknown"),
                    "task": item.get("task", str(item)),
                    "due_date": item.get("due_date", "Not specified"),
                })
            else:
                normalised_items.append({"owner": "Unknown", "task": str(item), "due_date": "Not specified"})
        validated["action_items"] = normalised_items

        return json.dumps(validated)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"JSON parse failed: {exc}", "raw": raw_json[:500]})


# ─────────────────────────────────────────────
# 7. State Builder Tool
# ─────────────────────────────────────────────
@tool
def state_builder_tool(validated_json: str) -> str:
    """
    Converts a validated JSON extraction string into a flat state-ready JSON
    that maps directly to MeetingState fields.
    Returns a JSON string with keys: summary, action_items, extraction_complete.
    """
    try:
        data = json.loads(validated_json)
        if "error" in data:
            return json.dumps({
                "summary": None,
                "action_items": None,
                "extraction_complete": False,
                "error": data["error"],
            })

        return json.dumps({
            "summary": data.get("summary", ""),
            "action_items": data.get("action_items", []),
            "decisions": data.get("decisions", []),
            "key_topics": data.get("key_topics", []),
            "extraction_complete": True,
        })
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({
            "summary": None,
            "action_items": None,
            "extraction_complete": False,
            "error": str(exc),
        })


# ─────────────────────────────────────────────
# 8. Context Construction Tool  (used by QA Agent)
# ─────────────────────────────────────────────
@tool
def context_construction_tool(
    summary: str,
    action_items: str,
    decisions: str,
    key_topics: str,
) -> str:
    """
    Assembles a readable context block from extracted state fields.
    All inputs should be strings (lists serialised as JSON strings).
    Returns a single formatted string to be used as QA grounding context.
    """
    try:
        items = json.loads(action_items) if action_items else []
        decs  = json.loads(decisions)    if decisions    else []
        topics = json.loads(key_topics)  if key_topics   else []
    except (json.JSONDecodeError, TypeError):
        items, decs, topics = [], [], []

    lines = ["=== MEETING CONTEXT ===", ""]

    lines.append("SUMMARY:")
    lines.append(summary or "Not available.")
    lines.append("")

    lines.append("KEY TOPICS:")
    if topics:
        for t in topics:
            lines.append(f"  • {t}")
    else:
        lines.append("  None recorded.")
    lines.append("")

    lines.append("DECISIONS MADE:")
    if decs:
        for d in decs:
            lines.append(f"  • {d}")
    else:
        lines.append("  None recorded.")
    lines.append("")

    lines.append("ACTION ITEMS:")
    if items:
        for a in items:
            if isinstance(a, dict):
                lines.append(f"  • [{a.get('owner', '?')}] {a.get('task', '')} — Due: {a.get('due_date', 'N/A')}")
            else:
                lines.append(f"  • {a}")
    else:
        lines.append("  None recorded.")

    lines.append("")
    lines.append("======================")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 9. Response Generation Tool  (used only by Input Validation Agent)
# ─────────────────────────────────────────────
@tool
def response_generation_tool(reason: str, input_type: str) -> str:
    """
    Produces a polite, user-facing explanation for invalid/greeting inputs.
    Only used by the Input Validation Agent.
    Args:
        reason:     Why the input was rejected / classified as greeting.
        input_type: One of 'greeting', 'invalid', 'too_short', 'duplicate'.
    Returns a friendly string response to show the user.
    """
    llm = get_llm()

    templates = {
        "greeting": "The user greeted the system.",
        "invalid":  "The user provided text that does not look like a meeting transcript.",
        "too_short": "The user provided text that is too short to be a valid transcript.",
        "duplicate": "The user submitted a transcript that has already been processed in this session.",
    }

    situation = templates.get(input_type, reason)

    prompt = f"""You are a polite assistant for a Meeting Intelligence System.

Situation: {situation}
Additional detail: {reason}

Write a brief, friendly response (2-4 sentences) that:
1. Acknowledges what the user did.
2. Explains what the system expects (a meeting transcript with speaker turns, discussion, action items, etc.).
3. Invites them to paste a valid transcript or ask a question about an already-processed one.

Do NOT be condescending. Do NOT use bullet points. Return only the message text.
"""
    result = llm.invoke(prompt)
    return result.content.strip()


# ─────────────────────────────────────────────
# Tool registry — imported by agents
# ─────────────────────────────────────────────
ALL_TOOLS = [
    llm_tool,
    word_count_tool,
    text_hash_tool,
    transcript_normalization_tool,
    extraction_tool,
    json_parse_tool,
    state_builder_tool,
    context_construction_tool,
    response_generation_tool,
]
