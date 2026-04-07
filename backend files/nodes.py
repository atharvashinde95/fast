"""
nodes.py — All LangGraph node functions for the Meeting Assistant agent.

Node order in the graph:
  Start → Initialize State → Router Decide Input
         ↓ branches →
         Process Transcript Using Tools → Follow-Up Mode
         Load Session JSON              → Follow-Up Mode
         Ask User For Transcript        → (wait) → Router Decide Input
         Follow-Up Mode                 → self | Router Decide Next Action
         Router Decide Next Action      → Ask User For Transcript | End Session
         End Session                    → TERMINATE
"""

import json
import re
import os
from typing import Literal

from config import get_llm
from tools.text_extractor import extract_text, clean_text
from tools.session_manager import save_session, load_session, list_sessions
from graph.state import AgentState

llm = get_llm()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: call LLM with a plain string prompt
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()


def _parse_json_block(text: str) -> any:
    """Extract JSON from an LLM response that may have markdown fences."""
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(clean)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Initialize State
# ─────────────────────────────────────────────────────────────────────────────

def node_initialize_state(state: AgentState) -> AgentState:
    """Reset/initialize all state fields at the start of a session."""
    return {
        **state,
        "transcript": None,
        "summary": None,
        "tasks": [],
        "priority_tasks": [],
        "session_id": None,
        "loaded_session": None,
        "available_sessions": list_sessions(),
        "mode": "idle",
        "intent": None,
        "file_path": state.get("file_path"),           # preserve if already set
        "selected_session_id": state.get("selected_session_id"),
        "messages": [],
        "last_response": None,
        "context": {},
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — Router Decide Input  (returns routing key, not state mutation)
# ─────────────────────────────────────────────────────────────────────────────

def router_decide_input(state: AgentState) -> Literal[
    "process_transcript", "load_session", "ask_for_transcript", "end_session"
]:
    """
    Main brain — decides where to go based on the user's first action.
    """
    # 1. New transcript file uploaded
    if state.get("file_path") and os.path.exists(state["file_path"]):
        return "process_transcript"

    # 2. User selected an old session from the UI
    if state.get("selected_session_id"):
        return "load_session"

    # 3. Classify text intent with LLM
    user_text = ""
    messages = state.get("messages", [])
    if messages:
        user_text = messages[-1].get("content", "")

    if user_text:
        intent = _classify_intent_initial(user_text)
        if intent == "end":
            return "end_session"

    # Default: ask user for transcript
    return "ask_for_transcript"


def _classify_intent_initial(text: str) -> str:
    prompt = f"""You are classifying a user message for a meeting assistant.
Classify the following message into exactly one label:
- greeting   (hi, hello, hey, how are you, etc.)
- end        (end session, stop, quit, exit, bye)
- other      (anything else)

Message: "{text}"

Reply with ONLY the label, nothing else."""
    result = _llm_call(prompt).lower().strip()
    if result in ("greeting", "end", "other"):
        return result
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — Process Transcript Using Tools
# ─────────────────────────────────────────────────────────────────────────────

def node_process_transcript(state: AgentState) -> AgentState:
    """Extract, summarize, extract tasks, classify priority, save session."""
    file_path = state.get("file_path", "")

    try:
        # Step 1 — Extract text
        raw_text = extract_text(file_path)
        transcript = clean_text(raw_text)

        # Step 2 — Summarize
        summary = _summarize(transcript)

        # Step 3 — Extract tasks
        tasks = _extract_tasks(transcript)

        # Step 4 — Classify priority
        priority_tasks = _classify_priority(tasks)

        # Step 5 — Save session
        session_id = save_session(
            transcript=transcript,
            summary=summary,
            tasks=tasks,
            priority_tasks=priority_tasks,
            metadata={"source_file": os.path.basename(file_path)},
        )

        assistant_msg = (
            f"✅ Meeting transcript processed successfully!\n\n"
            f"**Session ID:** `{session_id}`\n\n"
            f"**Summary:**\n{summary}\n\n"
            f"**Tasks found:** {len(tasks)}\n"
            f"**High-priority tasks:** {len([t for t in priority_tasks if t.get('priority') == 'High'])}\n\n"
            f"You can now ask me anything about this meeting!"
        )

        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": assistant_msg})

        return {
            **state,
            "transcript": transcript,
            "summary": summary,
            "tasks": tasks,
            "priority_tasks": priority_tasks,
            "session_id": session_id,
            "mode": "follow_up",
            "intent": None,
            "file_path": None,
            "messages": messages,
            "last_response": assistant_msg,
            "error": None,
        }

    except Exception as e:
        err_msg = f"❌ Error processing transcript: {str(e)}"
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": err_msg})
        return {
            **state,
            "mode": "idle",
            "error": str(e),
            "messages": messages,
            "last_response": err_msg,
        }


def _summarize(transcript: str) -> str:
    # Truncate to avoid token limits
    truncated = transcript[:12000]
    prompt = f"""You are an expert meeting analyst.
Analyze the following meeting transcript and produce a structured summary with:
1. Meeting Overview (2-3 sentences)
2. Key Discussion Points (bullet list)
3. Decisions Made (bullet list)
4. Risks / Blockers mentioned (bullet list, if any)

Transcript:
\"\"\"
{truncated}
\"\"\"

Write a clear, professional summary."""
    return _llm_call(prompt)


def _extract_tasks(transcript: str) -> list:
    truncated = transcript[:12000]
    prompt = f"""You are an expert at extracting action items from meeting transcripts.

Extract ALL action items, tasks, and commitments from the transcript below.
For each task, identify:
- task: clear description of what needs to be done
- owner: person responsible (use "TBD" if unknown)
- deadline: due date or timeframe (use "Not specified" if unknown)
- blocker: anything blocking this task (use "None" if none)

Return ONLY a valid JSON array. No explanation, no markdown fences.
Example format:
[
  {{"task": "Review API design", "owner": "Alice", "deadline": "Friday", "blocker": "None"}},
  {{"task": "Set up CI/CD", "owner": "Bob", "deadline": "Next sprint", "blocker": "Waiting for DevOps approval"}}
]

Transcript:
\"\"\"
{truncated}
\"\"\"

JSON array:"""
    result = _llm_call(prompt)
    try:
        return _parse_json_block(result)
    except Exception:
        return []


def _classify_priority(tasks: list) -> list:
    if not tasks:
        return []
    tasks_json = json.dumps(tasks, indent=2)
    prompt = f"""You are a project manager classifying task priorities.

For each task in the list below, add a "priority" field with value: High, Medium, or Low.
Use these rules:
- High: urgent, has a deadline this week, or is blocking other work
- Medium: important but not immediately urgent
- Low: nice-to-have or far future

Return ONLY a valid JSON array with all original fields plus "priority".
No explanation, no markdown fences.

Tasks:
{tasks_json}

JSON array with priority added:"""
    result = _llm_call(prompt)
    try:
        return _parse_json_block(result)
    except Exception:
        # fallback: add Medium priority to all
        return [{**t, "priority": "Medium"} for t in tasks]


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — Load Session JSON
# ─────────────────────────────────────────────────────────────────────────────

def node_load_session(state: AgentState) -> AgentState:
    """Load a previously saved session from disk."""
    session_id = state.get("selected_session_id", "")
    try:
        data = load_session(session_id)

        assistant_msg = (
            f"📂 Session **{session_id}** loaded successfully!\n\n"
            f"**Summary preview:**\n{(data.get('summary') or '')[:400]}...\n\n"
            f"**Tasks:** {len(data.get('tasks', []))} total | "
            f"**High priority:** {len([t for t in data.get('priority_tasks', []) if t.get('priority') == 'High'])}\n\n"
            f"Ask me anything about this meeting!"
        )

        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": assistant_msg})

        return {
            **state,
            "transcript": data.get("transcript"),
            "summary": data.get("summary"),
            "tasks": data.get("tasks", []),
            "priority_tasks": data.get("priority_tasks", []),
            "session_id": session_id,
            "loaded_session": data,
            "mode": "follow_up",
            "intent": None,
            "selected_session_id": None,
            "messages": messages,
            "last_response": assistant_msg,
            "error": None,
        }

    except FileNotFoundError:
        err_msg = f"❌ Session `{session_id}` not found. Please choose a valid session."
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": err_msg})
        return {
            **state,
            "mode": "idle",
            "error": f"Session not found: {session_id}",
            "messages": messages,
            "last_response": err_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Follow-Up Mode (conversation loop)
# ─────────────────────────────────────────────────────────────────────────────

def node_follow_up(state: AgentState) -> AgentState:
    """Answer follow-up questions about the current meeting session."""
    messages = list(state.get("messages", []))
    user_text = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""

    # Detect routing-level intents first
    intent = _classify_intent_followup(user_text)

    if intent in ("new_meeting", "end"):
        return {**state, "intent": intent}

    # Build context for the LLM
    summary = state.get("summary") or "No summary available."
    tasks_json = json.dumps(state.get("priority_tasks") or state.get("tasks") or [], indent=2)
    transcript_excerpt = (state.get("transcript") or "")[:6000]

    # Build conversation history for context (last 10 turns)
    history_turns = messages[-20:]
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history_turns[:-1]
    )

    prompt = f"""You are an intelligent meeting assistant. Answer the user's question using ONLY the meeting data below.
Be concise, accurate, and helpful. Use markdown formatting where appropriate.

=== MEETING SUMMARY ===
{summary}

=== TASKS & PRIORITIES ===
{tasks_json}

=== TRANSCRIPT EXCERPT ===
{transcript_excerpt}

=== CONVERSATION HISTORY ===
{history_str}

=== USER QUESTION ===
{user_text}

Answer:"""

    response = _llm_call(prompt)
    messages.append({"role": "assistant", "content": response})

    return {
        **state,
        "messages": messages,
        "last_response": response,
        "intent": intent,
        "mode": "follow_up",
    }


def _classify_intent_followup(text: str) -> str:
    prompt = f"""Classify this message from a user talking to a meeting assistant.
Labels:
- new_meeting   (user wants to analyze a new meeting: "new meeting", "start over", "analyze another", "next file")
- end           (user wants to stop: "end session", "quit", "exit", "goodbye", "done")
- follow_up     (any question about the meeting content)

Message: "{text}"

Reply with ONLY the label."""
    result = _llm_call(prompt).lower().strip()
    if result in ("new_meeting", "end", "follow_up"):
        return result
    return "follow_up"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — Router Decide Next Action
# ─────────────────────────────────────────────────────────────────────────────

def router_decide_next_action(state: AgentState) -> Literal["ask_for_transcript", "end_session"]:
    """
    Called after follow-up mode detects 'new_meeting' or 'end'.
    """
    intent = state.get("intent", "")
    if intent == "new_meeting":
        return "ask_for_transcript"
    return "end_session"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — Ask User For Transcript
# ─────────────────────────────────────────────────────────────────────────────

def node_ask_for_transcript(state: AgentState) -> AgentState:
    """Prompt the user to upload a meeting transcript file."""
    msg = (
        "👋 Hello! I'm your **Meeting Intelligence Assistant**.\n\n"
        "Please upload a meeting transcript to get started.\n\n"
        "**Supported formats:** PDF · DOCX · TXT\n\n"
        "Or select a previous session from the sidebar to continue."
    )

    messages = list(state.get("messages", []))
    # Only add if not already the last assistant message
    if not messages or messages[-1].get("content") != msg:
        messages.append({"role": "assistant", "content": msg})

    return {
        **state,
        "mode": "idle",
        "intent": None,
        "file_path": None,
        "selected_session_id": None,
        "messages": messages,
        "last_response": msg,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 8 — End Session
# ─────────────────────────────────────────────────────────────────────────────

def node_end_session(state: AgentState) -> AgentState:
    """Terminate the session cleanly."""
    msg = "👋 Session ended. Your data has been saved. Refresh the page to start a new session!"
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": msg})

    return {
        **state,
        "mode": "ended",
        "messages": messages,
        "last_response": msg,
    }
