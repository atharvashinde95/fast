"""
app.py
------
Streamlit UI for the Meeting Intelligence System.

UI rules (NON-NEGOTIABLE):
  • UI is stateless — it carries session state only as a data bag.
  • Each button click = one graph execution.
  • UI passes the full current state back into graph on every run.
  • UI renders results returned from graph.
  • UI never decides logic.
"""

import json
import streamlit as st

from graph import meeting_graph
from state import MeetingState


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence System",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Meeting Intelligence System")
st.caption("Paste a meeting transcript to extract insights, then ask questions about it.")


# ─────────────────────────────────────────────
# Session state initialisation
# (Carries LangGraph state across Streamlit re-runs)
# ─────────────────────────────────────────────
if "graph_state" not in st.session_state:
    st.session_state.graph_state: MeetingState = {
        "user_input":          "",
        "validation_status":   "",
        "validation_reason":   "",
        "transcript":          None,
        "transcript_hash":     None,
        "summary":             None,
        "action_items":        None,
        "decisions":           None,
        "key_topics":          None,
        "extraction_complete": False,
        "response":            None,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": str}


# ─────────────────────────────────────────────
# Layout: two columns
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════
# LEFT COLUMN — Input & Chat
# ══════════════════════════════════════════════
with left_col:
    st.subheader("💬 Chat")

    # Render chat history
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input area
    user_text = st.chat_input(
        "Paste a transcript or ask a question about the processed meeting…",
        key="chat_input",
    )

    if user_text:
        # Append user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        # Build input state: carry ALL existing state + new user_input
        input_state: MeetingState = {
            **st.session_state.graph_state,
            "user_input": user_text,
            "response":   None,          # clear previous response
        }

        # ── One graph execution per user turn ──────────────────
        with st.spinner("Processing…"):
            output_state: MeetingState = meeting_graph.invoke(input_state)

        # Persist updated state for next turn
        st.session_state.graph_state = output_state

        # Surface response in chat
        agent_response = output_state.get("response") or "_(no response generated)_"
        st.session_state.chat_history.append({"role": "assistant", "content": agent_response})

        st.rerun()


# ══════════════════════════════════════════════
# RIGHT COLUMN — Extracted Knowledge Panel
# ══════════════════════════════════════════════
with right_col:
    st.subheader("📋 Extracted Meeting Knowledge")

    gs = st.session_state.graph_state

    # Status badge
    status = gs.get("validation_status", "")
    status_colours = {
        "valid_new":      "🟢",
        "valid_existing": "🔵",
        "duplicate":      "🟡",
        "greeting":       "⚪",
        "invalid":        "🔴",
        "":               "⚫",
    }
    badge = status_colours.get(status, "⚫")
    if status:
        st.markdown(f"**Status:** {badge} `{status}`")

    extraction_done = gs.get("extraction_complete", False)

    if not extraction_done:
        st.info("No transcript has been processed yet. Paste a meeting transcript in the chat.")
    else:
        # ── Summary ──────────────────────────────────────────────
        with st.expander("📝 Summary", expanded=True):
            st.write(gs.get("summary") or "_Not available._")

        # ── Key Topics ───────────────────────────────────────────
        topics = gs.get("key_topics") or []
        with st.expander(f"🏷️ Key Topics ({len(topics)})", expanded=False):
            if topics:
                for t in topics:
                    st.markdown(f"- {t}")
            else:
                st.write("_None recorded._")

        # ── Decisions ────────────────────────────────────────────
        decisions = gs.get("decisions") or []
        with st.expander(f"✅ Decisions ({len(decisions)})", expanded=False):
            if decisions:
                for d in decisions:
                    st.markdown(f"- {d}")
            else:
                st.write("_None recorded._")

        # ── Action Items ─────────────────────────────────────────
        items = gs.get("action_items") or []
        with st.expander(f"📌 Action Items ({len(items)})", expanded=True):
            if items:
                for item in items:
                    if isinstance(item, dict):
                        owner    = item.get("owner", "?")
                        task     = item.get("task", "")
                        due_date = item.get("due_date", "N/A")
                        st.markdown(
                            f"**[{owner}]** {task}  \n"
                            f"<span style='color:gray;font-size:0.85em'>Due: {due_date}</span>",
                            unsafe_allow_html=True,
                        )
                        st.divider()
                    else:
                        st.markdown(f"- {item}")
            else:
                st.write("_No action items recorded._")

    # ── Debug: raw state (collapsible) ───────────────────────────
    with st.expander("🔧 Debug: Raw Graph State", expanded=False):
        debug_state = {k: v for k, v in gs.items() if k != "transcript"}
        st.json(debug_state)
        if gs.get("transcript"):
            st.markdown("**Transcript (first 500 chars):**")
            st.code(gs["transcript"][:500] + ("…" if len(gs.get("transcript", "")) > 500 else ""))


# ─────────────────────────────────────────────
# Sidebar — session controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Session")

    if st.button("🗑️ Reset Session", use_container_width=True, type="secondary"):
        st.session_state.graph_state = {
            "user_input":          "",
            "validation_status":   "",
            "validation_reason":   "",
            "transcript":          None,
            "transcript_hash":     None,
            "summary":             None,
            "action_items":        None,
            "decisions":           None,
            "key_topics":          None,
            "extraction_complete": False,
            "response":            None,
        }
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("### How to use")
    st.markdown(
        """
1. **Paste** a meeting transcript in the chat box and press Enter.
2. The system extracts summary, decisions, action items and key topics automatically.
3. **Ask questions** about the meeting — answers are grounded strictly in extracted data.
4. **Reset** to start a new session with a different transcript.
        """
    )

    st.divider()
    st.markdown("### Agent flow")
    st.code(
        """START
  ↓
InputValidation
  ├─ greeting/invalid → END
  ├─ new transcript   → TranscriptProcessing → QA → END
  ├─ existing session → QA → END
  └─ duplicate        → QA → END""",
        language="text",
    )
