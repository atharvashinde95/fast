"""
app.py
======
Streamlit UI for the Meeting Transcript Analysis System.
All logic lives in agents.py / tools.py — this file only handles display and I/O forwarding.
"""

import streamlit as st
from agents import build_graph, run_graph_turn

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MeetMind — Transcript Analyst",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — refined dark editorial aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --border:    #1f2430;
    --accent:    #5b8fff;
    --accent2:   #a78bfa;
    --text:      #e2e4ed;
    --muted:     #6b7280;
    --user-bg:   #1a2035;
    --bot-bg:    #141820;
    --success:   #34d399;
    --warn:      #fbbf24;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  /* Sidebar title */
  .sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--accent);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.2rem;
  }
  .sidebar-sub {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
  }

  /* Session badge */
  .session-badge {
    background: #1a2035;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent2);
    word-break: break-all;
    margin-bottom: 0.5rem;
  }
  .session-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
  }

  /* Chat container */
  .chat-wrapper {
    max-width: 820px;
    margin: 0 auto;
    padding: 1.5rem 0 6rem 0;
  }

  /* Page headline */
  .page-headline {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
  }
  .page-tagline {
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
  }

  /* Chat bubbles */
  .msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
  }
  .msg-user .bubble {
    background: var(--user-bg);
    border: 1px solid #2a3560;
    border-radius: 16px 4px 16px 16px;
    padding: 0.75rem 1rem;
    max-width: 72%;
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.55;
  }

  .msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
    gap: 0.6rem;
    align-items: flex-start;
  }
  .bot-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-top: 3px;
  }
  .msg-bot .bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1rem;
    max-width: 78%;
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.6;
  }

  /* File upload zone */
  [data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
  }

  /* Input bar */
  [data-testid="stChatInput"] {
    border-top: 1px solid var(--border) !important;
    background: var(--surface) !important;
  }

  /* Buttons */
  .stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* Status pills */
  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
  }
  .pill-active  { background: #0d2d1f; color: var(--success); border: 1px solid #1a4a30; }
  .pill-none    { background: #1a1a2e; color: var(--muted);   border: 1px solid var(--border); }

  /* Upload label */
  .upload-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.3rem;
  }

  /* Dividers */
  hr { border-color: var(--border) !important; margin: 1rem 0; }

  /* Spinner */
  [data-testid="stSpinner"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "graph" not in st.session_state:
    with st.spinner("Initialising agents…"):
        st.session_state.graph = build_graph()

defaults = {
    "chat_history":    [],   # list of (role, text) tuples for display
    "lc_messages":     [],   # LangChain message objects passed to agents
    "session_id":      None,
    "collection_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">MeetMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Transcript Intelligence</div>', unsafe_allow_html=True)

    # Session status
    if st.session_state.session_id:
        st.markdown('<div class="session-label">Session</div>', unsafe_allow_html=True)
        st.markdown('<span class="pill pill-active">● Active</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="session-badge">{st.session_state.session_id}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.collection_name:
            st.markdown('<div class="session-label" style="margin-top:.6rem">Collection</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="session-badge">{st.session_state.collection_name}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span class="pill pill-none">○ No session</span>', unsafe_allow_html=True)
        st.caption("Upload a transcript to begin.")

    st.markdown("---")

    # File uploader
    st.markdown('<div class="upload-label">Upload Transcript</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT",
    )

    st.markdown("---")

    # Reset button
    if st.button("🗑  Clear session", use_container_width=True):
        for k in ["chat_history", "lc_messages", "session_id", "collection_name"]:
            st.session_state[k] = [] if "history" in k or "messages" in k else None
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.65rem;color:#3d4255;text-align:center;">'
        'Powered by LangGraph · ChromaDB · LangChain'
        '</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="page-headline">Meeting Transcript Analyst</div>'
    '<div class="page-tagline">Ask anything about your transcript — structured, session-scoped, grounded.</div>',
    unsafe_allow_html=True,
)

# Render chat history
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{text}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="bot-avatar">🎙</div>'
            f'<div class="bubble">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type a message or paste a transcript…")

if user_input:
    # Append user message to display
    st.session_state.chat_history.append(("user", user_input))
    st.markdown(
        f'<div class="msg-user"><div class="bubble">{user_input}</div></div>',
        unsafe_allow_html=True,
    )

    # Gather file bytes if uploaded
    file_bytes = None
    filename = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

    # Run the graph
    with st.spinner("Thinking…"):
        result = run_graph_turn(
            graph=st.session_state.graph,
            user_input=user_input,
            file_bytes=file_bytes,
            filename=filename,
            session_id=st.session_state.session_id,
            collection_name=st.session_state.collection_name,
            chat_history=st.session_state.lc_messages,
        )

    # Update state
    st.session_state.session_id      = result.get("session_id") or st.session_state.session_id
    st.session_state.collection_name = result.get("collection_name") or st.session_state.collection_name
    st.session_state.lc_messages     = result.get("messages", [])

    # Display assistant response
    response = result.get("agent_response", "I encountered an issue. Please try again.")
    st.session_state.chat_history.append(("bot", response))
    st.markdown(
        f'<div class="msg-bot">'
        f'<div class="bot-avatar">🎙</div>'
        f'<div class="bubble">{response}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.rerun()
