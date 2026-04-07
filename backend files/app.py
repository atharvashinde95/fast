"""
app.py — Flask server for the Meeting Intelligence Assistant.

Routes:
  GET  /                     → serve index.html
  POST /chat                 → main chat endpoint (text messages)
  POST /upload               → file upload + trigger processing
  POST /load-session         → load a past session by ID
  GET  /sessions             → list all saved sessions
  POST /end-session          → end the current session
  GET  /health               → health check
"""

import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from graph.graph_builder import build_graph
from tools.session_manager import list_sessions

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "meeting-assistant-secret-2025")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Compile the graph once at startup
graph = build_graph()

# In-memory thread registry: browser_session_id → thread_id
# For production, use Redis or a DB
_thread_registry: dict = {}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_thread_id() -> str:
    """Return (or create) a stable LangGraph thread_id for this browser session."""
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    return session["thread_id"]


def _run_graph(thread_id: str, state_patch: dict) -> dict:
    """
    Invoke the graph with a state patch and return the final state.
    The graph is compiled with MemorySaver so previous state is restored
    automatically from the thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state_patch, config=config)
    return result


def _extract_last_assistant_message(state: dict) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg["content"]
    return state.get("last_response", "")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "meeting-assistant"})


@app.route("/sessions", methods=["GET"])
def get_sessions():
    """Return list of all saved sessions for the sidebar."""
    try:
        sessions = list_sessions()
        return jsonify({"sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"sessions": [], "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle text messages from the user.
    Body: { "message": "..." }
    """
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    thread_id = _get_thread_id()
    logger.info(f"[{thread_id}] User: {user_message}")

    try:
        # Get current state snapshot to append the user message
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current = graph.get_state(config)
            current_values = current.values if current else {}
        except Exception:
            current_values = {}

        current_messages = list(current_values.get("messages", []))
        current_messages.append({"role": "user", "content": user_message})

        state_patch = {
            **current_values,
            "messages": current_messages,
            "intent": None,          # let the node re-classify
            "file_path": None,
            "selected_session_id": None,
        }

        # If we're in follow_up mode, go directly to follow_up node
        mode = current_values.get("mode", "idle")
        if mode == "follow_up":
            from graph.nodes import node_follow_up, router_decide_next_action
            from graph.nodes import node_ask_for_transcript, node_end_session
            # Manually step through follow-up since graph waits for human
            new_state = node_follow_up(state_patch)
            intent = new_state.get("intent", "follow_up")
            if intent == "new_meeting":
                new_state = node_ask_for_transcript(new_state)
            elif intent == "end":
                new_state = node_end_session(new_state)
            # Save checkpoint
            graph.update_state(config, new_state)
            final_state = new_state
        else:
            final_state = _run_graph(thread_id, state_patch)

        response_text = _extract_last_assistant_message(final_state)
        logger.info(f"[{thread_id}] Assistant: {response_text[:80]}...")

        return jsonify({
            "response": response_text,
            "mode": final_state.get("mode"),
            "session_id": final_state.get("session_id"),
            "task_count": len(final_state.get("tasks", [])),
        })

    except Exception as e:
        logger.exception(f"[{thread_id}] Chat error: {e}")
        return jsonify({"error": str(e), "response": f"❌ An error occurred: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle transcript file upload.
    Multipart form: file=<file>
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    thread_id = _get_thread_id()
    filename = secure_filename(file.filename)
    # Make filename unique per thread
    unique_name = f"{thread_id[:8]}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(file_path)
    logger.info(f"[{thread_id}] File saved: {file_path}")

    try:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current = graph.get_state(config)
            current_values = current.values if current else {}
        except Exception:
            current_values = {}

        state_patch = {
            **current_values,
            "file_path": file_path,
            "mode": "idle",          # reset so router re-evaluates
            "intent": None,
            "selected_session_id": None,
        }

        final_state = _run_graph(thread_id, state_patch)
        response_text = _extract_last_assistant_message(final_state)

        return jsonify({
            "response": response_text,
            "mode": final_state.get("mode"),
            "session_id": final_state.get("session_id"),
            "task_count": len(final_state.get("tasks", [])),
            "filename": filename,
        })

    except Exception as e:
        logger.exception(f"[{thread_id}] Upload error: {e}")
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e), "response": f"❌ Processing failed: {str(e)}"}), 500


@app.route("/load-session", methods=["POST"])
def load_session_route():
    """
    Load a past session by ID.
    Body: { "session_id": "meeting-xxxx" }
    """
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    thread_id = _get_thread_id()
    logger.info(f"[{thread_id}] Loading session: {session_id}")

    try:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current = graph.get_state(config)
            current_values = current.values if current else {}
        except Exception:
            current_values = {}

        state_patch = {
            **current_values,
            "selected_session_id": session_id,
            "file_path": None,
            "mode": "idle",
            "intent": None,
        }

        final_state = _run_graph(thread_id, state_patch)
        response_text = _extract_last_assistant_message(final_state)

        return jsonify({
            "response": response_text,
            "mode": final_state.get("mode"),
            "session_id": final_state.get("session_id"),
            "task_count": len(final_state.get("tasks", [])),
        })

    except Exception as e:
        logger.exception(f"[{thread_id}] Load session error: {e}")
        return jsonify({"error": str(e), "response": f"❌ Failed to load session: {str(e)}"}), 500


@app.route("/end-session", methods=["POST"])
def end_session_route():
    """End the current session and reset."""
    thread_id = _get_thread_id()
    # Clear browser session so next request gets a fresh thread
    session.pop("thread_id", None)
    return jsonify({"response": "👋 Session ended. Start a new one anytime!", "mode": "ended"})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    logger.info(f"🚀 Meeting Assistant running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
