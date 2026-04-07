import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "sessions")


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(
    transcript: str,
    summary: str,
    tasks: List[Dict],
    priority_tasks: List[Dict],
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save processed meeting data to a JSON file.
    Returns the session_id used.
    """
    _ensure_sessions_dir()

    if not session_id:
        session_id = f"meeting-{str(uuid.uuid4())[:8]}"

    payload = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
        "transcript": transcript,
        "summary": summary,
        "tasks": tasks,
        "priority_tasks": priority_tasks,
    }

    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return session_id


def load_session(session_id: str) -> Dict[str, Any]:
    """
    Load a previously saved session JSON.
    Raises FileNotFoundError if the session doesn't exist.
    """
    _ensure_sessions_dir()
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Session not found: {session_id}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> List[Dict[str, Any]]:
    """
    Return a list of all saved sessions with basic metadata.
    Sorted by creation time descending (newest first).
    """
    _ensure_sessions_dir()
    sessions = []

    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        session_id = fname[:-5]
        try:
            with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(
                {
                    "session_id": session_id,
                    "created_at": data.get("created_at", ""),
                    "summary_preview": (data.get("summary") or "")[:120],
                    "task_count": len(data.get("tasks", [])),
                }
            )
        except Exception:
            continue

    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session JSON file. Returns True if deleted."""
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False
