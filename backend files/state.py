from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    # Core transcript data
    transcript: Optional[str]
    summary: Optional[str]
    tasks: List[Dict[str, Any]]
    priority_tasks: List[Dict[str, Any]]

    # Session management
    session_id: Optional[str]
    loaded_session: Optional[Dict[str, Any]]
    available_sessions: List[str]

    # Routing signals
    mode: str                        # idle | processing | follow_up | ended
    intent: Optional[str]            # greeting | new_transcript | old_session | follow_up | new_meeting | end
    file_path: Optional[str]         # path to uploaded file
    selected_session_id: Optional[str]

    # Conversation history for follow-up
    messages: List[Dict[str, str]]   # [{role, content}]
    last_response: Optional[str]

    # Context bag for misc data
    context: Dict[str, Any]

    # Error tracking
    error: Optional[str]
