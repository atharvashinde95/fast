"""
agents.py
=========
Three LangChain agents + LangGraph orchestration for the
Meeting Transcript Analysis System.

Nodes (agents):
  - InputAgent    (entry point)
  - IngestionAgent
  - ChatAgent

Control flow:
  InputAgent  → InputAgent | IngestionAgent | ChatAgent
  IngestionAgent → InputAgent | ChatAgent
  ChatAgent   → InputAgent | END
"""

import os
from typing import TypedDict, Literal, Optional, Any

from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

from tools import (
    session_manager_tool,
    load_pdf_tool,
    load_docx_tool,
    load_txt_tool,
    clean_text_tool,
    hash_transcript_tool,
    store_embeddings_tool,
    retrieve_context_tool,
)

load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        temperature=0,
    )

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    # Conversation history (LangChain messages)
    messages: list[BaseMessage]
    # Current user text input
    user_input: str
    # Raw file bytes (None if text input)
    file_bytes: Optional[bytes]
    # Filename for extension detection
    filename: Optional[str]
    # Active session id
    session_id: Optional[str]
    # ChromaDB collection name for active session
    collection_name: Optional[str]
    # Which agent should run next
    next_agent: Literal["InputAgent", "IngestionAgent", "ChatAgent", "END"]
    # Latest agent response text
    agent_response: str


# ===========================================================================
# AGENT 1 — InputAgent
# ===========================================================================
INPUT_AGENT_SYSTEM = """
You are the InputAgent of a Meeting Transcript Analysis System.
Your ONLY job is to classify the user's intent and route accordingly.

YOU MUST:
1. Classify every user message into exactly one intent:
   - greeting          : the user says hello, hi, hey, etc.
   - invalid_input     : out-of-scope, unrelated, or empty message
   - transcript_input  : the user is providing or uploading a meeting transcript
   - follow_up_question: the user is asking a question about a previously uploaded transcript

2. For greetings → respond warmly, explain the system purpose, and ask them to upload/paste a transcript.
3. For invalid_input → politely decline and explain this system only analyses meeting transcripts.
4. For transcript_input → call session_manager_tool with action="detect_duplicate" is NOT your job.
   Just set your routing decision. The IngestionAgent handles all ingestion logic.
5. For follow_up_question → only route to ChatAgent if a session_id is already active (provided in state).
   If no session is active yet, ask the user to upload a transcript first.

RULES:
- Never answer general knowledge questions.
- Never ingest or embed anything yourself.
- Never access ChromaDB directly.
- If session_id is None and user asks a question, respond asking for a transcript first.

Always end your response with one of these routing tags on its own line:
  ROUTE: InputAgent
  ROUTE: IngestionAgent
  ROUTE: ChatAgent
"""

def run_input_agent(state: GraphState) -> GraphState:
    llm = _get_llm()
    tools = [session_manager_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", INPUT_AGENT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    file_hint = ""
    if state.get("file_bytes") and state.get("filename"):
        file_hint = f"\n[User also uploaded a file: {state['filename']}]"

    context = (
        f"Active session_id: {state.get('session_id') or 'None'}\n"
        f"Collection: {state.get('collection_name') or 'None'}"
    )

    result = executor.invoke({
        "input": state["user_input"] + file_hint + f"\n\n[SYSTEM CONTEXT]\n{context}",
        "chat_history": state.get("messages", []),
        "agent_scratchpad": [],
    })

    response_text = result["output"]

    # Parse routing tag
    route = "InputAgent"
    lines = response_text.strip().splitlines()
    clean_lines = []
    for line in lines:
        if line.strip().startswith("ROUTE:"):
            tag = line.strip().replace("ROUTE:", "").strip()
            if tag in ("InputAgent", "IngestionAgent", "ChatAgent"):
                route = tag
        else:
            clean_lines.append(line)

    display_response = "\n".join(clean_lines).strip()

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["user_input"]),
        AIMessage(content=display_response),
    ]

    return {
        **state,
        "messages": new_messages,
        "agent_response": display_response,
        "next_agent": route,
    }


# ===========================================================================
# AGENT 2 — IngestionAgent
# ===========================================================================
INGESTION_AGENT_SYSTEM = """
You are the IngestionAgent of a Meeting Transcript Analysis System.

Your ONLY responsibilities:
1. Extract text from the user's input (plain text) or uploaded file (PDF/DOCX/TXT).
2. Clean and normalize the transcript using clean_text_tool.
3. Generate a content hash using hash_transcript_tool.
4. Check for duplicate transcript using session_manager_tool(action="detect_duplicate", transcript_hash=...).
5. If DUPLICATE:
   - Load existing session using session_manager_tool(action="load_session", session_id=...).
   - Inform user the transcript already exists and they will resume the existing session.
   - Route to ChatAgent.
6. If NOT duplicate:
   - Create a new session using session_manager_tool(action="create_session").
   - Store the hash using session_manager_tool(action="store_transcript_hash", session_id=..., transcript_hash=...).
   - Chunk and embed the transcript using store_embeddings_tool(text=..., collection_name=...).
   - Confirm successful ingestion with a summary (e.g. number of chunks, session id).
   - Route to ChatAgent.
7. If the content is NOT a valid meeting transcript:
   - Explain politely that this content doesn't appear to be a meeting transcript.
   - Route to InputAgent.

VALIDATION RULES for a valid meeting transcript:
- Must contain conversational exchanges OR speaker labels OR timestamps OR meeting metadata.
- Single sentence inputs, code snippets, or random text are NOT transcripts.

IMPORTANT:
- The file content will be passed as raw bytes in base64 via state — but you receive it decoded as a tool argument.
- Always use the tools in the correct sequence.
- After ingestion, clearly state the session_id and collection_name so the system can track them.

End your response with one of:
  ROUTE: ChatAgent
  ROUTE: InputAgent

Also include these special markers so the system can extract session info:
  SESSION_ID: <the session id>
  COLLECTION: <the collection name>
"""

def run_ingestion_agent(state: GraphState) -> GraphState:
    llm = _get_llm()
    tools = [
        session_manager_tool,
        load_pdf_tool,
        load_docx_tool,
        load_txt_tool,
        clean_text_tool,
        hash_transcript_tool,
        store_embeddings_tool,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", INGESTION_AGENT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    # Build the input message with file info if present
    user_msg = state["user_input"]
    if state.get("file_bytes") and state.get("filename"):
        ext = state["filename"].rsplit(".", 1)[-1].lower()
        # Pass bytes as a direct representation; the agent will call the right tool
        file_info = (
            f"\n[File uploaded: {state['filename']} ({ext.upper()})]"
            f"\n[Instruction: Use load_{ext}_tool with the file bytes provided in the tool call to extract text.]"
            f"\n[File size: {len(state['file_bytes'])} bytes]"
        )
        user_msg = user_msg + file_info

    result = executor.invoke({
        "input": user_msg,
        "chat_history": state.get("messages", []),
        "agent_scratchpad": [],
        # Pass file bytes so agent can use them via tool calls
        "file_bytes": state.get("file_bytes"),
        "filename": state.get("filename"),
    })

    response_text = result["output"]

    # Parse routing and session markers
    route = "ChatAgent"
    session_id = state.get("session_id")
    collection_name = state.get("collection_name")
    clean_lines = []

    for line in response_text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("ROUTE:"):
            tag = stripped.replace("ROUTE:", "").strip()
            if tag in ("InputAgent", "ChatAgent"):
                route = tag
        elif stripped.startswith("SESSION_ID:"):
            session_id = stripped.replace("SESSION_ID:", "").strip()
        elif stripped.startswith("COLLECTION:"):
            collection_name = stripped.replace("COLLECTION:", "").strip()
        else:
            clean_lines.append(line)

    display_response = "\n".join(clean_lines).strip()

    new_messages = state.get("messages", []) + [AIMessage(content=display_response)]

    return {
        **state,
        "messages": new_messages,
        "agent_response": display_response,
        "next_agent": route,
        "session_id": session_id,
        "collection_name": collection_name,
        # Clear file bytes after ingestion
        "file_bytes": None,
        "filename": None,
    }


# ===========================================================================
# AGENT 3 — ChatAgent
# ===========================================================================
CHAT_AGENT_SYSTEM = """
You are the ChatAgent of a Meeting Transcript Analysis System.

Your ONLY responsibility:
- Answer user questions STRICTLY based on the retrieved transcript context.
- Use retrieve_context_tool(query=..., collection_name=...) to fetch relevant chunks.
- If the answer is not in the transcript context, say so clearly — DO NOT hallucinate.
- Do NOT answer general knowledge questions.
- Do NOT make up information not present in the retrieved chunks.

After each answer:
- Ask the user: "Would you like to continue asking questions, start a new session, or end the session?"
- Based on response:
  - Continue → respond normally, route to InputAgent
  - New session → route to InputAgent (user will upload new transcript)
  - End → route to END

Always use retrieve_context_tool before answering any question.
The collection_name for the active session is provided in the system context.

End your response with one of:
  ROUTE: InputAgent
  ROUTE: END
"""

def run_chat_agent(state: GraphState) -> GraphState:
    llm = _get_llm()
    tools = [retrieve_context_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_AGENT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    context = (
        f"Active session_id: {state.get('session_id')}\n"
        f"Collection name: {state.get('collection_name')}"
    )

    result = executor.invoke({
        "input": state["user_input"] + f"\n\n[SYSTEM CONTEXT]\n{context}",
        "chat_history": state.get("messages", []),
        "agent_scratchpad": [],
    })

    response_text = result["output"]

    # Parse routing
    route = "InputAgent"
    clean_lines = []
    for line in response_text.strip().splitlines():
        if line.strip().startswith("ROUTE:"):
            tag = line.strip().replace("ROUTE:", "").strip()
            if tag in ("InputAgent", "END"):
                route = tag
        else:
            clean_lines.append(line)

    display_response = "\n".join(clean_lines).strip()

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["user_input"]),
        AIMessage(content=display_response),
    ]

    return {
        **state,
        "messages": new_messages,
        "agent_response": display_response,
        "next_agent": route,
    }


# ===========================================================================
# ROUTING LOGIC
# ===========================================================================
def route_from_input(state: GraphState) -> str:
    return state["next_agent"]

def route_from_ingestion(state: GraphState) -> str:
    return state["next_agent"]

def route_from_chat(state: GraphState) -> str:
    nxt = state["next_agent"]
    return END if nxt == "END" else nxt


# ===========================================================================
# BUILD LANGGRAPH
# ===========================================================================
def build_graph() -> Any:
    """
    Constructs and compiles the LangGraph state machine.

    Valid edges:
      InputAgent    → InputAgent | IngestionAgent | ChatAgent
      IngestionAgent → InputAgent | ChatAgent
      ChatAgent     → InputAgent | END
    """
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("InputAgent", run_input_agent)
    workflow.add_node("IngestionAgent", run_ingestion_agent)
    workflow.add_node("ChatAgent", run_chat_agent)

    # Entry point
    workflow.set_entry_point("InputAgent")

    # Conditional edges from InputAgent
    workflow.add_conditional_edges(
        "InputAgent",
        route_from_input,
        {
            "InputAgent":     "InputAgent",
            "IngestionAgent": "IngestionAgent",
            "ChatAgent":      "ChatAgent",
        },
    )

    # Conditional edges from IngestionAgent
    workflow.add_conditional_edges(
        "IngestionAgent",
        route_from_ingestion,
        {
            "InputAgent": "InputAgent",
            "ChatAgent":  "ChatAgent",
        },
    )

    # Conditional edges from ChatAgent
    workflow.add_conditional_edges(
        "ChatAgent",
        route_from_chat,
        {
            "InputAgent": "InputAgent",
            END:          END,
        },
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience: run one turn of the graph
# ---------------------------------------------------------------------------
def run_graph_turn(
    graph,
    user_input: str,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    session_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> dict:
    """
    Runs a single conversational turn through the compiled LangGraph.

    Returns a dict with:
      - agent_response   : The assistant's reply text
      - session_id       : Active session id (may be newly created)
      - collection_name  : Active ChromaDB collection
      - messages         : Updated message history
      - next_agent       : Last routing decision
    """
    initial_state: GraphState = {
        "messages":        chat_history or [],
        "user_input":      user_input,
        "file_bytes":      file_bytes,
        "filename":        filename,
        "session_id":      session_id,
        "collection_name": collection_name,
        "next_agent":      "InputAgent",
        "agent_response":  "",
    }

    final_state = graph.invoke(initial_state)
    return {
        "agent_response":  final_state.get("agent_response", ""),
        "session_id":      final_state.get("session_id"),
        "collection_name": final_state.get("collection_name"),
        "messages":        final_state.get("messages", []),
        "next_agent":      final_state.get("next_agent"),
    }
