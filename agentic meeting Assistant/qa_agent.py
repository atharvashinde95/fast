"""
agents/qa_agent.py — Conversational Q&A Agent

This agent answers follow-up questions about the meeting.
It has access to:
  1. The full raw transcript
  2. All structured extraction results (summary, action items, decisions, participants, topics)
  3. The full conversation history (multi-turn memory)

The agent uses all of this as context to answer questions accurately.
"""

import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from llm import get_llm


# ── Prompt ───────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a helpful Meeting Assistant. You have already analyzed a meeting transcript
and extracted structured insights. A user is now asking follow-up questions about the meeting.

Use the extracted data and the original transcript to answer accurately.
Be concise but complete. If the answer is not in the transcript, say so clearly.

--- MEETING CONTEXT ---

SUMMARY:
{summary}

ACTION ITEMS:
{action_items}

DECISIONS MADE:
{decisions}

PARTICIPANTS:
{participants}

KEY TOPICS:
{key_topics}

ORIGINAL TRANSCRIPT:
{transcript}

--- END OF CONTEXT ---

Answer the user's questions based strictly on this meeting context.
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),  # full chat history injected here
])


# ── Node function ─────────────────────────────────────────────────────────────

def run_qa_agent(state: dict) -> dict:
    """
    LangGraph node: answers the user's latest question using meeting context.

    Reads:  state["messages"] (last message is the user's question)
            state["summary"], state["action_items"], etc.
    Writes: state["messages"] (appends the AI response)
    """
    print("\n[QA Agent] Processing question...")

    llm = get_llm(temperature=0.3)
    chain = QA_PROMPT | llm

    # Format structured data as readable text for the prompt
    context = {
        "summary": state.get("summary") or "Not yet extracted.",
        "action_items": json.dumps(state.get("action_items") or [], indent=2),
        "decisions": json.dumps(state.get("decisions") or [], indent=2),
        "participants": json.dumps(state.get("participants") or [], indent=2),
        "key_topics": json.dumps(state.get("key_topics") or [], indent=2),
        "transcript": state.get("transcript") or "",
        "messages": state.get("messages") or [],
    }

    try:
        response = chain.invoke(context)
        print(f"  ✓ Answer generated")

        # Return the AI message — LangGraph's add_messages reducer appends it
        return {"messages": [AIMessage(content=response.content)]}

    except Exception as e:
        error_msg = f"I encountered an error while processing your question: {str(e)}"
        print(f"  ✗ QA failed: {e}")
        return {"messages": [AIMessage(content=error_msg)]}
