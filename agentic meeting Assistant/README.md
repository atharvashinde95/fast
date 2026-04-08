# Meeting Intelligence System
### A Fully Agentic LangGraph Pipeline for Meeting Transcript Analysis

---

## Problem Statement

Given a raw meeting transcript, the system must:
1. **Extract** structured insights — summary, action items, decisions, participants, topics
2. **Store** everything in a shared graph state (the agent's "memory")
3. **Answer** follow-up questions about the meeting in a conversational loop

---

## Architecture

```
TRANSCRIPT INPUT
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   EXTRACTION GRAPH                      │
│                                                         │
│   START → [Extractor Agent] → END                       │
│                │                                        │
│         Uses structured output (Pydantic)               │
│         to extract all fields in one LLM call           │
└─────────────────────────────────────────────────────────┘
      │
      │  State now contains: summary, action_items,
      │  decisions, participants, key_topics
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                     QA GRAPH (loop)                     │
│                                                         │
│   User Question                                         │
│       │                                                 │
│       ▼                                                 │
│   START → [QA Agent] → END                              │
│                │                                        │
│         Has full context: transcript +                  │
│         extracted data + chat history                   │
│                │                                        │
│         Writes answer to messages[]                     │
│       (add_messages reducer = auto memory)              │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
meeting_agent/
├── main.py          — Entry point, CLI, display, Q&A loop
├── graph.py         — LangGraph graph definitions (extraction + QA)
├── state.py         — MeetingState TypedDict + Pydantic schemas
├── llm.py           — LLM configuration (swap models here)
├── agents/
│   ├── extractor.py — Extraction agent (summary, actions, decisions, etc.)
│   └── qa_agent.py  — Conversational QA agent with memory
├── requirements.txt
└── .env.example
```

---

## Key Concepts Explained

### 1. State (state.py)
The `MeetingState` dict is the **shared memory** of the graph.
Every node reads from it and returns a partial dict to update it.
LangGraph merges updates automatically.

The `messages` field uses `Annotated[list, add_messages]` — this is LangGraph's
built-in **reducer** that automatically appends messages instead of replacing them.
That's how multi-turn memory works.

### 2. Structured Output (Pydantic)
The `ExtractionOutput` Pydantic model is passed to `llm.with_structured_output()`.
This forces the LLM to respond in a validated, typed format — no JSON parsing needed.

### 3. Two Graphs, Shared State
- **Extraction Graph**: runs once, populates the state
- **QA Graph**: runs N times (one per user question), reading the full state each time

The state dict is simply passed between calls in `main.py`. This is the simplest
form of "persistence" — no database needed for a single session.

### 4. Prompts (ChatPromptTemplate)
Each agent has its own prompt template with `MessagesPlaceholder` for the QA agent.
This injects the full conversation history into every new call — giving the QA agent
**multi-turn awareness**.

---

## Setup

```bash
cd meeting_agent

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your Anthropic API key

# Run with the built-in sample transcript
python main.py

# Run with your own transcript file
python main.py --file /path/to/your/transcript.txt
```

---

## How to Extend This

### Add a new extraction field (e.g., "risks identified")
1. Add `risks: list[str]` to `ExtractionOutput` in `state.py`
2. Add it to `initial_state()` in `state.py`
3. Update the extractor prompt in `agents/extractor.py`
4. Display it in `display_extraction_results()` in `main.py`

### Add a new agent node (e.g., a "sentiment analyzer")
1. Create `agents/sentiment.py` with a `run_sentiment_agent(state)` function
2. Register it in `graph.py`: `builder.add_node("sentiment", run_sentiment_agent)`
3. Add an edge: `builder.add_edge("extractor", "sentiment")`

### Add a tool to the QA agent (e.g., search past meetings)
```python
from langchain_core.tools import tool

@tool
def search_past_meetings(query: str) -> str:
    """Search through past meeting records for relevant context."""
    # your search logic here
    return "..."

llm_with_tools = llm.bind_tools([search_past_meetings])
```

### Add persistence (save state across sessions)
Use LangGraph's built-in checkpointing:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("meetings.db")
graph = builder.compile(checkpointer=memory)
```

---

## Example Q&A Session

```
You: Who is responsible for the mobile UI designs?
Assistant: Priya Nair is responsible for finalizing the mobile UI designs. 
Her deadline is this Friday, and she will send the Figma links to James by EOD Friday.
She has been specifically asked to prioritize WCAG accessibility compliance.

You: What was decided about the analytics dashboard?
Assistant: The team decided to descope the advanced analytics features to Q4. 
This decision was made to free up time for the mobile app launch, which Mark 
Thompson confirmed is non-negotiable for Q3.

You: What does Lisa need before she can start QA?
Assistant: Lisa Wang needs two things: a staging environment set up by September 23rd 
(3 weeks before launch), and a test plan document from James's team covering all 
critical user flows, which James committed to delivering by September 15th.
```
