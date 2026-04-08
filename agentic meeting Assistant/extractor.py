"""
agents/extractor.py — Meeting Extraction Agent

Extracts all structured information from the transcript in one LLM call.

Strategy — two-tier approach for proxy compatibility:
  1. PRIMARY:  with_structured_output(ExtractionOutput)
               Uses OpenAI tool/function calling under the hood.
               Works if the Capgemini proxy supports the tools API.

  2. FALLBACK: JSON-mode prompt → manual Pydantic parsing
               Used automatically if tool calling raises an exception.
               Always works as long as the proxy returns plain text.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from state import ExtractionOutput
from llm import get_llm


# ── Prompt (shared by both strategies) ───────────────────────────────────────

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert meeting analyst. Carefully read the transcript and extract
structured information with high precision.

Rules:
- Only extract information explicitly present in the transcript.
- For owners/names, use exact names as mentioned; if unknown write "Unknown".
- Deadlines: use exact phrases (e.g. "end of Q3", "next Friday"); if absent write "Not specified".
- Priority: infer from urgency language — "urgent"/"ASAP" = High, "when you can" = Low, default Medium.
- Decisions: only things actually decided, not just discussed.
- Be exhaustive — do not miss any action item or decision.

Return a JSON object with this exact structure:
{{
  "summary": "string (3-5 sentences)",
  "action_items": [
    {{"task": "string", "owner": "string", "deadline": "string", "priority": "High|Medium|Low"}}
  ],
  "decisions": [
    {{"decision": "string", "made_by": "string", "rationale": "string"}}
  ],
  "participants": [
    {{"name": "string", "role": "string", "key_contributions": ["string"]}}
  ],
  "key_topics": ["string"]
}}

Return ONLY valid JSON. No markdown, no explanation."""
    ),
    (
        "human",
        "Transcript:\n\n---\n{transcript}\n---\n\nExtract all structured information."
    )
])


# ── Primary: tool-calling structured output ───────────────────────────────────

def _extract_with_tool_calling(transcript: str) -> ExtractionOutput:
    """Uses with_structured_output — works if proxy supports tool calling."""
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ExtractionOutput)
    chain = EXTRACTION_PROMPT | structured_llm
    return chain.invoke({"transcript": transcript})


# ── Fallback: JSON prompt + manual parsing ────────────────────────────────────

def _extract_with_json_prompt(transcript: str) -> ExtractionOutput:
    """
    Fallback strategy when tool calling is not supported by the proxy.
    Prompts the model to return raw JSON, then parses it with Pydantic.
    """
    llm = get_llm(temperature=0.0)
    chain = EXTRACTION_PROMPT | llm

    response = chain.invoke({"transcript": transcript})
    raw_text = response.content

    # Strip accidental markdown fences if the model adds them
    if raw_text.strip().startswith("```"):
        raw_text = raw_text.strip().removeprefix("```json").removeprefix("```")
        raw_text = raw_text.rstrip("`").strip()

    data = json.loads(raw_text)
    return ExtractionOutput(**data)


# ── Node function ─────────────────────────────────────────────────────────────

def run_extractor(state: dict) -> dict:
    """
    LangGraph node: extracts all insights from the transcript.

    Tries tool-calling first; falls back to JSON-prompt on failure.

    Reads:  state["transcript"]
    Writes: state["summary"], state["action_items"], state["decisions"],
            state["participants"], state["key_topics"], state["extraction_complete"]
    """
    print("\n[Extractor Agent] Analyzing transcript...")

    result = None

    # -- Primary attempt --
    try:
        result = _extract_with_tool_calling(state["transcript"])
        print("  ✓ Used structured output (tool calling)")
    except Exception as e:
        print(f"  ⚠  Tool calling failed ({type(e).__name__}): {e}")
        print("  → Falling back to JSON-prompt extraction...")

    # -- Fallback attempt --
    if result is None:
        try:
            result = _extract_with_json_prompt(state["transcript"])
            print("  ✓ Used JSON-prompt fallback")
        except Exception as e:
            print(f"  ✗ Fallback also failed: {e}")
            return {
                "extraction_complete": False,
                "error": f"Extraction failed: {str(e)}",
            }

    print(f"  ✓ Summary generated")
    print(f"  ✓ {len(result.action_items)} action items found")
    print(f"  ✓ {len(result.decisions)} decisions found")
    print(f"  ✓ {len(result.participants)} participants identified")
    print(f"  ✓ {len(result.key_topics)} key topics extracted")

    return {
        "summary":             result.summary,
        "action_items":        [item.model_dump() for item in result.action_items],
        "decisions":           [d.model_dump() for d in result.decisions],
        "participants":        [p.model_dump() for p in result.participants],
        "key_topics":          result.key_topics,
        "extraction_complete": True,
        "error":               None,
    }
