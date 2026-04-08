"""
llm.py — Central LLM configuration

Uses the user's Capgemini OpenAI-compatible endpoint.
ChatOpenAI works because the endpoint speaks the OpenAI API spec,
even though the underlying model is Claude.

⚠️  PROXY NOTE — with_structured_output:
    Our extractor uses llm.with_structured_output(Pydantic model).
    Under the hood this uses OpenAI function/tool calling.
    If the Capgemini proxy supports tool calling → works automatically.
    If not → the extractor falls back to JSON-mode prompt + manual parsing.
    See agents/extractor.py for the fallback implementation.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a configured ChatOpenAI instance pointing at the Capgemini proxy.

    Args:
        temperature: 0.0 for extraction (deterministic structured output),
                     0.3 for conversational QA (more natural replies)
    """
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model    = os.getenv("OPENAI_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    if not api_key:
        print("⚠  WARNING: OPENAI_API_KEY not set in .env")
    if not base_url:
        print("⚠  WARNING: OPENAI_BASE_URL not set in .env — will hit OpenAI directly")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
