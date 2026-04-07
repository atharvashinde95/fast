from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model    = os.getenv("OPENAI_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    if not api_key:
        print("⚠ WARNING: OPENAI_API_KEY not set in .env")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2,
    )
