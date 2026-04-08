"""
config.py
---------
LLM configuration with startup validation and clear error messages.
"""

import os
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env — explicitly search from current working directory upward
load_dotenv(override=True)


def _check_dns(hostname: str) -> bool:
    """Returns True if hostname resolves via DNS."""
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except socket.gaierror:
        return False


def validate_config() -> dict:
    """
    Validates all required environment variables and network reachability.
    Returns a dict with keys: ok (bool), errors (list), warnings (list).
    """
    errors   = []
    warnings = []

    api_key  = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model    = os.getenv("OPENAI_MODEL", "").strip()

    if not api_key:
        errors.append("OPENAI_API_KEY is not set in your .env file.")
    elif api_key.lower() in ("your_key_here", "xxx", "changeme"):
        errors.append("OPENAI_API_KEY looks like a placeholder — please set a real key.")

    if not base_url:
        errors.append("OPENAI_BASE_URL is not set in your .env file.")
    else:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        if not hostname:
            errors.append(f"OPENAI_BASE_URL '{base_url}' is not a valid URL.")
        else:
            if not _check_dns(hostname):
                errors.append(
                    f"DNS resolution FAILED for '{hostname}'.\n"
                    f"  → Make sure you are connected to the Capgemini VPN / corporate network.\n"
                    f"  → Try: ping {hostname}"
                )

    if not model:
        warnings.append("OPENAI_MODEL not set — using default model.")

    return {
        "ok":       len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "api_key":  api_key[:8] + "…" if api_key else "(not set)",
        "base_url": base_url or "(not set)",
        "model":    model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    }


def get_llm():
    """
    Returns a configured ChatOpenAI instance.
    Raises RuntimeError with a clear message if config is invalid.
    """
    from langchain_openai import ChatOpenAI

    api_key  = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model    = os.getenv("OPENAI_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from your .env file.\n"
            "Create a .env file in your project root with:\n"
            "  OPENAI_API_KEY=<your-key>\n"
            "  OPENAI_BASE_URL=https://openai.generative.engine.capgemini.com/v1\n"
            "  OPENAI_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is missing from your .env file.\n"
            "Expected: OPENAI_BASE_URL=https://openai.generative.engine.capgemini.com/v1"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
