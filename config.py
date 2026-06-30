"""
Configuration module for the AI Data Analyst Agent.
Handles environment variable loading and Gemini client initialization.
Phase 1: Foundation
"""

from typing import Optional

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import google.api_core.exceptions

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback for Pydantic v1 or environments without pydantic-settings
    from pydantic import BaseSettings
    class SettingsConfigDict(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    gemini_api_key: str
    model_name: str = "gemini-2.0-flash"
    max_output_tokens: int = 4096
    max_retries: int = 3
    chroma_persist_dir: str = "./memory/chroma_store"
    export_dir: str = "./output/exports"
    max_file_size_mb: int = 50
    debug: bool = False
    # LLM provider selection: 'gemini' or 'groq'
    llm_provider: str = "gemini"
    groq_api_key: Optional[str] = None
    groq_api_url: Optional[str] = None

    # Pydantic v2: use model_config only (nested `class Config` is incompatible with v2).
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Singleton instance
settings = Settings()

def validate_config() -> None:
    """
    Validates that the required configuration variables are present.
    Raises:
        ValueError: If gemini_api_key is missing or empty.
    """
    if settings.llm_provider.lower() == "gemini":
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
    elif settings.llm_provider.lower() == "groq":
        if not settings.groq_api_key or not settings.groq_api_url:
            raise ValueError(
                "GROQ_API_KEY and GROQ_API_URL must be set when LLM_PROVIDER=groq. "
                "Set GROQ_API_KEY and GROQ_API_URL in your .env to point to your Groq inference endpoint."
            )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}. Supported: gemini, groq")

def get_gemini_client() -> genai.Client:
    """Returns a configured google.genai Client instance."""
    validate_config()
    return genai.Client(api_key=settings.gemini_api_key)


def _generate_with_gemini(prompt: str, system_instruction: str = "") -> str:
    client = get_gemini_client()

    config = types.GenerateContentConfig(
        max_output_tokens=settings.max_output_tokens,
        temperature=0.2,
        system_instruction=system_instruction if system_instruction else None,
    )

    response = client.models.generate_content(
        model=settings.model_name,
        contents=prompt,
        config=config,
    )
    return response.text


def _generate_with_groq(prompt: str, system_instruction: str = "") -> str:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)

    messages = []
    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=settings.max_output_tokens,
    )

    result = response.choices[0].message.content

    # Safety check — if response looks like HTML, raise clearly
    if result.strip().startswith("<!") or "<html" in result[:100]:
        raise ValueError(
            f"Groq returned HTML instead of text. "
            f"Check GROQ_API_KEY in .env. "
            f"Response preview: {result[:200]}"
        )

    return result

@retry(
    retry=retry_if_exception_type(
        google.api_core.exceptions.ResourceExhausted
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def generate_content(prompt: str, system_instruction: str = "") -> str:
    """
    Central function for all Gemini API calls in the project.
    All modules must call this instead of calling the client directly.
    This makes retry logic and error handling centralised.

    Returns the response text as a string.
    Raises the original exception if all retries fail.
    """
    provider = settings.llm_provider.lower()
    if provider == "groq":
        text = _generate_with_groq(prompt, system_instruction)
    elif provider == "gemini":
        text = _generate_with_gemini(prompt, system_instruction)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be 'groq' or 'gemini'."
        )

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    return text

if __name__ == "__main__":
    # Test loading
    try:
        print(f"Loading config for model: {settings.model_name}")
        validate_config()
        print("Config validated successfully.")
    except Exception as e:
        print(f"Config validation failed: {e}")
