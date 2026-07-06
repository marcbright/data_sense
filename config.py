"""
Configuration module for the AI Data Analyst Agent.
Handles environment variable loading and Groq client initialization.
Phase 1: Foundation (Groq-only)
"""

import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback for Pydantic v1 or environments without pydantic-settings
    from pydantic import BaseSettings
    class SettingsConfigDict(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


def _get_setting(key: str, default=None):
    """
    Checks Streamlit secrets first (for cloud deployment),
    falls back to environment variables (for local dev).
    """
    try:
        import streamlit as st
        if (
            hasattr(st, "secrets")
            and key in st.secrets
            and not _is_cloud_environment()
        ):
            return st.secrets[key]
    except Exception:
        pass

    if _is_cloud_environment() and key in {"CHROMA_PERSIST_DIR", "EXPORT_DIR"}:
        return default

    return os.environ.get(key, default)


def _is_cloud_environment() -> bool:
    """
    Detects if running on Streamlit Community Cloud.
    Cloud mounts app at /mount/src/ and has no writable
    home directory outside /tmp.
    """
    return (
        os.path.exists("/mount/src") or
        os.environ.get("STREAMLIT_SHARING_MODE") == "1" or
        "/mount/src" in os.environ.get("PWD", "")
    )


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or
    Streamlit secrets. Groq-only configuration.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    groq_api_key: str = ""
    groq_api_url: Optional[str] = None
    model_name: str = "llama-3.3-70b-versatile"
    max_output_tokens: int = 4096
    max_retries: int = 3
    chroma_persist_dir: str = (
        "/tmp/chroma_store" if _is_cloud_environment() else "./memory/chroma_store"
    )
    export_dir: str = (
        "/tmp/exports" if _is_cloud_environment() else "./output/exports"
    )
    max_file_size_mb: int = 50
    debug: bool = False

    def __init__(self, **values):
        values = dict(values)
        values.setdefault("groq_api_key", _get_setting("GROQ_API_KEY", ""))
        values.setdefault("groq_api_url", _get_setting("GROQ_API_URL", None))
        values.setdefault("model_name", _get_setting("MODEL_NAME", "llama-3.3-70b-versatile"))
        values.setdefault("max_output_tokens", _get_setting("MAX_OUTPUT_TOKENS", 4096))
        values.setdefault("max_retries", _get_setting("MAX_RETRIES", 3))
        values.setdefault(
            "chroma_persist_dir",
            _get_setting(
                "CHROMA_PERSIST_DIR",
                "/tmp/chroma_store" if _is_cloud_environment() else "./memory/chroma_store"
            )
        )
        values.setdefault(
            "export_dir",
            _get_setting(
                "EXPORT_DIR",
                "/tmp/exports" if _is_cloud_environment() else "./output/exports"
            )
        )
        values.setdefault("max_file_size_mb", _get_setting("MAX_FILE_SIZE_MB", 50))
        values.setdefault("debug", _get_setting("DEBUG", False))
        super().__init__(**values)
        self.chroma_persist_dir = values.get("chroma_persist_dir", self.chroma_persist_dir)
        self.export_dir = values.get("export_dir", self.export_dir)


# Singleton instance
settings = Settings()


def validate_config() -> None:
    """
    Validates that GROQ_API_KEY exists.
    Raises ValueError with a clear message if missing.
    """
    if not settings.groq_api_key or not settings.groq_api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(local) or Streamlit Cloud secrets (deployed). "
            "Get a free key at https://console.groq.com/keys"
        )


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
        model=settings.model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=settings.max_output_tokens,
    )

    result = response.choices[0].message.content

    # Safety check — if response looks like HTML, raise clearly
    if result.strip().startswith("<!") or "<html" in result[:100]:
        raise ValueError(
            f"Groq returned HTML instead of text. "
            f"Check GROQ_API_KEY. "
            f"Response preview: {result[:200]}"
        )

    return result


try:
    from groq import APIStatusError as _GroqAPIStatusError
except Exception:
    _GroqAPIStatusError = Exception


@retry(
    retry=retry_if_exception_type(_GroqAPIStatusError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def generate_content(prompt: str, system_instruction: str = "") -> str:
    """
    Central function for all LLM calls in the project.
    All modules must call this instead of calling the Groq
    client directly. This makes retry logic and error handling
    centralised.

    Returns the response text as a string.
    Raises the original exception if all retries fail.
    """
    text = _generate_with_groq(prompt, system_instruction)

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