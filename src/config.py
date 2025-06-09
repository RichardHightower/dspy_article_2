"""Configuration for LLM providers - supports OpenAI, Anthropic, and Ollama."""

import os
from typing import Optional
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()


def configure_llm() -> Optional[dspy.LM]:
    """Configure the LLM based on environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm: Optional[dspy.LM] = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        llm = dspy.LM(model=f"openai/{model}", api_key=api_key, max_tokens=2000)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        llm = dspy.LM(model=f"anthropic/{model}", api_key=api_key, max_tokens=2000)

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "phi3:latest")
        llm = dspy.LM(
            model=f"ollama_chat/{model}",
            base_url=base_url,
            max_tokens=1024,
            temperature=0.0,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    # Configure DSPy with the selected LLM
    dspy.settings.configure(lm=llm)
    print(f"✓ Configured DSPy with {provider} provider using {model} model")

    return llm
