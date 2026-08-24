#!/usr/bin/env python3
"""
Centralized LLM Provider Adapter Layer (OpenRouter).
Normalizes payloads and tracking metrics for OpenAI-compatible calls
routed through the OpenRouter API.
"""
from openai import OpenAI
from bash_agent import config


# Cache for the shared OpenRouter client instance
_CLIENT_CACHE = {}


def get_llm_client():
    """Fetches a cached OpenRouter client or initializes one if it doesn't exist."""
    if "openrouter" not in _CLIENT_CACHE:
        _CLIENT_CACHE["openrouter"] = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
    return _CLIENT_CACHE["openrouter"]


def get_attribution_headers() -> dict:
    """Returns OpenRouter App Attribution headers for identifying this app
    on OpenRouter's analytics dashboards and public model rankings."""
    return {
        "HTTP-Referer": config.APP_URL,
        "X-OpenRouter-Title": config.APP_TITLE,
        "X-OpenRouter-Categories": config.APP_CATEGORIES,
    }


def create_chat_completion(model, messages, max_tokens=None, extra_body=None, reasoning_effort=None):
    """Wraps and normalizes OpenAI-compatible chat completion calls."""
    client = get_llm_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,

        # OpenRouter App Attribution headers (identify this app on OpenRouter)
        "extra_headers": get_attribution_headers(),
    }

    ebody = extra_body or {}

    # Inject reasoning preferences if provided
    if reasoning_effort:
        if "reasoning" not in ebody:
            ebody["reasoning"] = {}
        ebody["reasoning"]["effort"] = reasoning_effort

    # Handle dynamic provider whitelisting
    model_providers = getattr(config, "MODEL_PROVIDERS", {})
    if model and model in model_providers:
        ebody["provider"] = {
            "only": model_providers[model],
        }

    if ebody:
        kwargs["extra_body"] = ebody

    response = client.chat.completions.create(**kwargs)
    return response


def create_embedding(model, input_texts):
    """Wraps and normalizes OpenAI-compatible embedding calls."""
    client = get_llm_client()
    return client.embeddings.create(
        model=model,
        input=input_texts,
        extra_headers=get_attribution_headers(),
    )
