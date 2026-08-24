#!/usr/bin/env python3
"""
Centralized LLM Provider Adapter Layer.
Normalizes payloads and tracking metrics between OpenRouter and Gemini backends.
"""
import os
from openai import OpenAI
from bash_agent import config


# Thread-safe cache for LLM client instances keyed by backend name
_CLIENT_CACHE = {}

def get_backend(model_name: str) -> str:
    """Infers which LLM backend to use based on model name and available API keys."""
    if model_name and model_name.startswith("google") and config.GEMINI_API_KEY:
        return "gemini"
    return "openrouter"


def get_llm_client(backend: str = "openrouter"):
    """Fetches a cached client or initializes one if it doesn't exist."""
    if backend not in _CLIENT_CACHE:
        if backend == "gemini":
            _CLIENT_CACHE[backend] = OpenAI(
                api_key=config.GEMINI_API_KEY, 
                base_url=config.GEMINI_BASE_URL
            )
        else:
            _CLIENT_CACHE[backend] = OpenAI(
                api_key=config.OPENROUTER_API_KEY, 
                base_url=config.OPENROUTER_BASE_URL
            )
            
    return _CLIENT_CACHE[backend]


def get_attribution_headers() -> dict:
    """Returns OpenRouter App Attribution headers for identifying this app
    on OpenRouter's analytics dashboards and public model rankings.
    These headers are only meaningful for OpenRouter-backed requests."""
    return {
        "HTTP-Referer": config.APP_URL,
        "X-OpenRouter-Title": config.APP_TITLE,
        "X-OpenRouter-Categories": config.APP_CATEGORIES,
    }


def normalize_model_string(model_name: str) -> str:
    """Strips provider prefixes when using the Gemini direct endpoint."""
    backend = get_backend(model_name)
    if backend == "gemini" and "/" in model_name:
        # e.g., "google/gemini-3-flash-preview" -> "gemini-3-flash-preview"
        return model_name.split("/")[-1]
    return model_name


def calculate_gemini_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Manually calculates cost per million tokens for Gemini models."""
    # 2026 Placeholder Pricing Tiers (per 1,000,000 tokens)
    pricing_tiers = {
        "gemini-3-flash": {"input": 0.075, "output": 0.30},
        "gemini-3.1-flash-lite": {"input": 0.03, "output": 0.12},
        "default": {"input": 0.15, "output": 0.60}
    }
    
    tier = pricing_tiers.get("default")
    for key in pricing_tiers:
        if key in model_name:
            tier = pricing_tiers[key]
            break
            
    input_cost = (prompt_tokens / 1_000_000) * tier["input"]
    output_cost = (completion_tokens / 1_000_000) * tier["output"]
    return input_cost + output_cost


def create_chat_completion(model, messages, max_tokens=None, extra_body=None, reasoning_effort=None):
    """Wraps and normalizes OpenAI-compatible chat completion calls."""
    backend = get_backend(model)
    client = get_llm_client(backend)
    target_model = normalize_model_string(model)
    
    kwargs = {
        "model": target_model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    # Payload Normalization: Drop OpenRouter specific parameters for Gemini backend
    if backend == "openrouter":
        ebody = extra_body or {}

        # OpenRouter App Attribution headers (identify this app on OpenRouter)
        kwargs["extra_headers"] = get_attribution_headers()

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
    
    # Cost Tracking Normalization for Gemini
    if backend == "gemini" and hasattr(response, "usage"):
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        total_cost = calculate_gemini_cost(target_model, prompt_tokens, completion_tokens)
        
        # Monkey-patch response.model_dump to transparently pass cost downstream
        original_model_dump = response.model_dump
        def patched_model_dump(*args, **kwargs):
            dumped = original_model_dump(*args, **kwargs)
            if "usage" not in dumped or dumped["usage"] is None:
                dumped["usage"] = {}
            dumped["usage"]["cost"] = total_cost
            return dumped
        response.model_dump = patched_model_dump

    return response


def create_embedding(model, input_texts):
    """Wraps and normalizes OpenAI-compatible embedding calls."""
    backend = get_backend(model)
    client = get_llm_client(backend)
    target_model = normalize_model_string(model)
    kwargs = {"model": target_model, "input": input_texts}
    # OpenRouter App Attribution headers (only meaningful for OpenRouter)
    if backend == "openrouter":
        kwargs["extra_headers"] = get_attribution_headers()
    return client.embeddings.create(**kwargs)
