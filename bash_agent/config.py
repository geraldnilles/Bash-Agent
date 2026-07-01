import os

# Model & API Configuration

# Gemini Native Standalone Settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Experimental Model Slugs (uncomment to use as override, or refer for testing)
#DEFAULT_MODEL = "google/gemma-4-31b-it"
#DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
#DEFAULT_MODEL = "minimax/minimax-m3"
#DEFAULT_MODEL = "xiaomi/mimo-v2.5"
#DEFAULT_MODEL = "stepfun/step-3.7-flash"
#DEFAULT_MODEL = "tencent/hy3-preview"
#DEFAULT_MODEL = "google/gemma-4-31b-it:free"
#DEFAULT_MODEL = "z-ai/glm-5.1"
#DEFAULT_MODEL = "minimax/minimax-m2.7"
#DEFAULT_MODEL = "deepseek/deepseek-v3.2"
#DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
#DEFAULT_MODEL = "openrouter/elephant-alpha"
#DEFAULT_MODEL = "xiaomi/mimo-v2-flash"
#DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
#DEFAULT_MODEL = "openai/gpt-oss-120b"
#DEFAULT_MODEL = "google/gemini-3-flash-preview"
#DEFAULT_MODEL = "qwen/qwen3-coder-next"
#DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
#DEFAULT_MODEL = "xiaomi/mimo-v2.5"
#DEFAULT_MODEL = "google/gemini-3.5-flash"

#DEFAULT_MODEL = "xiaomi/mimo-v2.5-pro"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"

# Limits & Timeouts
HISTORY_FILE = os.path.abspath(".bash_agent_tmp/history.json")
CONTEXT_LIMIT = 256000
SCRATCHPAD_LIMIT = 80000
OUTPUT_LIMIT = 10000
MAX_PIXELS = 2_000_000  # Maximum image resolution (2MP) for vision/multimodal features
BASH_TIMEOUT = 60 # seconds

# Session Budget
DEFAULT_BUDGET = 0.10 # USD

# Console Colors
COLOR_CMD = "\033[96m"    # Cyan for Bash Commands
COLOR_OUT = "\033[93m"    # Yellow for Bash Output
COLOR_PY_CMD = "\033[95m"  # Magenta for Python Commands
COLOR_COST = "\033[92m"  # Bright Green for Cost Info
COLOR_RESET = "\033[0m"    # Reset to default terminal color

# OpenRouter Reasoning Effort (none, minimal, low, medium, high)
DEFAULT_REASONING_EFFORT = "low"

# Max output tokens
DEFAULT_MAX_TOKENS = 1024*8

# OpenRouter Provider Whitelists
# Maps model slugs to approved list of providers
MODEL_PROVIDERS = {
    "deepseek/deepseek-v4-pro": ["deepseek"],
    "xiaomi/mimo-v2.5-pro": ["xiaomi"],
}
