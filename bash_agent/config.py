import os

# Model & API Configuration

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
#DEFAULT_MODEL = "openai/gpt-oss-120b"
#DEFAULT_MODEL = "qwen/qwen3-coder-next"
#DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
#DEFAULT_MODEL = "xiaomi/mimo-v2.5"

#DEFAULT_MODEL = "xiaomi/mimo-v2.5-pro"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

# Audio-enabled default model. Selected when the --audio CLI flag is passed,
# overriding DEFAULT_MODEL (but still below an explicit --model / config.json /
# OPENROUTER_MODEL override). Chosen because xiaomi/mimo-v2.5 accepts audio
# as an input modality.
DEFAULT_AUDIO_MODEL = "xiaomi/mimo-v2.5"

# Limits & Timeouts
HISTORY_FILE = os.path.abspath(".bash_agent_tmp/history.json")
# Fallback context ceiling (TOKENS) used when the OpenRouter /models probe for
# the selected model's context_length fails or is unavailable. The Agent (via
# bash_agent.token_budget.resolve_budget) derives the real per-session ceiling
# from the model's context_length — defaulting to a QUARTER of it, overridable
# with the --token-budget CLI flag (absolute tokens or a % of the window) — so
# this constant is only the safety net that keeps sessions bounded when
# offline/unknown. 327,680 = ¼ of the DeepSeek deepseek-v4-flash-0731
# context_length (1,310,720 tokens).
CONTEXT_LIMIT = 327680
CONTEXT_WARN_PERCENT = 99
SCRATCHPAD_LIMIT = 80000
OUTPUT_LIMIT = 10000
MAX_PIXELS = 2_000_000  # Maximum image resolution (2MP) for vision/multimodal features
MAX_CODE_BLOCKS = 1      # Maximum number of code blocks executed per LLM response
# Default per-block timeout (seconds) when the LLM does not specify a
# per-command directive. A single BASH/PYTHON block may extend this up to
# MAX_COMMAND_TIMEOUT via a first-line `# timeout: N` directive.
BASH_TIMEOUT = 60 # seconds
# Hard ceiling (seconds) for the optional per-command `# timeout: N`
# directive. Values above this are clamped down (with an advisory note).
MAX_COMMAND_TIMEOUT = 600

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
    "deepseek/deepseek-v4-pro-0813": ["deepseek"],
    "deepseek/deepseek-v4-flash-0731": ["fireworks","deepseek"],
    "deepseek/deepseek-v4-flash-vision-exp": ["fireworks"],
    "deepseek/deepseek-v4.1-flash": ["io-net","deepseek"],
    "xiaomi/mimo-v2.5-pro": ["novita","xiaomi"],
    "xiaomi/mimo-v2.5": ["xiaomi"],
    #"moonshotai/kimi-k3": ["modal","baseten"]
}

# OpenRouter App Attribution
# Identifies this application on OpenRouter's analytics dashboards and public model rankings.
APP_URL = "https://github.com/geraldnilles/Bash-Agent"
APP_TITLE = "Bash Agent"
APP_CATEGORIES = "cli-agent"
