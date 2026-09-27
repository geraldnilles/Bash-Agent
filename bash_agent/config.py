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
DEFAULT_AUDIO_MODEL = "xiaomi/mimo-v2.6-flash"

# Hierarchical multi-agent defaults. The top-level agent acts as an
# ORCHESTRATOR that plans and delegates tactical work to cost-effective
# SUBAGENTS (see AGENTS.md "Hierarchical Orchestrator / Subagent Mode").
# These are the fallbacks used when neither the CLI nor config.json pins a
# per-role model.
DEFAULT_ORCHESTRATOR_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_SUBAGENT_MODEL = "deepseek/deepseek-v4.1-flash"

# Prefix (relative to .bash_agent_tmp/) for subagent history files. Subagents
# get their OWN history file so a nested run never overwrites the parent
# orchestrator's history.json.
SUBAGENT_HISTORY_PREFIX = "history_subagent_"

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
# Fallback session cost limit (USD) for a standalone / subagent session that
# did not receive an explicit budget. A top-level ORCHESTRATOR gets the larger
# DEFAULT_ORCHESTRATOR_BUDGET because it must pay for both its own planning
# turns and every subagent it delegates to.
DEFAULT_BUDGET = 0.10 # USD
# Session cost limit (USD) applied when a top-level orchestrator is started
# without an explicit --budget / constructor budget.
DEFAULT_ORCHESTRATOR_BUDGET = 0.20 # USD
# Fraction of the OVERALL parent budget granted to a spawned subagent by
# default (25%). An orchestrator can override this per delegation with the
# `subagent --budget <USD>` command flag, or globally with --subagent-budget.
DEFAULT_SUBAGENT_BUDGET_FRACTION = 0.25
# Absolute floor (USD) applied to any derived subagent budget so a tiny or
# already-nearly-exhausted parent budget still lets a subagent make progress.
MIN_SUBAGENT_BUDGET = 0.01 # USD

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

# Orchestrator reasoning effort. None means "defer to the model's own built-in
# default reasoning level" (the LLM call omits reasoning.effort), rather than
# forcing a fixed level. Only the top-level orchestrator uses this; subagents
# and standalone/direct agents keep DEFAULT_REASONING_EFFORT.
DEFAULT_ORCHESTRATOR_REASONING_EFFORT = None

# Orchestrator max output tokens (2x) so it has room for longer reasoning and
# larger plan documents. Only the top-level orchestrator uses this; subagents
# and standalone/direct agents keep DEFAULT_MAX_TOKENS.
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 1024*16

# OpenRouter Provider Whitelists
# Maps model slugs to approved list of providers
MODEL_PROVIDERS = {
    "deepseek/deepseek-v4-pro": ["deepseek"],
    "deepseek/deepseek-v4-pro-0813": ["deepseek"],
    "deepseek/deepseek-v4-flash-0731": ["fireworks","deepseek"],
    "deepseek/deepseek-v4-flash-vision-exp": ["fireworks"],
    "deepseek/deepseek-v4.1-flash": ["deepseek","deepinfra","fireworks","novita"],
    "xiaomi/mimo-v2.5-pro": ["novita","xiaomi"],
    "xiaomi/mimo-v2.5": ["deepinfra","novita","xiaomi" ],
    "moonshotai/kimi-k3": ["inference-net"],
}

# OpenRouter App Attribution
# Identifies this application on OpenRouter's analytics dashboards and public model rankings.
APP_URL = "https://github.com/geraldnilles/Bash-Agent"
APP_TITLE = "Bash Agent"
APP_CATEGORIES = "cli-agent"
