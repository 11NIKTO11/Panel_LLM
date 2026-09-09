# Centralized constants and canonical names

RESULTS_DIR = "results"
DATA_DIR = "data"
TOPICS_DIR = "config/topics"

# Models
GPT_41_NANO = "gpt-4.1-nano"    # GPT-4.1 Nano (cheapest, fastest)
GPT_4o_MINI = "gpt-4o-mini"     # GPT-4o Mini (cheaper, faster)
GPT_41 = "gpt-4.1"              # GPT-4.1 (flagship reasoning model)
GPT_4o = "gpt-4o"               # GPT-4o (fast, high quality)
GEMINI_3_FLASH = "gemini-3-flash-preview"
CLAUDE_SONNET_45= "claude-sonnet-4-5"

MODELS = [GPT_41_NANO, GPT_4o_MINI, GPT_41, GPT_4o]

# Provider identifiers moved to panel/llm/base.py (the LLM package owns them).

# Regions
COUNTRY = "Česko"
REGIONS = [
    "Hlavní město Praha",
    "Středočeský kraj",
    "Jihočeský kraj",
    "Plzeňský kraj",
    "Karlovarský kraj",
    "Ústecký kraj",
    "Liberecký kraj",
    "Královéhradecký kraj",
    "Pardubický kraj",
    "Kraj Vysočina",
    "Jihomoravský kraj",
    "Olomoucký kraj",
    "Zlínský kraj",
    "Moravskoslezský kraj",
]
REGIONS_ALL = [COUNTRY] + REGIONS

# Prompt
PROMPT = "Prompt"

# Attendance
# NOTE: a topic's turnout block owns these column names now (config/topics/*.json).
# They stay here until panel_analysis stops importing them (refactor step 6).
VOTED = "Voted"
NOT_VOTED = "Not Voted"
ATTENDANCE_COLUMNS = [VOTED, NOT_VOTED]

# Parties, and the 2021/2025 party lists, moved to config/topics/*.json (refactor step 3).
# Read them with panel.topic.load_topic(name).options / .short / .aliases.