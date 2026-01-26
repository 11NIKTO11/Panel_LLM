# Centralized constants and canonical names
# Models
GPT_41_NANO = "gpt-4.1-nano"    # GPT-4.1 Nano (cheapest, fastest)
GPT_4o_MINI = "gpt-4o-mini"     # GPT-4o Mini (cheaper, faster)
GPT_41 = "gpt-4.1"              # GPT-4.1 (flagship reasoning model)
GPT_4o = "gpt-4o"               # GPT-4o (fast, high quality)
GEMINI_3_FLASH = "gemini-3-flash-preview"
CLAUDE_SONNET_45= "claude-sonnet-4-5"

MODELS = [GPT_41_NANO, GPT_4o_MINI, GPT_41, GPT_4o]

OPENAI = "openai"
GOOGLE = "google"
ANTHROPIC = "anthropic"

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
VOTED = "Voted"
NOT_VOTED = "Not Voted"
ATTENDANCE_COLUMNS = [VOTED, NOT_VOTED]

# Parties
ANO = "ANO 2011"
SPOLU = "Koalice Spolu (ODS, TOP 09, KDU-ČSL)"
PIRSTAN = "Koalice PIRÁTI a STAROSTOVÉ"
PIRATI = "Česká pirátská strana"
STAN = "STAROSTOVÉ A NEZÁVISLÍ (STAN)"
SPD = "Svoboda a přímá demokracie – Tomio Okamura (SPD)" #"Svoboda a přímá demokracie (SPD)"
AUTO = "Motoristé sobě (AUTO)"
STACILO = "Stačilo! (KSČM, SD-SN, ČSNS, SOCDEM)"
KSCM = "Komunistická strana Čech a Moravy (KSČM)"
CSSD = "Česká strana sociálně demokratická (ČSSD)"
TSS = "Trikolóra, Svobodní a Soukromníci" #"Trikolóra, Svobodní a Soukromníci (TSS)"
PRISAHA = "Přísaha Roberta Šlachty"
JINA_STRANA = "Jiná strana"

PARTY_COLUMNS_2021 = [
    ANO, SPOLU, PIRSTAN, SPD, PRISAHA, CSSD, KSCM, TSS, JINA_STRANA
]

PARTY_COLUMNS_2025 = [
    ANO, SPOLU, PIRATI, STAN, SPD, AUTO, STACILO, JINA_STRANA
]

ALL_COLUMNS = ATTENDANCE_COLUMNS + PARTY_COLUMNS_2021