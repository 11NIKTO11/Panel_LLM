# Panel_LLM

Predicting Czech voting behaviour with LLMs from survey data, and comparing the predictions
against the official election result.

Each respondent of the *Society of Distrust* (CZ) survey is turned into a Czech-language
self-description; an LLM returns a structured probability distribution over turnout and
party choice; the predictions are aggregated and compared against both the official result
and what respondents claimed they voted.

What the model is asked — the options, the question, the system prompt — is a config file, not
code (§3), so a different election or a different issue question needs no code change.

Contributors: see `CLAUDE.md` for the module map and `AGENTS.md` for working conventions.

---

## 1. Setup

### 1.1 Install

Python 3.11 or newer (3.13 is what the project runs on).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 1.2 Dataset

Not distributed with the repo.

- download the [Society of Distrust (CZ) dataset](https://archivdv.soc.cas.cz/dataset.xhtml?persistentId=doi:10.14473/CSDA/ZKWKVZ).
- extract and place the `.sav` file at:

  ```
  ./datasets/sod/SoD-CZ/Society_of_Distrust - CZ.sav
  ```

- a different location or name means editing the path in `datasets/sod/prepare.ipynb`.

### 1.3 API keys

Create a `.env` in the repo root (gitignored). Only the provider you actually use is needed.

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

OpenRouter reaches OpenAI, Anthropic and Google models through one key, and is what the
project uses for models without a native client configured.

**On Windows:** set `PYTHONIOENCODING=utf-8` before running any script that prints party or
region names. The default console encoding is cp1252 and Czech text raises
`UnicodeEncodeError` — a console artefact, not a bug in the data.

---

## 2. Running

**Run everything from the repo root** — several modules use paths relative to it.

```bash
jupyter lab
```

Then, in order:

| step | notebook | what it does |
|---|---|---|
| 1 | `datasets/sod/prepare.ipynb` | `.sav` → `data/clean_data.csv` |
| 2 | `election_simulation.ipynb` | respondents → LLM → `results/<topic>/` |
| 3 | `election_results_analysis.ipynb` | evaluate one run: MAE by region, covariance, distributions |
| 4 | `panel_analysis.ipynb` | statistical layer: MANOVA, PERMANOVA, distribution distances |

Step 4 is **currently broken on purpose** — it is the last notebook still on the pre-refactor
result names. See `CLAUDE.md` §5.

### 2.1 Declaring a run

The run-setup cell of `election_simulation.ipynb`, right after the imports, is the only place a
run is declared. Pick a topic, build its response models, write one `RunSpec`, and select a
named variant:

```python
TOPIC = load_topic("election_2021", paths.TOPICS_DIR)
VProb, VPick = make_response_models(TOPIC)
DESCRIPTION = "self_description_v1"

people = pd.read_csv(paths.DATA_DIR / "clean_data.csv", index_col=0)
data = people.join(read_table(DESCRIPTION, people.index))   # the descriptions, rendered at cleanup

BASE = RunSpec(
    name="sod", topic=TOPIC.name, data=data,
    description=DESCRIPTION,
    question=TOPIC.question, instructions=TOPIC.instructions,
    client_type=llm.OpenRouterClient, model="gpt-4.1-nano",
    response_model=VProb, temperature=0,
)

SPECS = {"nano": BASE, "4o": BASE.variant(model="gpt-4o"), "nano-trial": BASE.variant(sample=200)}
SPEC = SPECS["nano"]
```

Everything downstream reads the spec, and the spec is written into the run's record, so a saved
run says how it was produced.

**Cost warning.** A full run makes one API call per respondent — 3880 of them. Use
`SPECS["nano-trial"]`, which sets `sample=200`, for a cheap trial; its results land under a
`_subset=200` stem and never overwrite a full run.

### 2.2 Other entry points

- `election_preferences.ipynb` — region-level prompts with no respondent profile, to probe what
  the model "believes" about a region directly. Ten repeats per region, two describer variants.
- `datasets/grid/generate_grid.py` — build a synthetic full-factorial profile grid instead
  of using real respondents. A config naming a `description_template` is a prompt source, and
  its grid is written with the description already in a column:

  ```bash
  python datasets/grid/generate_grid.py --config datasets/grid/grid_config_sod.json
  ```

- `update_script.ipynb` — spent one-off migrations, kept as a record. Each has a `DRY_RUN` flag
  and has already been applied.

---

## 3. Topics

A topic is what a run is about. One JSON file in `config/topics/` owns the options the model may
answer with, the question appended to each description, the system prompt, and the real-world
results to compare against.

| topic | state |
|---|---|
| `election_2021` | current — 9 parties, turnout, official 2021 result |
| `election_2021_legacy` | frozen — the ten runs made before the system prompt existed |
| `election_2025` | prepared — 8 parties; needs an actual-results CSV before it can be evaluated |
| `media_fees_2026` | prepared — the colleague's ČT/ČRo issue question; two options, no turnout |

### 3.1 Adding one

Copy the closest existing file and change four things: `name` (must match the filename),
`instructions`, `question`, and `options`. Then `load_topic("<name>")` and run as in §2.1.

- each option is named once, with its short label and any foreign column aliases beneath it.
- **the order of the options is the order the model sees them** — it becomes the `enum` order in
  the schema sent to the provider, so it is part of the prompt, not presentation.
- `"turnout": null` for a question with no participation dimension; the response model then has
  no turnout field at all.
- the loader refuses unknown keys, a name that disagrees with the filename, duplicate short
  labels and reused aliases, so a typo fails at load instead of producing an empty column.

Two topics that share a vocabulary but were asked differently stay in **separate** topics — the
folder is what keeps their results from overwriting each other, since the filename stem is the
same either way.

---

## 4. Output

| path | contents | in git |
|---|---|---|
| `data/clean_data.csv` | one cleaned row per respondent, keyed by the survey `id` | no |
| `data/legacy_clean_data.csv` | the same table frozen at 2026-09-08, keyed by row position, for the legacy runs | no |
| `data/election_data.csv` | official 2021 result, country + 14 regions | yes |
| `data/prompts/<variant>.csv` | the rendered descriptions, keyed by survey `id` | no |
| `data/prompts.csv` | the descriptions the ten legacy runs were sent | yes |
| `results/<topic>/` | model predictions, run records, derived tables | no |
| `graphs/` | generated figures | no |

One run is four files sharing a stem, plus two archived copies:

```
results/election_2021_legacy/
├── sod_OpenAI_gpt-4o_VProb_t=0.0.csv            predictions, one row per respondent
├── sod_OpenAI_gpt-4o_VProb_t=0.0.json           the same, as returned
├── sod_OpenAI_gpt-4o_VProb_t=0.0_normalized.csv turnout and options each rescaled to sum to 1
├── sod_OpenAI_gpt-4o_VProb_t=0.0_run.json       the record: spec, row count, git commit, time
├── analysis/   derived tables, named after the run they came from
└── archive/    timestamped copies, one set per save
```

The stem is `<dataset>_<client>_<model>_<response model>_t=<temperature>`, plus `_subset=<n>`
for a sampled run. The topic is the folder, not part of the stem. To load a run you need only
its stem — the record supplies everything else, including the topic whose options it used.

---

## 5. Status

Under active refactoring. `_Refactor_Plan.md` holds the analysis, the target structure and the
step order; steps 1–3 are done (bug batch, run objects and the results layout, topics as data).

Before trusting a number, read `CLAUDE.md` §5 — it lists what is knowingly broken, including
`panel_analysis.ipynb` and the regional error figures in `graphs/`, which predate a label fix.
