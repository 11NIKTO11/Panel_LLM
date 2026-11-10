# Panel_LLM

Predicting voting behaviour with LLMs from survey data, and comparing the predictions against
the official election result.

Each subject - a survey respondent, a synthetic profile, or a whole region - is turned into a
self-description; an LLM returns a structured probability distribution over turnout and party
choice; the predictions are aggregated and compared against the official result and against what
respondents claimed they voted.

Everything the model is shown is data: the subjects (`datasets/`), the description template
(`config/descriptions/`), the options and the wording per language (`config/topics/`). A run
names one of each plus a model, so a new election, a new issue question, a new description or a
new language is a config change, never a code edit.

**In transition (2026-11-10):** the prepare step was reshaped - `python prepare.py sod` writes
`data/sod/values.csv` and `data/sod/prompts/`; see `DECISIONS.md` §7. The run, the analysis
notebooks and the other datasets are not ported yet, so the sections below that mention
`data/profiles/`, `config/descriptions/` or `datasets/schema.json` describe the previous shape.

Contributors: `CLAUDE.md` maps the modules, `AGENTS.md` sets the working conventions,
`DECISIONS.md` records why things are the way they are.

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

- download the [Society of Distrust (CZ) dataset](https://archivdv.soc.cas.cz/dataset.xhtml?persistentId=doi:10.14473/CSDA/ZKWKVZ)
  and place the `.sav` files at

  ```
  ./datasets/sod/SoD-CZ/Society_of_Distrust - CZ.sav
  ./datasets/sod/SoD-EN/Society_of_Distrust - EN.sav      only the sod_en dataset needs this one
  ```

- a different location or name is `--sav PATH` on the prepare script.

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
`UnicodeEncodeError` otherwise. An artefact of the console, not of the data.

### 1.4 The `datasets` folder

`datasets/` is a plain folder of this repo. If the HuggingFace `datasets` package is ever
installed into this venv, the two collide on import. Do not install it here.

---

## 2. Running

**Run everything from the repo root.** Imports resolve against it; data paths do not depend on
it (`panel/paths.py` anchors them on the package).

| step | command or notebook | what it does |
|---|---|---|
| 1 | `python -m datasets.sod.prepare` | `.sav` -> `data/profiles/sod.csv`, then `data/prompts/sod/*.csv` |
| 1 | `python -m datasets.grid.generate`, `datasets.regions.prepare`, `datasets.sod_en.prepare` | the other datasets' profiles and tables |
| 2 | `simulate.ipynb` | subjects -> LLM -> `results/<topic>/` |
| 3 | `analyze.ipynb` | evaluate one run: MAE by region, covariance, distributions, every run of the topic |
| 4 | `panel_stats.ipynb` | statistical layer: MANOVA, PERMANOVA, distances by area, spread of options |

Every notebook restarts and runs to the end offline: `simulate.ipynb` through the fake client,
the other two over the saved legacy runs. Nothing is defined in a notebook; the code is in `panel/`.

### 2.1 Declaring a run

The run-setup cell of `simulate.ipynb`, right after the imports, is the only place a run is
declared: one `RunSpec` of names and numbers, a dict of named variants, and the one to run:

```python
BASE = RunSpec(dataset="sod", description="self_v1", topic="election_2021", framing="cs",
               client="OpenRouter", model="gpt-4.1-nano", shape="VProb", temperature=0)
SPECS = {"nano": BASE, "4o": BASE.variant(model="gpt-4o"), "nano-trial": BASE.variant(sample=200)}
SPEC = SPECS["nano"]

RUN = resolve(SPEC)                                  # loads the five artifacts, joins profile and prompt table
STORE = ResultStore(paths.RESULTS_DIR, SPEC.topic)
execute(RUN, STORE, client_class(SPEC.client)())     # batch, capped rerun of skipped calls, archived + plain save
```

`resolve` loads what the names point at and lays out the calls; `execute` runs the batch, reruns
skipped calls a capped number of times, and saves twice - an archived copy, then the plain files.
The record beside them holds the spec, the template and framing text, the response schema and the
exact subject ids, so a saved run says how it was produced.

**Cost warning.** A full run makes one API call per subject - 3880 for the survey. Use
`SPECS["nano-trial"]`, which sets `sample=200`, for a cheap trial; its results land under a
`_subset=200` stem and never overwrite a full run. `SPECS["fake"]` costs nothing and exercises
the whole path.

### 2.2 Tests

```bash
python -m pytest
```

128 tests in about eight seconds. They hold the code to what it produced before the overhaul
(`tests/fixtures/`, see its README): the survey renders identically on every row, the legacy
results reparse and aggregate to the same numbers, the frozen legacy schema is reproduced byte
for byte, and an offline run writes every file it should. Tests that need gitignored inputs
(`results/`, `data/profiles/sod.csv`) skip by name on a machine without them.

---

## 3. The axes

A run is five names: a dataset, a description, a topic, a framing, and a model. Each is a file.

### 3.1 Datasets - `datasets/<name>/`

| dataset | subjects | language | prepared by |
|---|---|---|---|
| `sod` | 3880 survey respondents, keyed by the survey's `id` | cs | `python -m datasets.sod.prepare` |
| `sod_en` | the same respondents, the survey's English edition | en | `python -m datasets.sod_en.prepare` |
| `grid` | 288 synthetic profiles, the full factorial of `grid_income3.json` | cs | `python -m datasets.grid.generate` |
| `regions` | the country and its 14 regions, for place-level prompts | cs | `python -m datasets.regions.prepare` |

- `dataset.json` declares the language of the values, the columns every row must carry, the
  descriptions rendered for it, and per topic a *claims map* (`claims/<topic>.json`) sorting the
  dataset's own answer strings into option ids, abstained and excluded.
- every profile is checked against `datasets/schema.json`, the shared vocabulary of concepts and
  their representation columns (`age` is an exact age, `age_range` a band), so a column means the
  same thing in every dataset that carries it. A concept may declare common levels for grouping
  across datasets; `age` does.

### 3.2 Descriptions - `config/descriptions/`

| description | language | for |
|---|---|---|
| `self_v1` | cs | the first-person self-description the legacy runs were built from; renders the survey and the grid |
| `self_v2` | cs | `self_v1` plus EU, NATO and covid answers |
| `self_v1_en` | en | the English counterpart of `self_v1`, concept for concept |
| `latent_citizen_v1` | cs | "I am an inhabitant of <region>" |
| `latent_direct_v1` | cs | the whole prompt for a place; `standalone`, so no question follows it |

- a template is a list of sentences, each a list of alternatives in priority order; the first
  whose slots are all filled wins. That is how a sentence degrades when an answer is missing and
  how one template serves datasets with different columns.
- `{column|filter}` passes a value through a filter of the description's language pack
  (`panel/describe/lang/<code>/`), which is where grammar lives.
- a variant is additive: new wording means a new file and a new name, never an edit.

### 3.3 Topics and framings - `config/topics/<topic>/`

| topic | framings | state |
|---|---|---|
| `election_2021` | `cs`, `en` | current - 9 parties, turnout, official 2021 result |
| `election_2021_legacy` | `cs` | frozen - the ten runs made before the system prompt existed |
| `election_2025` | `cs` | prepared - 8 parties; needs an actual-results CSV before it can be evaluated |
| `media_fees_2026` | `cs` | prepared - the colleague's issue question; two options, no turnout |

- `topic.json` is the language-neutral half: option ids (ASCII, the result columns) with their
  native names and foreign aliases, the turnout roles, and the reference results with their key
  column and total row.
- a framing (`cs.json`, `en.json`) is everything the model reads that is not the description:
  system prompt, question, label overrides, and the wording of the response schema.
- **the order of the options is the order the model sees them** - it becomes the `enum` order in
  the schema sent to the provider, so it is prompt content, not presentation.
- to add a topic, copy the closest folder and change `name` and `options` in `topic.json`, then
  `instructions` and `question` in `cs.json`. The loaders refuse unknown keys, a name that
  disagrees with the folder, a label for an unknown id, duplicate labels and reused aliases.

### 3.4 Models

The client (`OpenAI`, `Anthropic`, `Gemini`, `OpenRouter`, `Fake`), the provider's model id, the
temperature and the number of repeats. The `Fake` client answers offline and deterministically.

---

## 4. Output

| path | contents | in git |
|---|---|---|
| `data/profiles/<dataset>.csv` | one row per subject, keyed by `id`, checked against the schema | survey editions no, grid and regions yes |
| `data/prompts/<dataset>/<description>.csv` | the rendered descriptions, keyed by `id` | yes |
| `data/election_data.csv` | official 2021 result, country + 14 regions | yes |
| `data/legacy_clean_data.csv` | the frozen table the legacy runs were built from, keyed by row position | no |
| `data/prompts.csv` | the descriptions the ten legacy runs were sent | yes |
| `results/<topic>/` | model predictions, run records, derived tables | no |
| `graphs/` | generated figures | no |

One run is four files sharing a stem, plus two archived copies:

```
results/election_2021/
├── sod_v1_cs_OpenRouter_gpt-4.1-nano_VProb_t=0.0.csv             one row per call: id, [repeat], options, Prompt
├── sod_v1_cs_OpenRouter_gpt-4.1-nano_VProb_t=0.0.json            the same, as returned
├── sod_v1_cs_OpenRouter_gpt-4.1-nano_VProb_t=0.0_normalized.csv  turnout and options each rescaled to sum to 1
├── sod_v1_cs_OpenRouter_gpt-4.1-nano_VProb_t=0.0_run.json        the record
├── analysis/   derived tables, named after the run they came from (regenerable)
└── archive/    timestamped copies of the CSV and the record, one set per save
```

The stem is `<dataset>_<description>_<framing>_<Client>_<model>_<Shape>_t=<t>`, plus
`_x<repeats>` for a repeated run and `_subset=<n>` for a sampled one; every part is the
artifact's `short`. A spec with `temperature=None` sends no temperature and reads `t=default`;
Claude models require it. The topic is the folder. Legacy stems lack the description and framing parts;
their records keep their older shape, and their CSV headers are native names that `Topic.rename`
turns into ids in memory.

---

## 5. Status

The architecture overhaul of 2026-09-09 is complete on this branch and awaits review; the
decisions and their reasons are in `DECISIONS.md`. Nothing here has been sent to a provider since:
the live smoke call per provider is still owed and belongs to the first real run. `election_2025`
has no reference results yet, and the English framing's wording has not been reviewed.
