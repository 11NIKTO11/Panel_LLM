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

**In transition (2026-11-11):** prepare, the run and the analysis were reshaped - `python
prepare.py sod` writes `data/sod/values.csv` and `data/sod/prompts/`, `python run.py` sends a
run, `python analyze.py` and `python compare.py` read runs back; see `DECISIONS.md` §7, §8
and §9. `sod` and `regions` are ported; `sod_en` and `grid` are not, and their code left the tree
on 2026-11-13 until they are (`DECISIONS.md` §12, last at commit `ccea2c6`), so the sections below
that mention `data/profiles/`, `config/descriptions/` or `datasets/schema.json` describe the
previous shape.

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
| 1 | `python prepare.py sod` | `.sav` -> `data/sod/values.csv`, then `data/sod/prompts/*.csv` |
| 2 | `python run.py <name>` | subjects -> LLM -> `results/<topic>/` |
| 3 | `python analyze.py <topic> <run>` | one saved run -> tables and figures |
| 4 | `python compare.py <topic> <run> <run> ...` | several runs of a topic side by side |
| 5 | `python benchmark.py <topic> <run> ...` | runs against a model trained on the subjects' own answers |

`analyze.ipynb`, `compare.ipynb` and `benchmark.ipynb` are steps 3 to 5 one block at a time. `run.py` replaced
`simulate.ipynb` and `analyze.py` replaced `panel_stats.ipynb`; both are gone. Nothing is defined in a
notebook; the code is in `panel/`.

### 2.1 Declaring a run

A run is the base run with presets laid over it, all data in `config/presets.json`:

```json
{
    "base": {"dataset": "sod", "description": "default", "topic": "election_2021", "framing": "cs",
             "question": "default", "shape": "Prob", "repeats": 1,
             "client": "Fake", "model": "fake-model", "temperature": 0.0, "effort": null},
    "data": {
        "sod-extended": {"dataset": "sod", "description": "extended"},
        "direct":       {"dataset": "regions", "description": "direct", "question": null, "repeats": 10}
    },
    "models": {
        "nano":      {"client": "OpenRouter", "model": "gpt-4.1-nano", "temperature": 0.0},
        "haiku-5.5": {"client": "Anthropic", "model": "claude-haiku-5-5", "temperature": null, "effort": "low"}
    }
}
```

- the **base** is a complete run and states every field; its model is the Fake client, so a
  run that names no model sends nothing.
- a **data preset** (`-d`) says what is asked, of whom and in what form; a **model preset**
  (`-m`) says which model answers and how. Each states only the fields of its own kind that
  it sets, so the two never state the same field.
- later replaces earlier: base, `-d`, `-m`, single fields (`--question neutral`).
- how many subjects a run takes is in none of them: it is an option.

```bash
python run.py                                    # lists the base and the presets, every field
python run.py -d sod -m nano                     # a data preset and a model preset
python run.py -m nano                            # the base's data, that model
python run.py -d citizen                         # that data, the base's model: offline
python run.py -d sod -m nano --sample 200        # a trial: 200 subjects; `none` is no value
python run.py -d sod -m nano --question neutral  # with a field replaced
python run.py -d sod -m nano --show              # the system prompt and one prompt; no call
python run.py -d sod -m nano --one               # one call, printed; nothing saved
python run.py -d sod -m nano --check             # look over the saved run; no call
python run.py --sample 20                        # the whole path offline
```

Every field of a run is an option. The
script prints the stem and the number of calls, says when the stem already holds a run, and
asks before sending to anything but the Fake client (`--yes` skips that). After saving it
prints a check of the run - whether every call was answered, how many prompts were identical
to another subject's, whether turnout and the option
probabilities sum to 1, the mean and the top-choice share per option - a sanity check and a
first reading, not a comparison with any reference.

`resolve` loads what the names point at and lays out the calls; `execute` runs the batch, reruns
skipped calls a capped number of times, and saves twice - an archived copy, then the plain files.
The record beside them holds the spec, the template, question and framing text, the response
schema and the exact subject ids, so a saved run says how it was produced.

**Cost warning.** A full run makes one API call per subject - 3880 for the survey. Set `sample`
for a cheap trial; its results land under a `_subset=<n>` stem and never overwrite a full run.

### 2.2 Analyzing

```bash
python analyze.py                                    # every topic's saved runs
python analyze.py election_2021 <run>                # all analyses of one run
python analyze.py election_2021 <run> --only effects --predictors gender,age,region --delta 0.001
python compare.py election_2021 <run> <run> --name haiku
python compare.py --set haiku                        # a named set of config/comparisons.json
python benchmark.py election_2021 <run> <run> --name haiku --predictors gender,age,region
```

A run is named by its stem or by any text only one stem contains. Tables go to
`results/<topic>/analysis/`, figures to `results/<topic>/plots/`; `--no-save` writes nothing,
`--plain` ignores the topic's `plot.json`.

| analysis | what it gives |
|---|---|
| `answers` | the mean of every column and how often each option is an answer's highest |
| `country` | predicted, claimed and actual over all the run's subjects, with intervals |
| `distance` | MAE, RMSE, turnout difference, JS, KL and correlation between the three |
| `area` | every distance metric per area, with subject counts; areas in the dataset's declared order (regions by population) |
| `spread` | each column's distribution over subjects; distance from the mean answer |
| `correlation` | correlation and covariance between options |
| `groups` | the prediction at each level of every grouping column |
| `effects` | which columns the answers follow: MANOVA and PERMANOVA |
| `compass` | the model's map of the options against expert positions (CHES 2019) |

`compare.py` has `shares`, `distances` (polls included), `areas`, `spread`, `effects` and
`compass` across the named runs. `--unweighted` on either script sums option shares without
weighting them by the probability of voting; its files get the suffix `_unweighted`. Claimed is counted from the run's own subjects; intervals are bootstrap
over subjects. What was decided and what changed from the earlier statistics is `DECISIONS.md` §9.

`benchmark.py` scores runs subject by subject against what each subject claimed, beside a
ridge logit trained on those claims (cross-validated) and the plain claimed shares: `scores`
(log-loss, Brier), `knowledge` (what the description bought; how much of the trained model's
knowledge a run holds), `agreement` and `calibration`. It needs a full run to mean anything:
on 200 subjects the trained model does not beat the claimed shares.

### 2.3 Tests

```bash
python -m pytest
```

Prepare, the topics, the run and the analysis are covered: descriptions render, schemas match
their snapshots (`tests/fixtures/`, see its README), an offline run writes every file it
should and its record round-trips, and every analysis and comparison runs on fake runs. In
transition: the English edition has no tests until it is ported. The legacy runs, migrated to today's shape
(`DECISIONS.md` §10), are the regression fixture: their frozen aggregates must hold. Tests that
need gitignored inputs (`results/`, `data/sod/`) skip by name on a machine without them.

---

## 3. The axes

A run is six names: a dataset, a description, a topic, a framing, a question, and a model. Each
is a file. The question is `config/topics/<topic>/questions/<name>.json`, one file holding every
language; its wording may depend on the subject (`by`), as the Czech one does on gender.

### 3.1 Datasets - `datasets/<name>/`

| dataset | subjects | language | prepared by |
|---|---|---|---|
| `sod` | 3880 survey respondents, keyed by the survey's `id` | cs | `python -m datasets.sod.prepare` |
| `sod_en` | the same respondents, the survey's English edition | en | not ported, out of the tree |
| `grid` | 288 synthetic profiles, the full factorial of `grid_income3.json` | cs | not ported, out of the tree |
| `regions` | the country and its 14 regions, for place-level prompts | cs | `python prepare.py regions` |

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
| `election_2021` | `cs_legacy` | frozen - what the runs made before the system prompt existed were shown (nine are kept); with question `legacy`. Until 2026-11-13 a topic of its own, `election_2021_legacy` |
| `election_2025` | `cs` | prepared - 8 parties; needs an actual-results CSV before it can be evaluated |
| `media_fees_2026` | `cs` | prepared - the colleague's issue question; two options, no turnout |

- `topic.json` is the language-neutral half: option ids (ASCII, the result columns) with their
  native names and foreign aliases, the turnout roles, and the reference results with their key
  column and total row.
- a framing (`cs.json`, `en.json`) is the system prompt, the label overrides and the wording
  of the response schema. The system prompt says only what holds for every run of the topic;
  the schema text says what the answer looks like.
- a question (`questions/<name>.json`) is what follows the description, per language.
- **the order of the options is the order the model sees them** - it becomes the `enum` order in
  the schema sent to the provider, so it is prompt content, not presentation.
- to add a topic, copy the closest folder and change `name` and `options` in `topic.json`, then
  `instructions` in `cs.json` and the text in `questions/default.json`. The loaders refuse unknown keys, a name that
  disagrees with the folder, a label for an unknown id, duplicate labels and reused aliases.

### 3.4 Models

The client (`OpenAI`, `Anthropic`, `AnthropicLegacy`, `Gemini`, `OpenRouter`, `Fake`), the
provider's model id, the temperature, the effort and the number of repeats. The `Fake` client
answers offline and deterministically.

- `Anthropic` takes no temperature (`"temperature": null`); `AnthropicLegacy` pins the API
  version that still accepts one, for Opus 4.6, Sonnet 4.6 and Haiku 4.5.
- `effort` (`low` .. `max`) is sent by the two Anthropic clients only; any other client
  refuses a run that sets it. Haiku 4.5 takes none.

---

## 4. Output

| path | contents | in git |
|---|---|---|
| `data/profiles/<dataset>.csv` | one row per subject, keyed by `id`, checked against the schema | survey editions no, grid and regions yes |
| `data/prompts/<dataset>/<description>.csv` | the rendered descriptions, keyed by `id` | yes |
| `data/election_data.csv` | official 2021 result, country + 14 regions | yes |
| `data/sod_legacy/values.csv` | dataset `sod_legacy`: the survey as cleaned for the runs of 2025, the subjects and ids of `sod`; prepared data only, never rebuilt | no |
| `data/sod_legacy/prompts/default.csv` | its one description, stored as sent in 2025 (the same text as `data/sod/prompts/legacy.csv`) | no |
| `data/grid/` | the grid's tables in the September shape, parked until the grid is ported | profile and prompts yes |
| `results/<topic>/` | model predictions, run records, derived tables and figures | no |

One run is four files sharing a stem, plus two archived copies:

```
results/election_2021/
├── sod_default_cs_default_OpenRouter_gpt-4.1-nano_Prob_t=0.0.csv             one row per call: id, [repeat], options, Prompt
├── sod_default_cs_default_OpenRouter_gpt-4.1-nano_Prob_t=0.0.json            the same, as returned
├── sod_default_cs_default_OpenRouter_gpt-4.1-nano_Prob_t=0.0_normalized.csv  turnout and options each rescaled to sum to 1
├── sod_default_cs_default_OpenRouter_gpt-4.1-nano_Prob_t=0.0_run.json        the record
├── analysis/   derived tables, named after the run they came from (regenerable)
├── plots/      figures, named the same way (regenerable)
└── archive/    timestamped copies of the CSV and the record, one set per save
```

The stem is `<dataset>_<description>_<framing>_<question>_<Client>_<model>_<Shape>_t=<t>`, plus
`_e=<effort>` when the run sets one, `_x<repeats>` for a repeated run and `_subset=<n>` for a
sampled one; every part is the artifact's `short`. A spec with `temperature=None` sends no
temperature and reads `t=default`; the `Anthropic` client requires it. The topic is the folder.
Legacy stems lack the description, framing and question parts and say `VProb` / `VPick`;
their records keep their older shape, and their CSV headers are native names that `Topic.rename`
turns into ids in memory.

---

## 5. Status

Prepare, the run and the analysis are in their reshaped form (`DECISIONS.md` §7, §8, §9).
The only full `sod` runs in `results/` are the nine legacy ones (eight GPT, one Gemini), migrated
and read by the analysis; for gpt-4.1 the regional MAE equals the 2025 table, the rest is not yet
compared with the earlier statistics. Sent live since the overhaul: two 200-subject trials through
the Anthropic clients (Haiku 4.5 with a temperature, Haiku 5.5 with effort `low`), removed from
`results/` on 2026-11-13 with the other trial runs, so the named set `haiku` of
`config/comparisons.json` and the default `STEM` of `analyze.ipynb` find no run until full Haiku
runs exist; OpenAI, Gemini and OpenRouter have not been
called since the overhaul. `election_2025` has no reference results yet, and the English
texts have not been reviewed.
