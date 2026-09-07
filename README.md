# Panel_LLM

Predicting Czech voting behaviour with LLMs from survey data, and comparing the predictions
against the official 2021 election result.

Each respondent of the *Society of Distrust* (CZ) survey is turned into a Czech-language
self-description; an LLM returns a structured probability distribution over turnout and
party choice; the predictions are aggregated and compared against both the official result
and what respondents claimed they voted.

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
  ./SoD-CZ/Society_of_Distrust - CZ.sav
  ```

- a different location or name means editing the path in `SoD/dataset_preparation.ipynb`.

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

---

## 2. Running

**Run everything from the repo root** — several modules use paths relative to it.

```bash
jupyter lab
```

Then, in order:

| step | notebook | what it does |
|---|---|---|
| 1 | `SoD/dataset_preparation.ipynb` | `.sav` → `data/clean_data.csv` |
| 2 | `election_simulation.ipynb` | respondents → LLM → `results/*.json` + `*.csv` |
| 3 | `election_results_analysis.ipynb` | evaluate one run: MAE by region, covariance, distributions |
| 4 | `panel_analysis.ipynb` | statistical layer: MANOVA, PERMANOVA, distribution distances |

Also:

- `election_preferences.ipynb` — region-level prompts with no respondent profile, to probe
  what the model "believes" about a region directly.
- `experiments/artificial_matrix/generate_grid.py` — build a synthetic full-factorial
  profile grid instead of using real respondents:

  ```bash
  python experiments/artificial_matrix/generate_grid.py --config experiments/artificial_matrix/grid_config_sod.json
  ```

**Cost warning.** Step 2 makes one API call per respondent — 3880 by default. Its cell 15
has a commented `data.sample(n=200)` for a cheap trial run. Which model, client and response
model a run uses is selected in cell 3 by commenting lines in and out.

---

## 3. Output

| path | contents | in git |
|---|---|---|
| `data/clean_data.csv` | one cleaned row per respondent | no |
| `data/election_data.csv` | official 2021 result, country + 14 regions | yes |
| `data/prompts.csv` | the generated description per respondent | yes |
| `results/` | model predictions, aggregates, matrices | no |
| `graphs/` | generated figures | no |

Result filenames encode the run:
`voting_results_OpenAIClient_gpt-4o_VotingProbabilities_n=3880_t=0.0.csv` —
client, model, response model, respondent count, temperature. A `normalized_` prefix means
attendance and party columns were each rescaled to sum to 1 per row.

---

## 4. Status

Under active refactoring. `_Refactor_Plan.md` holds the analysis, the target structure and
the step order; `CLAUDE.md` §5 lists the known problems worth being aware of before
trusting a number — including a label transposition in the regional error heatmaps and
duplicate function definitions in `panel_analysis.ipynb`.
