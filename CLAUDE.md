# Panel_LLM — project instructions

What this repo is and how its parts fit together. Read before touching anything. Agent workflow
conventions (commands, tests, git) live in `AGENTS.md`; why things are the way they are in
`DECISIONS.md`; human setup in `README.md`.

**Accuracy note:** rewritten from the actual tree on 2026-09-09, at the end of the architecture
overhaul on branch `refactor_complete`. If it disagrees with the code, the code is right - say so
rather than following it.

**In transition (2026-11-10):** prepare was reshaped (`DECISIONS.md` §7): `python prepare.py sod`
writes `datasets/sod/prepared/values.csv` and `datasets/sod/prepared/prompts/` (prepared data lives in its dataset's folder since 2026-11-14, `DECISIONS.md` §23; there is no `data/`); wording lives in `datasets/sod/forms.json`,
templates in `datasets/sod/descriptions/`; `panel/schema.py`, `panel/describe/lang/`,
`panel/describe/tables.py`, `datasets/schema.json` and `config/descriptions/self_v*.json` are gone.
`sod_en` / `grid` and their tests are NOT ported; since 2026-11-13 their code is out of the tree, in the gitignored `_delete/_to_sort` (`DECISIONS.md` §12), and the old `data/grid/` is in `_delete/data_grid/`. The analysis was ported on 2026-11-11
(`DECISIONS.md` §9): `python analyze.py <topic> <run>` and `python compare.py <topic> <run> <run>`,
`analyze.ipynb` / `compare.ipynb` the same block by block; every analysis is `(run, out) ->
(table, figures)` in `panel/analysis/`, comparisons in `panel/analysis/comparison/`; figures are
matplotlib without pyplot. `python benchmark.py <topic> <run> ...` / `benchmark.ipynb` score runs
subject by subject against a model trained on the subjects' claims; every benchmark is
`(ground, out) -> (table, figures)` in `panel/analysis/benchmark/`. §3.6 below is stale; `panel_stats.ipynb` is superseded. `regions` is
ported (2026-11-11): `python prepare.py regions`, descriptions `citizen` and `direct`
(`direct` is standalone: `question` is None).
§2-§4 below still describe the previous shape for those parts.
Grids are datasets (2026-11-14, `DECISIONS.md` §15): `grid_sod` (new 2026-11-15, incomes cut at 25/40k, NO run yet, §28), `grid_sod_retired` and `grid_sod45_retired` (the former `grid_sod` / `grid_sod45`, kept only for their runs `grid-sod-retired_...x3` / `grid-sod45-retired_...`), `grid_simple`,
`grid_simple_en` (the colleague's grid, values only; `grid_colleague` until 2026-11-15, `DECISIONS.md` §27); `"source": "dataset.json"`, values = `panel/grid.py:
full_factorial` of `levels.order`, which each grid's `build.py` names. Their runs are in `results/` since 2026-11-14 (§15.1, `migrate_grid.py`): framing `cs_grid`;
the colleague's two delivered tables are runs with a `delivered` record block, never resolved.
Runs read as kernels (2026-11-14, `DECISIONS.md` §16): `python kernel.py <topic> <run> ...` / `kernel.ipynb`,
`panel/analysis/kernel/` (one module per analysis over a `KernelSet`), math in `panel/analysis/stats/`;
reproduces `CoRe/llm_pushforward` on the legacy GPT runs (`tests/fixtures/pushforward.json`).

**The run was ported on 2026-11-11 (`DECISIONS.md` §8) and §3.4, §3.5, §3.8, §4 and §6 below are
stale where they disagree with this:**
- `python run.py [-d data] [-m model] [--field value ...] [--show | --one | --check] [--yes]` sends a
  run: the `base` run of `config/presets.json` (`panel/run/presets.py`), then a data preset,
  a model preset and single fields laid over it, later replacing earlier. The base is complete
  and its model is Fake; a preset states only fields of its own kind; `sample` is an option, in
  none of them. `run.py` alone lists them. `simulate.ipynb` is superseded and does not run.
- a framing is `config/topics/<topic>/framings/<name>.json` (2026-11-14, `DECISIONS.md` §18); the topic
  root holds only the one-per-topic files (`topic.json`, `plot.json`, `polls.json`, `positions.json`).
- named sets of runs are `config/topics/<topic>/comparisons.json` (`set -> {label -> run}`, `DECISIONS.md`
  §19); `--set` replaces the runs, the topic is always named: `python compare.py election_2021 --set grids`.
  Runs named one by one need `--name` (2026-11-16): a comparison is never called after its topic.
- a short of several words is hyphenated (2026-11-16, `DECISIONS.md` §33): `grid-simple-en`, `cs-legacy`, `direct-legacy`;
  a run whose record has a `migrated` or `delivered` block is never saved over (`ResultStore.refuse`), with or without `overwrite`.
- `analysis/resolution.py` (2026-11-17, `DECISIONS.md` §34): how finely a model tells subjects apart - the values each column holds (stems) and
  how many different answers there are against how many prompts; a run that repeats is counted within each repeat. On the answers as saved,
  never normalised. `spread` has no violins; the check prints `distinct answers`; `SavedRun.prompts` holds the saved prompts. Not in `compare.py`.
- prepare refuses to replace a table a saved run was built from when it comes out different
  (`DECISIONS.md` §20); `python prepare.py <dataset> --overwrite` forces it. Never pass it unasked.
- `datasets/grid_simple_en/prepared/prompts/` holds a note, not a table: the colleague's prompts are unknown (§21).
- a topic's reference results are `config/topics/<topic>/reference.csv`, the `reference.path` of `topic.json`
  relative to the topic's folder (`DECISIONS.md` §22).
- results are a folder per run and per named comparison (`DECISIONS.md` §24):
  `results/<topic>/runs/<stem>/` holds `results.csv`, `results.json`, `normalized.csv`, `run.json`, `archive/`,
  `analysis/`, `plots/` (and `delivered/`); `results/<topic>/comparisons/<name>/` holds `runs.json`
  (label -> stem), `analysis/`, `plots/`. `ResultStore.run_dir` / `comparison_dir` / `paths` are the only
  places these are built; `Output(folder, save, suffix=)` writes into one such folder, and
  `comparison_output(store, name, runs)` makes a comparison's and records its runs. §3.8 below is stale.
- the place-level tables of 2025 are runs (`DECISIONS.md` §25): `regions_citizen_cs-legacy_legacy_OpenAI_<model>_..._x10`
  and `regions_direct-legacy_cs-legacy_OpenAI_<model>_..._x10`, one row of means per area with a `delivered` block;
  `direct_legacy` is a stored description of `regions`. A `delivered` block now means "kept only as a table",
  ours or someone else's. `results/election_2021/` holds nine legacy runs, eight of these, two Haiku `regions`
  runs and four grid runs.
- a run is packed by `python pack.py <topic> <run> [--zip] [--copy]` (`panel/pack.py`, `DECISIONS.md` §26): `packs/<topic>/<stem>/`
  with `run/`, `dataset/`, `topic/` and `MANIFEST.json` (sha256 per file); refused when values, prompts or forms are not
  the files the record hashes; never written over. `packs/` is gitignored; a `sod` pack holds survey values.
- a run is six names: the **question** is `config/topics/<topic>/questions/<name>.json`
  (`panel/topic/question.py`), possibly worded per subject via a dataset's `shared`
  vocabulary (`panel/dataset.py`: `SharedRef`, `shared_keys`). A framing has no question.
- response classes are `Probabilities` / `Pick`, shapes `Prob` / `Pick`, the list is `options`
  (a saved `parties` is no longer read, 2026-11-15). **Do not rename these, `SHORT_NAME`, or the fields** `voted_or_not`,
  `voted`, `not_voted`, `options`, `name`, `probability`, `choice`. The legacy runs were migrated
  to today's shape on 2026-11-12 (`DECISIONS.md` §10): stems `sod_legacy_cs-legacy_legacy_...` in `results/election_2021/` (the legacy topic was folded into `election_2021` on 2026-11-13, `DECISIONS.md` §13: framing `cs_legacy`, question `legacy`), description
  `legacy` is a stored description (prompts as sent, never rendered), each record carries the schema it
  was sent. The legacy tests pass.
- effort (2026-11-17, `DECISIONS.md` §35) is OpenRouter's vocabulary, `minimal` .. `max`, taken by the Anthropic clients and `OpenRouter`;
  `panel/run/effort.py:effort_tier` asks the provider which levels the model takes (`client.efforts(model)`) and refuses another before the
  first call; the record's `effort` block is `{level, rank, of, levels}`. `resolve` and `--show` stay offline. **Not verified live.**
- a call fails in one of three ways (2026-11-17, `DECISIONS.md` §36), `panel/llm/errors.py`: `RateLimited` (wait, retry), `CallFailed` (retry),
  `BatchFailed` (stop the batch); every client turns its provider's errors into these and never returns None; anything else a call raises
  stops the batch. `run_llm_batch` has no default argument - `execute` owns `max_workers`, `retries`, `rerun_rounds`. **Not verified live.**
- an answer is judged before it is accepted (2026-11-17, `DECISIONS.md` §37), `panel/run/accept.py:judge`: sums within the spec's
  `sum_low` .. `sum_high` (base preset 0.95 / 1.05, None = open, not in the stem), no option twice; a flawed call is tried again, the best
  usable one is kept and named in the record's `faulty`, an unusable one never. A run with a call missing, or stopped, is saved as
  `<stem>_incomplete` (`incomplete`, `missing` in its record), which `load_run` refuses. `run_llm_batch` returns (results, flawed, skipped).
  `run.py ... --missing` (§38) sends only what the incomplete run lacks and saves the whole under the stem, with a `resumed` block.
- the record says what sending took (2026-11-17, `DECISIONS.md` §39): `sending` (calls sent, answered, refused per flaw, failed, rate limited,
  rerun rounds, the settings) and `attempts` per call; `run_llm_batch` returns a `Batch`; `check` prints `calls sent`.
- an answer can be read off the token distribution (2026-11-17, `DECISIONS.md` §40): a framing states `schema` or `answer` (a menu of
  one-token answers, `cs_letters`), exactly one; `OpenAITokens` (`panel/llm/client_openai_tokens.py`, `READS_TOKENS`) sends no schema and reads
  `top_logprobs`; not having voted is a menu entry; `mass` is the probability on the menu, saved with each answer and counted under no entry,
  `voted` / `not_voted` / the options are shares of it and sum to 1 (`TokenProbabilityResponse.from_tokens`); such an answer is not judged
  and never sent again for its content. `resolve` refuses a menu without such a client and the reverse; shape `Prob` only; non-reasoning
  models only. `python run.py -d sod-letters -m nano --client OpenAITokens`; offline `-d sod-letters --client FakeTokens`. No model preset of its own. **Not verified live.** Ideas not acted on are in `TODO.md`.
- a dataset whose subjects are wholes is never pooled (2026-11-17, `DECISIONS.md` §41): `regions` declares `"aggregate": true`, and
  `country`, `distance`, `correlation`, `compass`, `spread` raise `NotApplicable` for it (`analysis/saved.py:pools`), with them `shares`,
  `distances`, `compass` of `compare.py`. `area` is what such a run is read by. Aggregating on purpose is a goal in `TODO.md`, not designed.
- a text edited under its name makes another run (2026-11-17, `DECISIONS.md` §42): `record.changed` sets the record's copied `description`,
  `framing`, `question` against today's files; `ResultStore.refuse` and `--missing` refuse on a difference, `check` prints a `framing` line.
  `run.py --overwrite` saves over a run of the same stem; specs are compared with missing fields at their defaults (`store._same_spec`).
- `cs_area` (2026-11-17, `DECISIONS.md` §43) frames a place instead of a person - shares of voters and votes, built on `cs` sentence for
  sentence; the `direct` preset names it, `citizen` keeps `cs`. `grid_sod` words income and education as `sod` does.
- `cs` was reworded in place on 2026-11-17 (`DECISIONS.md` §44) - the three trial runs that recorded it read `NOT as recorded`;
  `cs_individual` is `cs` plus the rule to estimate this respondent, not the overall result; `election_2025` has both.
- a record's digests are of the file's text, every line ending as `\n` (2026-11-17, `DECISIONS.md` §45): `panel/digest.py:content_sha256`,
  so a checkout that rewrites line endings orphans no run; `file_sha256` is the plain one, for a pack's manifest. 76 digests in the saved
  records were rewritten, each record with a `restamped` block.
- `election_2025` is set up (2026-11-17, `DECISIONS.md` §46): `reference.csv` from the election office's results file, `plot.json`, `polls.json`
  (numbers via Wikipedia's table), `positions.json` from CHES 2024; no dataset has claims for it, nothing has been run.
- `senate_2026` is a topic (2026-11-18, `DECISIONS.md` §47): the first round of the 2026 Senate election, an option is the party that
  nominated the candidate (six above 4% and `Other`, 27%); `cs` adds the rule that the respondent's district voted and every offered party
  stood there; `reference.csv` is the 27 districts, a region's row the part of it that voted (a party is 0 where it nominated no one).
  `media_fees_2026` was polished the same day: `cs` and the question reworded in place (no run recorded them), `plot.json`, `polls.json` with
  one poll that no figure draws. Nothing run on either; `python run.py -d sod --topic senate_2026`.
- `RunSpec` has `question` and `effort`; the stem is
  `<dataset>_<description>_<framing>[_<question>]_<Client>_<model>_<Shape>_t=<t>[_e=<effort>][_x<n>][_subset=<n>]`.
- `resolve` reads `datasets/<dataset>/prepared/prompts/<description>.csv` and checks it against `values.csv`;
  the record carries `values_sha256`, `prompts_sha256`, `forms_sha256` and a `question` block.
- `panel/run/check.py` looks over a saved run; `panel/topic/normalize.py` holds `normalize_groups`
  (`panel.run` does not import `panel.analysis`).
- system prompts say only what is fixed for every run of a topic; the schema text carries the
  form of the answer. Effort is set by the spec, never in a client.
- the generated schema has no description beside a `$ref` (2026-11-15, `DECISIONS.md` §29): the framing's
  `turnout.field` follows `turnout.description` in the turnout model's description. The record carries
  `response_schema_sent` (`client.schema_sent`), the schema as the client's request carries it;
  `tests/test_llm.py` holds both Anthropic clients to the request as it leaves the SDK.
- an analysis module is `compute(run, ...) -> {name: table}`, `draw(run, tables, ...) -> {name: figure}` and the
  public `<name>(run, out, ...)` that saves both (2026-11-15, `DECISIONS.md` §31); a comparison calls `compute`.
  `analysis/estimate.py` names `<x>_low` / `<x>_high`; `metrics.py` and `bootstrap.py` are in `analysis/stats/`;
  `analysis/levels.py:banded` cuts bands; `panel/config.py` and the root `cli.py` hold what loaders and scripts share.
- `analysis/kernel/` is `matrix` (the kernel), `loaded`, `factors`, `replicates` and one module per analysis; `load_ground`
  takes `{label: run}`; `check(store, stem, topic, framing)` reads saved files only; `ResultStore.save` writes archive and
  plain files in one call; `comparison/`, `benchmark/` and `kernel/` analyses stay one function each (2026-11-15, `DECISIONS.md` §32).
- live so far: Haiku 4.5 (`AnthropicLegacy`) and Haiku 5.5 (`Anthropic`, effort low), 200 subjects each - trials,
  out of `results/` since 2026-11-13 (`DECISIONS.md` §14). `results/election_2021/` holds nine legacy runs and two `regions` runs.

---

## 1. What the project does

Uses LLMs to predict how subjects answer a question - which party they voted for, with what
turnout - and compares the predictions against real results.

- three reference points, used throughout
  - **predicted** - what the model output.
  - **claimed** - what the respondent said in the survey.
  - **actual** - the official result.
- **a run is five names and a few numbers**, each name a file:
  - a **dataset** (where subjects come from) emits a *profile table* in a shared schema;
  - a **description** template renders each profile row into a self-description, once, into a
    stored *prompt table*;
  - a **topic** owns the options (ids, the result columns) and the reference results;
  - a **framing** words the topic in one language: system prompt, question, labels, schema text;
  - a **model**: client, model id, temperature, repeats.
  - `RunSpec` holds the names and numbers; `resolve` loads and cross-checks them; `execute` sends
    the calls and saves; the record beside the results copies every text the run was built from.
- **language** is a property of the dataset's values, the description and the framing; a run
  must agree on all three. Two packs exist, `cs` and `en`.

---

## 2. Structure

```
Panel_LLM/
├── panel/                          the library; one importable package, no import-time IO
│   ├── paths.py                      ROOT anchored on __file__; every data path derives here
│   ├── schema.py                     the shared profile schema: concepts, columns, common levels
│   ├── dataset.py                    Dataset declaration, claims maps, profile read/write + checks
│   ├── llm/                          provider clients, schema fix-ups, parallel batch runner (§3.1)
│   ├── describe/                     templates, language packs, rendering, prompt tables (§3.3)
│   │   ├── template.py  render.py  tables.py
│   │   └── lang/cs/  lang/en/         grammar.py + filters.py + forms/regions.json
│   ├── topic/                        vocabulary.py (topic.json), framing.py (<lang>.json), response.py
│   ├── run/                          spec, resolve, execute, record, store, fake, registry (§3.5)
│   └── analysis/                     aggregate, groups, compare, validate; stats/; plots/ (§3.6)
│
├── datasets/                       one slice per dataset: its code and its own data
│   ├── schema.json                   the profile schema (data)
│   ├── sod/       dataset.json  clean.py  prepare.py  claims/  SoD-CZ/ SoD-EN/ (raw, not in git)
│   ├── sod_en/    dataset.json  clean.py  prepare.py  claims/  regions.json
│   ├── grid/      dataset.json  generate.py  grid_income3.json  (grid_config_simple.json: not a dataset)
│   └── regions/   dataset.json  prepare.py  regions.csv
│
├── config/
│   ├── descriptions/   self_v1.json  self_v2.json  self_v1_en.json  (the region descriptions `citizen` / `direct` are in `datasets/regions/descriptions/`)
│   └── topics/<topic>/ topic.json  framings/<name>.json  questions/<name>.json  [plot, polls, positions, comparisons]
│
├── (no data/: prepared tables are datasets/<name>/prepared/values.csv and prompts/<description>.csv)
├── results/<topic>/ runs/<stem>/   comparisons/<name>/   (gitignored)
├── graphs/          generated figures (gitignored)
├── tests/           pytest suite; fixtures/ holds the pre-overhaul baselines (see its README)
├── simulate.ipynb  analyze.ipynb  panel_stats.ipynb      thin; at the root; no def in any of them
├── DECISIONS.md     the decision record
└── update_script.ipynb (gitignored, spent migrations)   _*.md (untracked helper docs)
```

Two root packages, `panel` and `datasets`; notebooks stay at the root so the kernel's cwd is the
import root. Data paths anchor on `paths.py`, so cwd matters for imports only.

---

## 3. Core components

### 3.1 `panel/llm/` - the LLM layer

Self-contained by design: no imports from the rest of the project, no import-time side effects,
so it can be copied to another project as a unit. **Keep that contract.**

| module | responsibility |
|---|---|
| `base.py` | `BaseLLMClient` interface, provider ids, `detect_provider`; `last_model` attribute |
| `client_openai.py` | OpenAI responses API + `text_format`, `instructions=` |
| `client_anthropic.py` | Anthropic `messages.parse` + `output_format` (sent as `output_config.format`), `system=`, `tools=[]` - a response that used a server tool fails (`DECISIONS.md` §17); no temperature. `AnthropicLegacyClient` pins API version `2023-06-01` and sends one, for Opus 4.6 / Sonnet 4.6 / Haiku 4.5 / Opus 4.5 |
| `client_gemini.py` | Gemini `response_json_schema`, `system_instruction` |
| `client_openrouter.py` | raw HTTP, routes `provider/model`, prepends a system message |
| `schema_compat.py` | per-provider JSON-schema fix-ups for strict mode |
| `factory.py` | `create_client(model)` - native client by model name, else OpenRouter |
| `runner.py` | `run_llm_batch` - thread pool, retries, tqdm; returns results + skipped |

- `call_llm(model, prompt, response_format, temperature, instructions=None)` is the one
  interface. After a call, `client.last_model` holds the model id the provider reported (a
  guarded lookup; `None` when a provider reports nothing) - **not verified live**.
- repeats are not the runner's business: `resolve` lays out one row per call.

### 3.2 `panel/paths.py`, `panel/schema.py`, `panel/dataset.py`

- `paths.py`: `ROOT`, `CONFIG_DIR`, `TOPICS_DIR`, `DESCRIPTIONS_DIR`, `DATASETS_DIR`, `DATA_DIR`,
  `PROFILES_DIR`, `PROMPTS_DIR`, `RESULTS_DIR`, `GRAPHS_DIR` - all `Path`s anchored on the
  package. Run records store no filesystem paths, so these never reach saved results.
- `schema.py`: `load_schema()` reads `datasets/schema.json` - concepts, their representation
  columns, and per concept an optional common scale (`levels`; per column `bins` or `map`).
  Raises on a column declared twice or a mapping that does not fit its levels.
- `dataset.py`: `load_dataset(name)` reads `datasets/<name>/dataset.json` (name, short, title,
  language, source, required, descriptions, claims); `load_claims(dataset, topic)` reads a
  claims map (`options`, `abstained`, `excluded`); `write_profile` / `read_profile` validate the
  table both ways - index named `id` and unique, required columns present and non-null, every
  column in the schema or a declared claim column.

### 3.3 `panel/describe/` - descriptions as data

- `template.py`: `PromptTemplate` (name, short, language, sentences, standalone),
  `load_template(path)`, `load_description(name)`, `render_frame`, `RowFilter`.
  - a template is a list of sentences, each a list of alternatives in priority order; the first
    whose slots are all filled wins. `{column|filter}` passes a value through a named filter.
  - a `standalone` description is the whole prompt; the run appends no question.
- `lang/<code>/`: `grammar.py` (primitives: age clause, negation, yes/no, non-answer markers),
  `filters.py` (`FILTERS` the templates may name, `usable`), `forms/regions.json` (the whole
  prepositional phrase per area, keyed by the canonical Czech names). `load_language(code)`.
  - **why forms are tables**: Czech vocalises `v` -> `ve` before st-/zl- clusters; a closed set of
    fifteen is cheaper to tabulate than to derive, and an unknown area raises.
  - the Czech `vaccinated` filter is a `RowFilter` - it reads `gender` for agreement. The survey's
    two-gender shorthand (`spokojený/á`) is resolved at cleaning, so `satisfied` needs no row.
- `render.py`: `render(dataset, template, profile)` checks the languages agree, merges the pack's
  filters with dataset extras, and refuses a row that renders empty; `render_all(dataset,
  profile)` renders every description the dataset declares.
- `tables.py`: `write_table` / `read_table` at `data/prompts/<dataset>/<description>.csv`;
  `read_table(..., expected_index)` refuses a table that does not cover exactly the rows about to
  be run.

### 3.4 `panel/topic/` - vocabulary, framing, response models

- `vocabulary.py`: `Topic` from `topic.json` - `options` (ids, file order), `names`, `aliases`,
  `turnout` (yes, no columns) or `None`, `reference` (`path`, `key`, `total`) or `None`;
  `columns`, `normalize_groups`, `rename(frame)`, `load_reference()`.
  - **option order is the order in the file**, which becomes the `enum` order the provider sees.
    Sorting on load was rejected.
  - `rename` turns native names and aliases into ids, which is how legacy CSVs, the reference
    CSV and the colleague's `p_*` tables read as ids without a file changing.
- `framing.py`: `Framing` from `framings/<name>.json` in the topic folder - language, short, title,
  instructions, question, complete `labels` (defaults to the native names), `schema` text
  (turnout / option / probabilities / pick blocks, class titles included). Raises on a label for
  an unknown id, non-unique labels, or schema text disagreeing with the topic on turnout.
- `response.py`: `make_response_models(topic, framing)` -> `(VotingProbabilities, VotingPick)`
  via `pydantic.create_model`; the enum is the framing's labels in topic order; `to_series()`
  writes ids in the topic's column order. **Do not rename the classes, `SHORT_NAME`, or the
  fields** `voted_or_not`, `voted`, `not_voted`, `parties`, `name`, `probability`, `choice`.
  - the legacy framing reproduces `results/election_2021_legacy/config.json` byte for byte.

### 3.5 `panel/run/` - what a run is

- `spec.py`: `RunSpec(dataset, description, topic, framing, client, model, shape, temperature,
  repeats, sample, random_state)`; `variant(**changes)`, `to_dict()`, `from_dict()`.
  - `temperature=None` sends none. **Claude models need it**: the Anthropic API takes no
    temperature, and `AnthropicClient` raises on a set one rather than dropping it.
- `resolve.py`: `resolve(spec) -> Resolved` - loads the artifacts, checks that the dataset
  declares the description, that the three languages agree, that client and shape are known and
  the model id fits a filename; joins profile and prompt table, samples; `stem`, `question`,
  `prompt(row)`, `calls` (one row per subject and repeat).
- `execute.py`: `execute(resolved, store, client, ...)` - batch, capped rerun rounds over the
  skipped calls, a `RuntimeError` if any remain, then the double save.
- `record.py`: `build_record` - stem, n, saved_at, spec, dataset (rows, profile digest),
  description text, framing text and schema text, `response_schema`, `model_resolved`,
  `git_commit`, `git_dirty`, SDK versions.
- `store.py`: `ResultStore(results_dir, topic)` owns every results path: `paths(stem)`,
  `runs()`, `record()`, `frame(stem, normalized)`, `raw()`, `load(stem, response_model)`,
  `save(resolved, results, client, timestamped, overwrite)`, `analysis(name)`, `path(name)`.
  - refuses a stem that already holds a run of a different spec unless `overwrite=True`.
  - reads legacy records and both raw-JSON shapes; nothing under `election_2021_legacy/` is
    rewritten.
- `fake.py`: `FakeClient` - deterministic offline answers; `fail_calls` exercises retries and
  rerun rounds. `registry.py`: `client_class(name)`, `short_name(cls)`.

### 3.6 `panel/analysis/`

- `aggregate.py`: `aggregate_results_from_df(frame, topic, weighted, normalized)` (predicted),
  `claimed_distribution(frame, claims, topic)` (claimed), `normalize_groups`.
- `groups.py`: `by(frame, key, aggregate, total)` is the one group-by; `predicted_by`,
  `claimed_by`, `spread_by`; `to_levels(profile, concept, schema)` maps onto a concept's common
  scale or raises.
- `compare.py`: `distances(p, q)` (MAE, RMSE, correlation, KL, JS, Wasserstein),
  `distance_table`, `smooth`, `pairwise` (the regional MAE table).
- `validate.py`: output consistency of parsed responses (sums to 1, duplicates).
- `stats/`: `compositional` (Helmert basis, ILR), `manova`, `permanova` - the colleague's code
  from the former `panel_analysis.ipynb`, per results frame.
- `plots/`: `comparison`, `matrices`, `distributions`, `distances` (plotly), `style`. Every
  function returns a `Figure`, shows nothing, saves nothing, changes no global style, and takes
  labels and titles as arguments.

### 3.7 `datasets/` - the slices

- `sod/clean.py` maps the raw `.sav` columns (`AGE1`, `KRAJ`, `Q21`, ...) to the schema;
  `prepare.py` writes the profile and renders the declared descriptions in the same pass.
  Employment and income are the two answers a respondent may lack; everything else is required.
- `sod_en/`: the English edition - same rows and ids, translated labels, two quirks of its own
  handled in cleaning; regions come back as the canonical Czech names via `regions.json`.
- `grid/generate.py` builds the full factorial of `grid_income3.json` (levels in the survey's
  vocabulary); `grid_config_simple.json` is the colleague's English-id design, not a dataset.
- `regions/`: fifteen areas; `regions.csv` also carries approximate populations, used only to
  order figures.

### 3.8 Results on disk

```
results/<topic>/
├── <stem>.csv                one row per call: id, [repeat], turnout, option ids, Prompt
├── <stem>.json               {"<id>" or "<id>:<repeat>": {"prompt", "result"}}
├── <stem>_normalized.csv     column groups rescaled to sum to 1 per row
├── <stem>_run.json           the record (§3.5)
├── <name>_claimed_votes.csv  topic-level reference series (legacy)
├── analysis/                 <statistic>_<stem>.csv - regenerable
└── archive/                  <stem>_<timestamp>.csv + _run.json, one set per save
```

Stem: `<dataset>_<description>_<framing>_<Client>_<model>_<Shape>_t=<t>[_x<repeats>][_subset=<n>]`;
`t=default` when the spec's temperature is `None`.
Legacy stems have no description or framing part; their headers are native names.

---

## 4. Workflow

1. **Prepare** - `python -m datasets.<name>.prepare` (grid: `generate`), from the repo root.
   Writes `data/profiles/<name>.csv`, then every description the dataset declares into
   `data/prompts/<name>/`.
2. **Simulate** - `simulate.ipynb`: the run-setup cell is the only place a run is declared;
   `SPECS["fake"]` runs offline. `execute` writes the six files.
3. **Analyze** - `analyze.ipynb`: a topic and a stem in; country and regional comparison,
   covariance, distributions, every run of the topic, region runs by area out.
4. **Statistics** - `panel_stats.ipynb`: MANOVA, PERMANOVA (sampled), distances by area,
   country-level MAE with the polling agencies, spread by area.
5. **Tests** - `python -m pytest`; see `AGENTS.md` §3.

---

## 5. Known problems and unverified parts

- **no live call has been made since the overhaul.** The generated schemas have the shape the
  providers accepted before, only the wording moved to data; `last_model` reads a guarded
  attribute per SDK. The first real run is the smoke test.
- **the legacy runs are migrated** as dataset `sod`, description `legacy`. **`sod_legacy` is a dataset of its own**
  (2026-11-13, `DECISIONS.md` §12): prepared data only (`"source": null`, no build), stored description `default`,
  preset `sod-legacy`; the migrated runs do not point at it yet.
- **(stale: the folder is gone, §13)** `results/election_2021_legacy/analysis/` held duplicates under older names and
  `corr_matrix_*` files that contain covariance (except the one regenerated); all regenerable.
- **four legacy CSVs carry `Unnamed: 0`** - an artefact of an old backfill; nothing reads
  columns by position.
- **English texts are unreviewed translations** (system prompt, question, schema text, forms).
- **`media_fees_2026`** text is ours, not the colleague's, and its labels changed to Czech in the
  overhaul; no run of ours, and no reference to hold one against.
- **income has no common levels**; `to_levels(..., "income", ...)` raises until the cut points
  are decided.

---

## 6. Conventions

- **Everything the model sees is data**, and is recorded with the run: template text, framing
  text, response schema. Changing a prompt-facing text means a new variant file, never an edit
  of one a run has recorded.
- **Option ids are ASCII and are the result columns**; native names and aliases map onto them on
  load. Core code never hardcodes a party, region or label.
- **Config at construction time.** No module-level clients, file reads or env reads. Notebook
  first cells and the `prepare` scripts are the composition roots.
- **Fail loudly.** Visible errors over silent fallbacks; a recovery path stays written but
  disabled and marked `NOTE:`.
- **One responsibility per module**, split by change-reason; vertical slices as folders.
- **Never rename the response classes, `SHORT_NAME`, or the model fields** (§3.4). Stems, records
  and saved JSON are keyed by them.
- **Plot functions return a `Figure`**; no `show`, no saving, no global style, no domain names.
- **Notebooks define nothing** and restart-and-run-all offline.
- **Type hints** on public functions; a type goes in the signature, never in prose.
- **A docstring or comment says only what the code cannot** (2026-11-16; the former `Args:` / `Returns:` /
  `Raises:` rule was a mistake):
  - what earns a line: a why, a precondition or invariant, a shape or unit, the order of a returned tuple,
    the keys or columns of a returned table, an error raised deeper down than the function's own body.
  - what does not: a line that restates a name, the signature or the body; a `Raises:` for a `raise` in the
    body; a copy of a data file; the `Args:` / `Returns:` / `Raises:` skeleton for its own sake.
  - a function whose signature and body say it all has no docstring; a docstring is never longer than
    the body it describes.
  - a module docstring states the responsibility and the reasoning behind the method, once; the
    functions do not repeat it.
- **Czech and English texts are data**; identifiers, comments and docs are English.
- **Terse code, no explanatory comments, no status printouts.**

---

## 7. Open / next

- confirm the topic folder shape, the repeats semantics, the `media_fees_2026` labels and the
  income cut points (`DECISIONS.md` §6).
- the live smoke call per provider - with the first real run.
- an actual-results CSV for `election_2025`, then a `reference` block in its `topic.json`.
- the grid-comparison layer kept standalone in `CoRe/_grid_separate` overlaps `schema.json`'s
  levels and `to_levels`; one of the two should go.
- the `p_*_std` / `n_valid_repeats` aggregation of repeated runs is not written.
- replay onto this branch anything landed on `dev` after 2026-09-09.
