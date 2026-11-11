# Panel_LLM — decision record

Decisions that shape the code, with their reasons, so they are not re-argued; rejected
alternatives stay listed for the same reason. Dated entries are Martin's. The undated ones were
made while implementing the 2026-09-09 architecture overhaul on branch `refactor_complete` and
are open to his review. `CLAUDE.md` describes the tree this produced; this file says why.

---

## 1. The axes of a run

- **A run is five names and a few numbers** (2026-09-09): dataset, description, topic, framing,
  client + model + temperature, plus `sample`, `random_state` and `repeats`. `RunSpec` holds
  nothing else - no frame, no callable, no loaded object.
  - so a spec serialises into the record as it is, and a record turns back into the spec.
  - resolving the names is a separate step (`panel.run.resolve`), and it is where the axes are
    checked against each other before a call is paid for.
- **Option ids are the result columns** (Q1, 2026-09-09): the keys of `topic.json`, pure ASCII,
  with the official label as `name` and foreign column names as `aliases`.
  - the reasoning: a 2021 result header used to be the label the model saw; an English run shows
    a different label, so the column had to become an id and each language maps its label onto it.
  - applies to the legacy topic too: its files stay as they are and `Topic.rename` turns native
    names into ids in memory. The reparse gate compares after that rename.
- **The stem encodes every axis by its short** (Q2, 2026-09-09):
  `<dataset>_<description>_<framing>_<Client>_<model>_<Shape>_t=<t>[_x<repeats>][_subset=<n>]`.
  Shorts are ASCII without underscores, because the underscore separates parts.
- **Templates are shared across datasets, generalised by concept** (Q3, 2026-09-09):
  `datasets/schema.json` names the concepts and their representation columns; a template phrases
  whichever representation it knows, in priority order, and a dataset emits whichever it has.
  - rejected: one column per concept with the dataset declaring its kind - the same header over
    incomparable values hides the difference, and cross-representation comparison needs the
    level map either way.
- **`datasets` keeps its name** (Q4, 2026-09-09); the HuggingFace collision is a README remark.
- **A topic is a folder**: `topic.json` (vocabulary) plus one framing file per language.
  Implemented on the handoff's recommendation; confirmed by Martin 2026-11-10.
  - one multilingual file was rejected: two system prompts nested in one JSON are unreadable,
    editing the Czech prompt touches the English one's file, and what must be shared sits mixed
    with what must not.
  - one self-contained file per language was rejected: comparability inside a results folder
    would rest on discipline, a typo in a short would silently create a new column, and turnout
    roles, reference and aliases would be duplicated. The folder lets the loader raise on a
    framing that misses or invents an id.
- **Language is a property of three artifacts, never a global**: dataset values, description
  sentences, framing text. `render` and `resolve` refuse a mismatch.

## 2. Superseded earlier decisions

Each with the reason that superseded it, so the old reason is not re-argued.

- **Schema field descriptions as data** - rejected 2026-09-08 because it hid what the model saw.
  Language forced it: an English framing needs English descriptions, class titles included. The
  mitigation is stronger than the objection: every run record carries the exact
  `model_json_schema()`, as the legacy folder's `config.json` already did.
- **One describer per dataset, inside its slice** - decided 2026-09-08 when describers were
  code. Templates are data and are written against schema columns; the loud failure on a missing
  answer moved to the dataset's `required` list at cleaning.
- **`utils` not renamed** - a wart kept deliberately. Every import moved anyway, so the rename
  to `panel` was free once and never again.
- **`prepare.ipynb` below the root with a `chdir`** - replaced by a script per dataset
  (`python -m datasets.<name>.prepare`). Deterministic work is a script; a notebook may explore.
- **Prompt tables in one flat folder**, and the grid rendering into its own CSV - unified under
  `data/prompts/<dataset>/<description>.csv`, written by the dataset's prepare step.
- **A stem without the description** - two descriptions of one model would have overwritten each
  other.
- **Templates shared across datasets, generalised by concept** (§1, Q3) - superseded 2026-11-10
  by §7. The shared schema was the in-between level nobody needed yet, and it blocked progress
  repeatedly: age and income come as numbers, bands and differently cut bands, education levels
  are split or bundled per source, and casting to a shared vocabulary loses information.
- **Profile values phrase-ready** (lower-cased, gender resolved at cleaning) - superseded by §7:
  one cell was both the category analysis groups by and the fragment a sentence splices.
- **Language packs with grammar code** (`panel/describe/lang/`) - superseded by §7: every phrase
  is written down per dataset in `forms.json`.

## 3. Still binding from `_Refactor_Plan.md`

- option order is file order: no sorting on load, no `priority` field. It is prompt content.
- no `Topic` object on `RunSpec`; the string key is the record.
- no free-text option names: the `Literal` enum is why results are usable.
- no papermill, no CLI replacing notebooks, no folder per run, no kind-first results folders.
- YAML/JSON run files: not now; a names-only `RunSpec` makes it trivial later.
- the colleague's statistics are extracted and de-duplicated, never re-derived; colleague code
  lands verbatim first, and only what a second consumer needs is extracted.
- age and income banding for the grid, and which grid variant to standardise on, are research
  choices expressed in config; both grid configs were left as they are.
- the two stashes are not part of this work.

## 4. Made during implementation

Open to review; each names its reason.

- **`panel/topic/vocabulary.py`, not `topic/topic.py`** - the user-level rule against stuttering
  paths. `panel.topic` re-exports `Topic`, `load_topic`, `load_framing`, `make_response_models`.
- **Repeats expand the calls before the runner** rather than inside it - the LLM package's
  contract stays untouched. One row per call, a `repeat` column when `repeats > 1`, raw-JSON keys
  `<id>:<repeat>`; the colleague's `p_*_std` and `n_valid_repeats` become an analysis step.
- **Clients record the model id the provider reported** (`last_model`), read with a guarded
  attribute lookup per SDK (`response.model`, `model_version`, the JSON `model` field). Unverified
  live; the record field stays `null` when a provider reports nothing.
- **A `standalone` description is the whole prompt** and gets no question appended
  (`latent_direct_v1`). The old notebook set `question=""` by hand; the flag records the intent.
- **`to_series` writes columns in the topic's order** whatever order the model answered in.
  Legacy CSVs kept the model's order; nothing reads columns by position.
- **`media_fees_2026` shows Czech labels** instead of the colleague's English ids inside a Czech
  prompt. The one deliberate prompt change; the topic has never been run and the snapshot was
  regenerated on purpose. The colleague's ids stay as aliases.
- **Wasserstein distance** uses the position-weighted version the colleague wrote and left
  unused; the notebook had stubbed the metric to a constant 1.
- **PERMANOVA runs on a sample in the notebook** - the full 3880 x 3880 distance matrix and its
  permutations are out of reach; the colleague's own call was commented out at full size.
- **`regions.csv` lost its `level` column** - which row of a reference is the whole is the
  topic's business (`reference.total`) - and **gained `population`**, the notebook's approximate
  numbers, used only to order areas in figures.
- **Git policy for derived data**: prompt tables under `data/prompts/<dataset>/` are tracked (as
  before); profiles of the survey editions are ignored like `clean_data.csv` was; the grid and
  regions profiles are tracked, being synthetic and small.
- **`tests/descriptor_test.ipynb` became `tests/test_describe.py` in phase 3**, not phase 7: its
  subject was what phase 3 changed, and a notebook importing retired modules would have broken
  the commit. `prepare.ipynb` was replaced in phase 2 for the same reason.
- **Non-answers are the language pack's business** (`is_non_substantive` in `grammar.py`); the
  survey's two-gender shorthand is resolved at cleaning, so the `satisfied` filter no longer
  needs the row.
- **Claims maps sort answers into `options`, `abstained`, `excluded`**; the old
  `count_non_substantive_as_not_voted` flag survives as `count_excluded_as_abstained`. The
  legacy topic shares the 2021 map.
- **Common levels**: `age` carries the colleague's four bands (`bins` on the exact age, a `map`
  on the grid's bands); `income` declares none, because the cut points are a research choice
  and `to_levels` should raise rather than guess.
- **`sod_en` is a real dataset**: the English `.sav` is the same 3880 rows and ids with direct
  translations. Region names come back as the canonical Czech names every reference and language
  pack keys on; the English pack's forms table is keyed by them.
- **A dataset declares the descriptions it renders**, and its prepare step renders them in the
  same pass as the profile, so a table cannot lag the rows it describes.
- **The reference is a block** on the topic - `path`, `key`, `total` - so the analysis knows
  which row is the whole without a region list in code.
- **`git_dirty` in the record ignores untracked files**; the three untracked helper docs would
  otherwise mark every run dirty.
- **Fixture keys were renamed, digests unchanged** (`sod/self_v1`), schema snapshots live per
  framing, and the legacy schema fixture *is* the frozen `config.json`, not a generation.
- **Temperature is optional** (2026-11-10): `RunSpec.temperature=None` sends no temperature and
  the stem reads `t=default`; the default stays `0.0`, so earlier specs and stems are unchanged.
  - the reason: API version `2026-10-21` (what `anthropic` 2.x sends) rejects `temperature` on
    every Claude model, and Opus 4.7 and later rejected it before that.
  - the Anthropic client raises on a set temperature instead of dropping it; the other three
    clients send it only when it is set.
  - rejected: dropping it silently in the client for models known to refuse it - a model list
    hidden in code, and a stem and record claiming `t=0.0` for a run it was never applied to.
  - consequence: Claude runs are not deterministic; `repeats` is how their spread is measured,
    and comparing them with `t=0.0` runs compares two sampling settings.
- **The Anthropic client uses `messages.parse`** (2026-11-10), no beta header, `max_tokens`
  16000 because thinking is always on for current models and counts against the limit. A call
  with no parsed output (refusal, truncation) raises with its `stop_reason`. Server-side
  refusal fallbacks are deliberately off: a run answers with the model it names, or fails.
- **`AnthropicLegacy` is a separate client** (2026-11-10): the same call pinned to API version
  `2023-06-01`, the last that accepts a temperature, for the models that still honour one
  (Opus 4.6, Sonnet 4.6, Haiku 4.5, Opus 4.5). The temperature travels in `extra_body`.
  - a separate name so the stem and the record say which API behaviour a run used.
  - rejected: the `Anthropic` client switching version by itself when a temperature is set - a
    different API behind one client name.
  - a stopgap: it lasts as long as Anthropic serves that version. Unverified live.
- **Effort is fixed at `low` in `AnthropicClient`** (2026-11-10), a constructor default marked
  `NOTE:`; `AnthropicLegacy` sends none, because Haiku 4.5 rejects the setting.
  - debt: effort is not a `RunSpec` field yet, so neither the stem nor the record shows it. Add
    it to the spec before runs at different effort levels exist, or they share a stem.

## 5. Rejected during implementation

- rendering the English template over the Czech grid - mixed language; `render` refuses it by
  design. The English edition of the survey is the English dataset instead.
- a `config/models.json` - model ids are typed where a run is declared, and the record stores
  them; nothing iterates models.
- keeping the fake run the notebook check wrote inside `results/election_2021/` - a fake run in a
  real results folder would show up in every comparison. Removed after the check.
- a `panel/dataset/` subpackage - two modules (`dataset.py`, `schema.py`) carry the load.

## 7. Prepare reshaped (2026-11-10)

Decided with Martin in discussion, prototyped in a sandbox, then ported. **The branch is in
transition:** prepare works in this shape; `panel.run`, the analysis notebooks, the datasets
`sod_en`, `grid` and `regions`, and their tests still expect the old one and fail until ported.

- **Two levels, nothing in between**: general code in `panel/`, which knows no column and no
  value, and a dataset's own vocabulary in `datasets/<name>/`. "Swap the topic and the dataset
  and run" needs a contract, not a shared vocabulary.
  - comparing two datasets is a later, separate layer, written per pair.
  - a shared attribute schema is a refactor for later, if ever; do not reintroduce it in passing.
- **Prepare has two outputs**, under one folder per dataset:
  - `data/<dataset>/values.csv` - the dataset's own answers, untouched, for analysis and rendering.
  - `data/<dataset>/prompts/<description>.csv` - one rendered text per subject, for a run.
- **Wording is data**: `datasets/<name>/forms.json` maps each answer to the phrase a sentence
  needs. Several named forms per column (`{region|in}`, `{region|of}`), agreement with another
  column (`by`), non-answers listed once (`none`), several answers in one cell (`split`/`join`).
  An answer that is neither worded nor a non-answer raises.
  - rejected: a "just lower-case it" rule for the simple columns - a second mechanism to save a
    few lines, and it would let an unexpected answer into a prompt.
- **Descriptions belong to a dataset**: `datasets/<name>/descriptions/*.json`, found by listing
  the folder. A template names that dataset's columns, so sharing one only works by coincidence.
- **One representation per attribute**: the prompt and the analysis use the same column. SoD
  keeps the exact `age`; `age_range` is gone. Showing one level and analysing another is a
  later refactor.
- **The declaration states roles**: `columns`, `optional`, `groups` (what analysis may group
  by), `area` (the column naming a row of reference results), `claims`.
- **One entry point**: `python prepare.py [dataset] [-d descriptions] [--prompts-only]`; without
  a dataset it asks. A dataset supplies one function, `datasets/<name>/build.py::build(dataset)`.
- **SoD wording corrected against the survey's question text** - every description differs from
  the legacy prompts, deliberately:
  - income is the respondent's own net monthly income (`INCOMEP`), not the household's.
  - living standard is the household's (Q19): "Životní úroveň naší domácnosti považuji za ...".
  - employment agrees with gender; `OSVČ` keeps its capitals; several statuses join with
    "a zároveň"; education drops the repeated "vzdělání" and writes "+" as "nebo"; the survey's
    typo "Vice než" reads "více než" in the prompt only.
  - `self_v1` / `self_v2` are `default` / `extended`.

## 8. The run on the reshaped prepare (2026-11-11)

Decided with Martin while prototyping in the sandbox, then ported. **Still in transition:**
prepare and the run work in this shape; the analysis layer, the notebooks, the datasets
`sod_en`, `grid` and `regions` and their tests still expect the old one and fail.

- **A run reads the prompt table only.** The values table supplies the subjects that table must
  cover, and the one column a question worded by a shared vocabulary asks for. A prompt table
  rendered before the values were rebuilt is refused.
  - the record carries sha256 of `values.csv`, the prompt table and `forms.json`, not their
    text: the template text is copied as before, and every call's prompt is already in
    `<stem>.json` and the CSV.
  - `panel.run` no longer imports `panel.analysis`: `normalize_groups` lives in
    `panel/topic/normalize.py`.
- **The question is its own name**, the sixth of a run: `config/topics/<topic>/questions/<name>.json`,
  one file per question holding every language. The framing has none.
  - a wording may depend on the subject: `by` names a vocabulary (`gender`), `text` maps its
    keys to wordings ("... jsem volil:" / "... jsem volila:"), `fallback` is the key for a
    dataset that does not share the vocabulary. Without a fallback such a dataset cannot be asked.
  - a question lacking the framing's language is refused; a `standalone` description takes no
    question and any other needs one.
  - rejected: one folder per language; the question as free text on the command line; several
    named questions inside the framing file.
- **A dataset may declare `shared` vocabularies** in `dataset.json`: a column and a map from its
  answers to shared keys (`"Muž": "male"`). This is the one place a dataset meets a name from
  outside itself, and it is opt-in per question - not the shared schema §7 removed.
- **The system prompt says only what is fixed for every run of the topic**: the role, the
  task, turnout. Nothing about the profile's attributes, nothing about the form of the answer;
  the schema text carries that, per shape, in the topic's own words (parties and the election,
  stances for `media_fees_2026`).
  - why: the old prompts listed attributes the `default` description does not contain, and
    asked for probabilities of every party, which does not fit a pick.
  - the framings were edited in place, not as variants: no saved run had recorded any of them.
  - nothing links a system prompt to what else a run names; a framing states what it assumes
    in its `note`.
  - rejected: `instructions` split into a common part and one per shape; a framing per shape;
    whole prompts keyed by shape.
  - seen live (Haiku 4.5 and 5.5, 200 subjects each): with only the schema text asking for it,
    Haiku 5.5 kept every party distribution summing to 1; Haiku 4.5 missed in 7 of 200, by at
    most 0.05.
- **The response classes name no topic**: `Probabilities` / `Pick`, shapes `Prob` / `Pick`, the
  list `options`. This reverses "never rename the classes, `SHORT_NAME` or the fields"; the rule
  stands again from here, for the new names.
  - why: class title and field name are sent in the schema, so a stance topic was asked for
    `parties` by `VotingProbabilities`.
  - `voted_or_not`, `voted`, `not_voted` stay: only a topic with turnout has them, and turnout
    is about voting. Rejected: `attended` / `not_attended`.
  - the legacy runs stay readable: `options` also reads a saved `parties`; their stems keep
    `VProb` / `VPick`, which are file names and no longer shape names.
- **What this breaks for legacy - known, to migrate later:**
  - the legacy framing no longer regenerates the schema the legacy runs were sent (class
    titles, `parties`): `test_legacy_schema_identical` and the legacy schema snapshot fail.
  - legacy records name `VotingProbabilities`; `test_legacy_results_reparse` looks the model
    up by that name and fails. `panel_stats.ipynb` picks runs by `("VotingProbabilities", "VProb")`.
  - raw JSON differs by age: legacy files hold `parties`, new ones `options`.
  - the note in `election_2021_legacy/topic.json` ("reproduces byte for byte") is stale.
  - ways out, not chosen: rewrite the legacy files once and drop the alias; or keep them
    frozen and hold the old names in one legacy reader.
- **Effort is a field of the run**: `low` .. `max`, or None, which sends nothing. It is in the
  recorded spec and in the stem (`..._t=default_e=low`). The client gets it at construction;
  the `low` that was fixed in `AnthropicClient` is gone. A client that takes none refuses it.
- **Named runs are data**: `config/runs.json`. `base` says what is asked, of whom, in what
  form; every run states the model's settings - client, model, temperature, effort, sample -
  and the base may hold none of them. A run may also replace a base field.
  - why: the base can change, and a model's settings must not change with it.
  - rejected: a base holding model settings with runs overriding freely - editing the base
    silently changed every run that did not override; no base, every run in full - one
    experiment spelled out per run.
- **One entry point**: `python run.py [name] [--field value ...] [--show | --one] [--yes]`.
  It replaces `simulate.ipynb`. Anything but the Fake client asks before sending.
- **No fallback model**: a call the provider's safety classifier declines raises; a run
  answers with the model it names or fails.
- open
  - how system prompt and user prompt fit: the system prompt speaks of a respondent in the
    third person, the user prompt is the respondent speaking and ends in an unfinished
    sentence; the question presupposes voting while the answer carries turnout.
  - wording not reviewed: the `neutral` question, rule 3 of the system prompt, the
    `media_fees_2026` text, the English texts, "strana" for coalitions.
  - `Dataset.data_dir` is fixed to `paths.DATA_DIR`; a test cannot redirect the data folder.
  - `run.py` without arguments resolves every named run, reading the tables once per run.
  - no test covers a `standalone` description: no ported dataset has one.

## 6. Pending Martin's decision

- income cut points for `to_levels` (§4) - left open on purpose (2026-11-10); only needed for
  grid-versus-survey comparison.

Confirmed 2026-11-10, as implemented: the topic folder shape (§1), repeats as one row per call
(§4), the Czech `media_fees_2026` labels (§4).
