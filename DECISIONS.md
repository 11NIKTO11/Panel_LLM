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
`sod_en` and `grid` and their tests still expect the old one and fail.

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
- **Presets are data: a base run, and two kinds laid over it on the command line**:
  `config/presets.json`. The base is a complete run. A data preset says what is asked, of
  whom and in what form (dataset, description, topic, framing, question, shape, repeats); a
  model preset says which model answers and how (client, model, temperature, effort). Each
  states only the fields it sets: `run.py -d sod-extended -m haiku-5.5`.
  - priority: base, `-d`, `-m`, single fields; later replaces earlier. A preset may state
    only fields of its own kind, so `-d` and `-m` never state the same field and their
    order decides nothing.
  - the base states every field of both kinds, and its model is the Fake client: a run
    that names no model sends nothing, so one preset alone is allowed.
  - editing the base changes every preset that leans on the edited field. The stem and the
    record come from the spec, so no saved run is confused; `run.py` alone lists every
    preset with all its fields filled in.
  - `sample` and `random_state` are in none of the three: a trial is the same run, only
    smaller. They are options (`--sample 200`).
  - a preset name is a handle; a saved run is identified by its stem and its record.
  - rejected, each built first: a base holding model settings with whole runs overriding
    freely (nothing limited what a run touched, and a run was one line per combination);
    no base, every run in full (one experiment spelled out per run); a base for the data
    with every run stating the model settings (the list grew as model x variant, and a
    trial needed an entry of its own); two kinds with no base, each preset stating every
    field of its kind and both names required (every data preset repeated topic, framing,
    shape; one name alone was refused).
  - rejected, not built: a plain name that merges a whole run while `-d` takes only its
    data fields - one name would mean two things by how it is passed.
  - not built: a third table naming pairs that belong together, for analysis; nothing
    needs it yet.
- **One entry point**: `python run.py [-d data] [-m model] [--field value ...] [--show | --one | --check] [--yes]`.
  It replaces `simulate.ipynb`. Anything but the Fake client asks before sending.
- **No fallback model**: a call the provider's safety classifier declines raises; a run
  answers with the model it names or fails.
- **`regions` is ported** (2026-11-11): `datasets/regions/` has `build.py`, `forms.json` and
  the descriptions `citizen` and `direct`, formerly `latent_citizen_v1` / `latent_direct_v1`.
  Both render word for word what the old tables held.
  - `direct` is `standalone`: a run names no question (`"question": null`).
  - `citizen` takes a question; `regions` shares no gender vocabulary, so a gendered
    question reads its fallback ("... jsem volil:") after "Jsem obyvatel/ka ...". The
    `neutral` question has one form and avoids that.
  - the population column of `regions.csv` stays out of the values, as before.
- open
  - how system prompt and user prompt fit: the system prompt speaks of a respondent in the
    third person, the user prompt is the respondent speaking and ends in an unfinished
    sentence; the question presupposes voting while the answer carries turnout.
  - wording not reviewed: the `neutral` question, rule 3 of the system prompt, the
    `media_fees_2026` text, the English texts, "strana" for coalitions.
  - `Dataset.data_dir` is fixed to `paths.DATA_DIR`; a test cannot redirect the data folder.
  - the system prompt speaks of a respondent, and a `regions` run has none: the subject is
    a place. Part of the fit question above.
  - identical prompts are each sent: a prompt that is word for word another subject's costs
    a call of its own. `prepare.py` and the run check both count them (50 of 3880 in SoD
    `default`, 20 in `extended`). They could be sent once and the answer copied to every
    subject sharing the prompt - not done:
    - the saving is about 1% on SoD; it grows with coarser descriptions and is large for a
      grid or regions.
    - a copied answer hides that a call is a sample. Even at temperature 0 Haiku 4.5 did
      not repeat itself: two runs of the same 200 prompts differed (2026-11-11). With
      repeats, or a client that sends no temperature, copying would understate the spread.
    - the results would have to say which answers are copies.

## 9. The analysis layer on the reshaped data (2026-11-11)

Rewritten on the reshaped prepare and run, prototyped in a sandbox and ported in one go. One
run: `answers`, `country`, `distance`, `area`, `spread`, `correlation`, `groups`, `effects`,
`compass`. Several runs: `shares`, `distances`, `areas`, `effects`, `compass`.

```
analyze.py                      the entry point: lists saved runs, or analyzes one; no logic
analyze.ipynb                   the same, one block per analysis; one setup cell with SAVE, STYLED
compare.py                      the entry point for several runs of one topic; no logic
compare.ipynb                   the same, one block per comparison
config/topics/<topic>/          optional, read by analysis only: plot.json (colours, labels),
                                polls.json (published polls), positions.json (expert positions)
panel/topic/                    plot.py, polls.py, positions.py load them
panel/analysis/
├── saved.py                      SavedRun: a run read back with its subjects, claims, reference
├── output.py                     Output: where tables and figures go; `save` off writes nothing
├── aggregate.py                  predicted_distribution, claimed_distribution
├── bootstrap.py                  interval: percentile bootstrap over subjects
├── sources.py                    predicted / claimed / actual: one frame, the three distributions
├── metrics.py                    distances: MAE, RMSE, correlation, JS, KL (one shared floor)
├── answers.py                    analysis: mean and top per column; a table, no figure
├── country.py                    analysis: predicted, claimed and actual over the whole run
├── distance.py                   analysis: every metric for each pair of sources
├── area.py                       analysis: MAE and RMSE between the sources per area, with subject counts
├── spread.py                     analysis: each column over subjects; distance from the mean answer
├── correlation.py                analysis: correlation and covariance between options, over subjects
├── groups.py                     analysis: predicted, claimed and spread per level of each grouping column
├── effects.py                    analysis: MANOVA and PERMANOVA of the shares on the grouping columns
├── compass.py                    analysis: the model's map of the options against the experts' positions
├── levels.py                     a grouping column as levels, as the dataset declares
├── stats/                        compositional.py (zero replacement, ILR), design.py, manova.py,
│                                 permanova.py, mapping.py (MDS, Procrustes, Mantel)
├── comparison/                   several runs: loaded.py (RunSet, load_runs), shares.py, distances.py,
│                                 areas.py, effects.py, compass.py
└── plots/                        style.py, comparison.py (source_bars), intervals.py (interval_panels),
                                  distributions.py (violins, histograms), matrices.py (heatmap, grid),
                                  stacked.py (stacked_bars), maps.py (maps)
```

- an analysis is a function `(run, out) -> (table, figures)`, listed in `ANALYSES`; the script
  runs them all (`--only` for one), the notebook has one block each
  - `figures` is a list: `answers` has none, `area` one per metric (`area_mae`, `area_rmse`)
  - `subjects` is the run's own subjects, not the whole values table: a run over a sample is
    compared with what that sample claimed
  - grouping columns and the area column come from `dataset.json` (`groups`, `area`)
- a run is named by any text only one stem contains; a text two stems contain raises and lists them
- refused, loudly
  - legacy runs (keyed by row position)
  - a values table rebuilt since the run was sent (sha256 against the record)
  - saved rows that are not the subjects the record lists; an area with no reference row
- tables go to `results/<topic>/analysis/<name>_<stem>.csv`, figures to
  `results/<topic>/plots/<name>_<stem>.png`; `ResultStore.plot` owns that path
  - rejected: a `graphs/` folder beside `results/` - a run's figures belong with its tables
- a topic's `plot.json` is applied by default: bars wear the option's colour and sources
  differ by strength; `--plain` / `STYLED = False` gives the default palette (a colour per
  source). Figures name an option by its id, the topic's short name and the result column;
  `labels` in `plot.json` overrides one, and the long native names are not drawn. Nothing
  a model sees reads this file
  - `last` names options a ranking puts at the end (`Other`); figures that rank the options
    (`country`, `spread`, the shares comparison) go by the actual result, largest first
    (`sources.ranked`), by the prediction on a topic without a reference
  - colours cover all the options or none, and both turnout columns or neither; the turnout
    pair is drawn green and red in `country`
  - `framing_names` skips `plot.json`, `polls.json` and `positions.json`, the one change
    this made to the run layer
- claimed is counted from scratch: the run's own subjects' answers in the claims column,
  sorted by the dataset's claims map; nothing reads `<name>_claimed_votes.csv`
- intervals are bootstrap over subjects (a subject's repeats are drawn together); they say how
  much the sample moves the number, not how much the model would differ on another call
- figures are matplotlib `Figure` objects made without pyplot, in one palette; predicted,
  claimed and actual keep their colour in every figure
- one resampling of subjects serves predicted and claimed together, in `country` and in
  `distance`: both are about the same subjects, so a distance between them is resampled as a pair
- an analysis raises `NotApplicable` for a run it has nothing to work on (no claims and no
  reference: no distance); the script prints that and goes on
- changed from the colleague's `calculate_distances`, to compare at the end
  - Wasserstein is gone: it read the options as ordered positions, and reordering them
    changed the value
  - KL adds one floor, 1e-6, to both distributions for every source; before, `distances`
    clipped at 1e-12 and the notebook smoothed only the claimed series with 1e-6. KL still
    moves a lot with an empty option - JS does not need a floor and is the safer read
  - the turnout difference (source minus reference) is reported beside the option metrics
- the check of what a run saved (`panel/run/check.py`, also `run.py --check`) is printed by
  `analyze.py` after the run's description and has its own notebook block; it is not an
  analysis: it writes nothing and compares with nothing
  - it is two parts: `check` / `report` say whether the run is complete and consistent;
    `means` is what the answers say (mean and top per column), which the `answers` analysis
    writes as a table and `run.py` prints
  - the report counts answers with P(voted) = 0, answers whose options sum to 0 (no
    preference; they are NaN once normalized) and their overlap
  - `top` is taken over answers that have a preference; before, an all-zero answer counted
    for the first option
- `area` resamples subjects within each area, 300 draws by default: fifteen areas at 1000
  each made the pass slow
  - its three panels share one scale; `interval_panels(same_scale=True)`. `distance` keeps a
    scale per panel, its metrics having different units, three panels to a row
- `spread` and `correlation` read one row per subject (`aggregate.subject_answers`): repeats
  are averaged first, so they show the spread between subjects, not the model's noise on one
  - on the normalized answers; a subject with no preference is left out and counted in the
    figure's note. Before, such rows made the cosine histogram all-NaN
  - heatmaps leave the diagonal blank: in a covariance matrix it took the whole colour range
  - the correlation figure says what no relation looks like, -1/(k-1): shares sum to 1, so
    the options are negatively correlated by construction
  - no seaborn: violins and histograms are plain matplotlib, without a density curve
- several runs are compared by their own entry points, `compare.py` and `compare.ipynb`, on
  `panel/analysis/comparison/`: the input is a topic and a list of runs, and the output has
  no single stem, so files are `runs_<comparison>_<name>`, the name being the topic unless given
  - rejected: a list of runs to `analyze.py` - "analyze each" and "compare these" would
    share one argument
  - runs are named one by one; "every run of the topic" would sweep in trial and Fake runs
  - a comparison builds on the single-run analyses (`country`, `distance`), so a run's
    numbers and intervals are the same in both places
  - a run's label is its model, plus the spec fields that differ where two runs share a model
  - claimed is one source when every run asked the same subjects, else one per set of subjects
  - polls are topic data, `polls.json` with option ids, copied from `data/agentury.csv`; they enter the tables and the distance figure, not the bar figure
  - `panel/analysis/compare.py` became `metrics.py`: `compare` now names the entry point
  - in the distance figures the polls come first, then the claimed, then the runs; polls are
    also held against the claimed distribution when every run asked the same subjects
  - heatmap colours go by meaning, named in `plots/style.py`: errors run yellow to red,
    effect sizes in cividis as the legacy figure did; the number's ink follows the cell
  - by area and the effects are heatmaps with the number in every cell and the interval or
    p-value underneath, columns by runs; a dot per run in every row was tried and read worse
  - in the shares figure the actual result is near-black and the claimed grey
    (`style.REFERENCE`, `style.SECONDARY`), so the runs alone carry colour: at most eight,
    more raises. The single-run `country` figure does the same when drawn `--plain`
- a dataset declares how a grouping column reads as levels, `levels` in `dataset.json`
  (`panel.dataset.Levels`, `panel/analysis/levels.py`): `order` along a scale, non-answers
  last; `bands` of a numeric column; `several` for a cell holding several answers. A column
  with no entry (gender, region) goes largest group first
  - age bands are the survey's own `AGE2` bands, declared, not a column of `values.csv`:
    rebuilding the values table changes its sha256 and every saved run then refuses to load
  - an employment cell holds several statuses; a subject counts at each, so the figure has
    the ten statuses and their counts add up to more than the subjects. In `effects` the
    column is an indicator per status (`stats/design.py`), not a level per combination
  - the separator is now stated three times (`clean.SEVERAL`, `forms.json` `split`,
    `levels.several`); open
- the old "spread of options by region" is the `groups_spread` table, for every grouping
  column, without a heatmap
- `effects`, changed from the colleague's MANOVA and PERMANOVA, to compare at the end
  - zeros are replaced multiplicatively by 0.005 before the log-ratio transform, not clipped
    at 1e-12: a clipped zero sat some 25 log-units away and decided the test
  - MANOVA reports partial eta squared (Pillai over its largest possible value) with F and
    p, so predictors with different numbers of levels compare; raw Wilks and Pillai did not
  - PERMANOVA tests each column alone and reports R squared; before, all columns went in as
    one design and gave a single F
  - age is numeric in both; before, PERMANOVA dummy-encoded it into about 46 columns
  - the `correlation` distance is between two subjects' rows; before, it centred across
    subjects. `mae` is gone (it is `l1` over the number of parts); `aitchison` is the old
    `euclidean_ilr`
  - a predictor that leaves no residual raises; before, a NaN F gave the smallest p
  - PERMANOVA samples 1000 subjects above that size; MANOVA uses all
  - which columns are tested is an argument, `effects(run, out, predictors)`: `--predictors`
    on both scripts, `PREDICTORS` in both notebooks. The default is the dataset's `groups`
    plus the claims column. Rejected: named predictor sets in `dataset.json` - which columns
    one analysis tests is not a fact about the dataset
  - the claims column (`voted_party`) is tested with the grouping columns, as the legacy
    figure had it: a sanity check, the model never saw it
  - the figure shows raw Pillai with its p-value beside partial eta squared, and the
    comparison a runs-by-columns heatmap of Pillai with p-values, laid out like the legacy one
  - open: PERMANOVA alone and MANOVA given-the-others answer different questions; a
    partial PERMANOVA would need a permutation scheme under the other columns
- `compass` is the colleague's CHES comparison from `llm_pushforward` (`ches_comparison`,
  `pushforward.py`), its correlation distance: MDS of the options' distances, Procrustes
  onto the experts' plane, exact permutation tests
  - the positions are topic data, `positions.json` with option ids, computed from that
    project's `ches2019.csv` the way its `ches_positions` does; six of the nine options
    are rated
  - not ported: its second distance (how alike two options' electorates are over the
    subjects' columns) and the comparison against a model trained on real votes
  - needs scikit-learn
- tests: `test_stats.py` and `test_plots.py` were rewritten for the new functions,
  `test_analyze.py` runs every analysis and comparison on fake runs; the snapshot helper
  keeps the aggregation the legacy baselines were frozen with
- decided 2026-11-11
  - the option id `Trikolora` became `TSS` in `election_2021` and `election_2021_legacy`
    (topic, claims map, `en.json`, `plot.json`); `Trikolora` stays as an alias, and every
    reader of a saved results table passes it through `Topic.rename`, so runs saved under
    the old id still load. No saved file was rewritten
  - the colleague's statistics may be corrected during the port (drop Wasserstein, one zero
    floor for KL, zero replacement before ILR, the PERMANOVA faults); at the end old and new
    numbers are compared and the results re-read
  - legacy runs are refused for now; they are to work again once migrated to the new names
  - the plotly figures were redrawn in matplotlib; plotly and seaborn are no longer used
- open
  - old and new numbers of the colleague's statistics are not yet compared: that needs the
    legacy runs, which are still refused
  - superseded and left in the tree: `panel_stats.ipynb`, `simulate.ipynb`,
    `tests/test_analysis.py`, `panel/analysis/plots/distances.py`; `sod_en` and `grid` are
    not ported

## 6. Pending Martin's decision

- income cut points for `to_levels` (§4) - left open on purpose (2026-11-10); only needed for
  grid-versus-survey comparison.

Confirmed 2026-11-10, as implemented: the topic folder shape (§1), repeats as one row per call
(§4), the Czech `media_fees_2026` labels (§4).
