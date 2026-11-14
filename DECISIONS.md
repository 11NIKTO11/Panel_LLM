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
    this made to the run layer (superseded by §18: framings have a folder, nothing is skipped)
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
    changed the value. The paper did not use it
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
  column, without a heatmap; no comparison puts it side by side for several runs - open,
  worth a look
- runs to compare are named one by one; the old notebook took every run of the topic with
  the same response shape
- `effects`, changed from the colleague's MANOVA and PERMANOVA, to compare at the end
  - zeros are replaced multiplicatively by 0.005 before the log-ratio transform, not clipped
    at 1e-12: a clipped zero sat some 25 log-units away and decided the test. The value is
    an argument, `effects(..., delta=)`: `--delta` on both scripts, `DELTA` in both
    notebooks, and the figure's note states it. It only matters where a run has zero
    shares: of the two trials Haiku 5.5 has none and Haiku 4.5 one
  - Wilks' lambda is no longer reported; the paper did not use it
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
    subjects' columns)
  - needs scikit-learn
- the benchmark is the colleague's comparison against a model trained on real votes
  (`ground_truth_section` in `pushforward.py`): `python benchmark.py <topic> <run> ...`,
  `benchmark.ipynb`, `panel/analysis/benchmark/`. Runs are scored subject by subject on what
  each subject claimed
  - a third kind beside analyses and comparisons, with its own script: every benchmark is
    `(ground, out) -> (table, figures)`, and `load_ground` fits the trained model once for
    all of them. `scores`, `knowledge`, `agreement`, `calibration`
  - an outcome is an option or, with turnout, not taking part; a run's answer is the
    options' shares times the probability of taking part, and the probability of not
  - scored are the subjects whose claim says something and that every named run answered;
    runs of different datasets are refused
  - the trained model is his: a ridge multinomial logit (`C` 0.3), 5-fold cross-validated,
    on an indicator per level with none left out (`stats/design.py` `indicators`), a missing
    answer being a level, age in the dataset's bands, employment an indicator per status.
    Recalibration is his too: a cross-validated logit (`C` 1) on log(p + 0.01). Scores mix
    in 1 % of a uniform distribution first
  - which columns the model is trained on is an argument (`--predictors`, `PREDICTORS`),
    by default the dataset's `groups` without the claims column. He used 10 columns
    without the district; ours are the 12 grouping columns
  - changed from his: the mean answer is over the scored subjects, his over distinct
    profiles; the knowledge ratio is empty when the trained model does not beat the claimed
    shares (his divided anyway, and a negative over a negative read as a good ratio);
    intervals are bootstrap over subjects with the fitted probabilities held fixed, so they
    leave out what refitting would add
  - on the two 200-subject trials 151 subjects are scored and the trained model does not
    beat the claimed shares, so the ratio is empty; it needs the full runs
  - open: TabPFN and the other classical models he also compared against; refitting inside
    the bootstrap
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

## 11. After reading the colleague's notebook again (2026-11-12)

`_delete/llm_panel.ipynb` held against the analysis layer; decided with Martin.

- by area every metric, not MAE and RMSE alone: MAE, RMSE, turnout difference, JS, KL,
  correlation, in `area` (a figure each) and in `compare.py` `areas` (a heatmap each; the
  turnout difference is signed and coloured by its size, correlation is not in the error
  colours)
- areas come in the order the dataset declares for its area column (`levels.<column>.order`),
  else most subjects first. `sod` and `regions` declare regions by population, the ranking of
  `datasets/regions/regions.csv`. The same order now shows in `groups`
  - rejected: reading the population file from analysis - an order is all a figure needs,
    and `levels.order` already says it
- `compare.py` `spread`: the standard deviation of each option within each area, a panel per
  run on one colour range (the notebook's grid of heatmaps); a run with one subject per area
  is left out
- how turnout enters a predicted distribution is a setting of the loaded run:
  `load_run(..., weighted=True)`, `--unweighted` on both scripts, `WEIGHTED` in both notebooks
  - weighted (default) is the notebook's "soft voting": each answer's option shares times its
    probability of voting, rescaled to 1. Unweighted counts every answer with a preference
    the same (the notebook's `drop_both`)
  - an unweighted pass writes under `<stem>_unweighted`, so it does not replace the default
  - not added: the notebook's third reading, "Not Voted" as a tenth category. It is the
    weighted shares times the mean turnout with the not-voting share beside them, so it says
    nothing the weighted shares and the turnout difference do not; an error over it mixes
    the two. Open if the paper's regional MAE has to be reproduced as it was
- the distance figures (one run, and across runs) draw a bar from zero per row with its
  interval, not a dot: read by length. Correlation, all near 1, reads worse this way
- `compare.py` `areas` also draws `runs_areas_lines`: every metric a panel, a line per
  source across the areas in their declared order, the source's value for the whole dashed -
  the notebook's regional shape, beside the heatmaps that hold the numbers and intervals
- named sets of runs are data, `config/comparisons.json` beside `presets.json`: a set is a
  topic and `label -> run`, the run named as on the command line. `--set <name>` on
  `compare.py` and `benchmark.py` takes one in place of a topic and runs; its labels name the
  runs and its name the files. `load_runs` takes such a mapping in place of a list
  - rejected: a file per topic folder - every file there is a framing unless the reader
    is told to skip it (reversed in §19, once framings had a folder: §18)
- open
  - interactivity, later
  - distance to the claimed by any grouping column (the notebook's "other aggregations")

## 10. The legacy migration (2026-11-12)

**Run on `results/` on 2026-11-12**, after a trial into a scratch folder: `python
migrate_legacy.py` lists, `--write` migrates, `--write --to <dir>` migrates into another
results folder. The logic is `panel/run/legacy.py`.

- a legacy run is parsed with today's response model and saved through `ResultStore.save`,
  so a migrated run is in every file what a run sent today leaves: keyed by subject id,
  option ids for columns, `options` in the JSON, a record `RunSpec.from_dict` reads
  - the row position of a legacy run is the row of `data/sod/values.csv`: every column the
    two tables share agrees row by row, so position -> id needs no lookup table
  - a `VPick` run saved no JSON; its answers are rebuilt from its table (the turnout pair and
    the one option at 1)
  - new stems, e.g. `sod_legacy_cs_default_OpenAI_gpt-4.1_Prob_t=0.0`; nothing is
    overwritten and the legacy files are only read
- the description is `legacy`, a **stored** description (`"stored": true`, no sentences, in
  `datasets/sod/descriptions/legacy.json`): its prompt table `data/sod/prompts/legacy.csv` is
  the prompts as sent, written once from `data/prompts.csv`, and prepare never renders it
  - why: the legacy wording is not this template language's, and a run's spec has to resolve
    for `check`, `run.py --check` and `analyze.py`
  - rejected: a template that reproduces the legacy text (the old describer's quirks would
    have to enter `forms.json`); a special case in the readers for runs without a description
- what a migrated record cannot know stays empty: `git_commit`, `git_dirty`, `versions`,
  `model_resolved`. `saved_at` is the legacy run's, `response_schema` is the schema that was
  sent (from the folder's frozen `config.json`, old class and field names), and `migrated`
  names the legacy stem and the name before that
- the 200-subject Sonnet run is `sample=200, random_state=42`: that draw over the values
  gives exactly its subjects
- checked on a trial into a scratch folder, all ten runs
  - every table holds the legacy numbers exactly; every normalized table equals the legacy
    one where there was one; the frozen aggregates of `tests/fixtures/legacy_aggregates.json`
    hold to 1e-16
  - the saved prompt equals the stored description with the question for the two runs that
    saved prompts in their JSON (Sonnet, Gemini), and the `Prompt` column of all six `VProb`
    tables equals `data/prompts.csv`. The four `VPick` runs saved no prompt anywhere: that
    they were sent the same prompts is assumed
  - `analyze`, `compare` and `benchmark` read them
- done on Martin's go
  - the ten runs are migrated in `results/election_2021_legacy/`; the legacy-shape files
    (ten runs and `archive/`) are in `_delete/legacy_shape/`, for Martin to remove
  - `legacy_aggregates.json` is rekeyed by the new stems, numbers untouched;
    `legacy_reparse.json` is regenerated (rows are keyed by subject id, the four `Pick` runs
    have JSON); the frozen schema is `tests/fixtures/legacy_schema_sent.json`, checked against
    `config.json` and every record, and `schemas/election_2021_legacy/cs.json` is today's
    generation like the others. The eight legacy tests pass again
  - the migration's own tests need a run in the legacy shape and skip without one
- still to do
  - the `parties` alias and the bare shape of raw JSON in `ResultStore.load` have no reader
    left in `results/` and can go; so can `migrate_legacy.py`, `panel/run/legacy.py` and
    their tests once the one-topic question below is settled
  - `results/election_2021_legacy/analysis/` still holds tables named by the legacy stems
- open, Martin's
  - one topic or two: as `election_2021_legacy` the migrated runs cannot be put beside the
    new runs in `compare.py` or `benchmark.py` (one topic per call), and the topic has no
    `plot.json`, polls or expert positions (the `groups` figure needs nine colours, `compass`
    is skipped). Folding them into `election_2021` as a framing and a question of their own
    would lift all of that; `Legacy` and the script's `TOPIC` are where it would be said

## 6. Pending Martin's decision

- income cut points for `to_levels` (§4) - left open on purpose (2026-11-10); only needed for
  grid-versus-survey comparison.

Confirmed 2026-11-10, as implemented: the topic folder shape (§1), repeats as one row per call
(§4), the Czech `media_fees_2026` labels (§4).

## 12. The tree after the port, sorted (2026-11-13)

Only what today's code reads stays in the tree; the rest is parked outside git, in two
gitignored folders that Martin empties himself.

- `data/` is one folder per dataset
  - `data/sod_legacy/` (not in git) is the prepared data of dataset `sod_legacy`
    (2026-11-13): `values.csv` is the table the legacy prompts were built from, keyed by the
    ids of `sod` (same subjects, same order), `prompts/default.csv` the prompts as sent
    without their question. `datasets/sod_legacy/` declares it with `"source": null` - no
    build step, `prepare` refuses to rebuild it - one stored description `default`, an empty
    `forms.json` and a copy of the 2021 claims map; preset `sod-legacy`
    - the originals keyed by row position (`legacy_clean_data.csv`, `prompts.csv`), the
      earlier cleaned and prompt tables and `overhaul/` (the September shape, the only
      prepared tables of `sod_en`) are in `_delete/sod_legacy/`
    - `employment_status` is not a grouping column there: several statuses are joined by
      ', ', which one status also contains
    - kept as a snapshot of the 2025 data, on Martin's word. The ten migrated runs stay
      dataset `sod`, description `legacy` and are not re-pointed at it: same subjects and
      ids, the prompts are stored as sent, and on `sod` they group by today's columns
    - rejected: a full mirror stack (runs re-pointed at `sod_legacy`); a template that
      rebuilds the 2025 prompts from this table (§10's rejection stands: 354 prompts hold a
      literal `nan`, which the template engine cannot word)
  - `data/prompts.csv` had been tracked although it is derived from the survey; it left the
    index here and stays in history
  - `data/grid/`: the grid's tables in the September shape, parked until the grid is ported
  - `data/profiles/` and `data/prompts/` are gone; the regions tables there were superseded by
    `data/regions/`
- unported code left the tree for `_to_sort/unported/`: `datasets/sod_en/`, `datasets/grid/`,
  `tests/test_english.py`, `config/descriptions/self_v1_en.json`. Their last tracked state is
  commit `ccea2c6`; a port starts from there
- superseded and removed: `datasets/regions/prepare.py` (`build.py`), the two
  `config/descriptions/latent_*` (`datasets/regions/descriptions/`), `paths.GRAPHS_DIR`,
  `panel_stats.ipynb`, `simulate.ipynb`, `tests/test_analysis.py`, `plots/distances.py`
- still in the tree without a reader: `data/skop.csv`, `data/agentury.csv` (the source
  `polls.json` was copied from) (the spent migration code
  left on the same day, §13)

## 13. The legacy topic folded into `election_2021` (2026-11-13)

`election_2021_legacy` is gone; the ten migrated runs are runs of `election_2021`. Settles
the "one topic or two" question of §10.

- why: the two `topic.json` were the same but for name and note. What sets the legacy runs
  apart sits below the topic - no system prompt and the task in the schema text, one
  ungendered question - and a topic holds several framings and questions
  - framing `cs_legacy` (stem part `cslegacy`), question `legacy`: the legacy topic's `cs.json`
    and `questions/default.json`, moved; only name, short and topic changed
  - rejected: keeping the stems by giving the legacy files the shorts `cs` and `default` - a
    new `sod` run with description `legacy` would land on the same stem with other text
- the runs: stems `sod_legacy_cslegacy_legacy_<Client>_<model>_<Shape>_t=0.0[...]` in
  `results/election_2021/`. Tables and raw JSON are byte copies; in each record only `stem`,
  the spec's topic, framing and question, and the name and short of the framing and question
  blocks changed, and `migrated.folded` names the stem and topic before. Checked per run
  before writing: the dataset and description blocks and the three digests are what the new
  spec resolves to
- what it gives: legacy and new runs in one `compare.py` / `benchmark.py` call; the topic's
  `plot.json`, `polls.json` and `positions.json` apply (`groups` and `compass` run)
- checked: the legacy tests pass on the new stems (aggregates to 1e-12, reparse digests,
  schema as sent); for gpt-4.1 `Prob` the area MAE equals the 2025 table; `compare.py` ran
  on that run beside the Haiku 5.5 run
- tests: `LEGACY_TOPIC` is `election_2021`, `LEGACY_FRAMING` `cs_legacy`; the legacy runs
  of the store are those whose record has `migrated`; the schema-as-sent test reads the
  fixture alone, the results folder's `config.json` left with the folder
- parked in `_delete/legacy_topic/`, for Martin to remove: `results/election_2021_legacy/`
  (the ten runs under the old stems, `config.json`, `preferences/`, `sod_claimed_votes.csv`),
  the legacy `topic.json`, and `stem_renames.json` (old stem -> new)
- the migration code left the tree the same day, on Martin's word: `migrate_legacy.py`,
  `panel/run/legacy.py` and `tests/test_legacy_migration.py` migrated into a topic that no
  longer exists; they are in `_delete/legacy_topic/code/`, last tracked at `2690a71`. §10
  describes what they did.
  `tests/baseline.py` did not import before this either (`profile_path` is gone)

## 14. Trial runs out of `results/` (2026-11-13)

On Martin's word: every subset run and every Fake run was a test and left `results/`.

- gone from `results/election_2021/`: the Fake run (`subset=20`), the two 200-subject Haiku
  trials (4.5 and 5.5) with their archive copies, per-run tables and plots, the legacy
  200-subject Sonnet run, and the `*_haiku` comparison and benchmark outputs, which were
  computed from the two Haiku trials alone
  - parked in `_delete/results_test_runs/`, for Martin to remove
- left: nine legacy runs (eight GPT, one Gemini) and the two `regions` runs (`x10`, `x3`)
- where earlier sections say "ten legacy runs", "the 200-subject Sonnet run" or "seen live
  on 200 subjects", they describe what was there then; what those trials showed (§8, §9)
  still stands as observed
- the Sonnet run's entry left `tests/fixtures/legacy_aggregates.json` and
  `legacy_reparse.json`
- left dangling on purpose: the set `haiku` in `config/comparisons.json` and the default
  `STEM` of `analyze.ipynb` name the Haiku runs without their subset, so they match the full
  runs once those are sent

## 15. The grids as datasets (2026-11-14)

Step (a) of porting the grid experiment from `CoRe/_grid_separate`: the grids its runs were
sent, as datasets of their own. Importing the runs (b) and the grid analysis come later.

- one dataset per grid that has runs, plus the colleague's
  - `grid_sod` (360): the survey's vocabulary, incomes `Do 15.000` / `30.001 - 40.000` /
    `Vice než 80.000` - the config as committed (`132038d`); its runs are the three of
    2026-09-11, `grid-sod-r1..r3`
  - `grid_sod45` (360): the same with incomes cut at 25 and 45 thousand - an uncommitted
    edit of that config, sent once on 2026-09-24 under the stem `grid-sod-r1`, which
    overwrote the first repeat of `grid_sod`; that repeat survives only in the old folder's
    `archive/`. Why the incomes changed is not recorded
  - `grid_simple` (384): the colleague's design in Czech words; one run, `grid-simple`
  - `grid_colleague` (384): the colleague's design in their own column names and English ids;
    no description - their prompt text was not delivered - so nothing to send. It is what
    their two delivered CSVs (election, media fees) are read against: their `profile_id` is
    a row's levels joined by `_`
  - not ported: `simple_sod` and the old `datasets/grid` (`income3`, no runs); `sod_en`,
    `simple_en` (English, wait with `sod_en`)
- a grid's source is its own declaration: `"source": "dataset.json"`; each column is a
  factor, its `levels.order` the factor's levels. `panel/grid.py: full_factorial` builds the
  rows (first column slowest, numbered from 1), and each grid's `build.py` is that function
  under the name `prepare` imports, so `panel/prepare.py` has no grid case
  - rejected: a separate factors file beside `dataset.json` - the levels would be written
    twice, once to build and once to order analysis
  - rejected: a grid case in `prepare` (a `"build": "grid"` key or similar)
- descriptions: `default` of `grid_sod`, `grid_sod45`, `grid_simple` is the old
  `grid_description_v1` sentence for sentence; the age clause is a `clause` form of
  `age_range`, the only form a grid needs. The two Czech grids also declare `shared.gender`,
  so a new run may use a question worded by gender
- checked against `_grid_separate`: values and descriptions equal its grid CSVs; description
  plus the old question equals the `Prompt` of every saved copy of every run, archives
  included (5 for `grid_sod`, 3 for `grid_sod45`, 2 for `grid_simple`); every delivered row of
  the colleague is one row of `grid_colleague`
- framing `cs_grid` (stem part `csgrid`) for the grid runs (2026-11-14): their system prompt
  is the one of September, which still names the profile's attributes and the form of the
  answer, and their schema text the general one of that time (options, not parties) - neither
  `cs` nor `cs_legacy`. Built from `_grid_separate`'s topic file, texts copied
  - the schema it generates is the one the runs were sent but for the fixed names
    (`VotingProbabilities` / `VotingPick`, `parties` and the title `Parties` pydantic derives
    from it): `tests/fixtures/grid_schema_sent.json` holds the schema as sent, and a test
    checks the two differ by those names only
  - the question is `legacy`: the same text, which the runs sent after a space
- dropped for now: the comparison scales (`sod_to_grid_categories.json`, `grid_categories.py`,
  `colleague_report_claims.json`); how grids are compared with the survey and each other is
  decided later. `data/grid/` and `_to_sort/unported/datasets/grid/` are superseded

### 15.1 The grid runs brought in (2026-11-14)

`python migrate_grid.py <path to _grid_separate> [--write [--to <dir>]]`, one-off like the
legacy migration: the old folder is only read, new stems, nothing overwritten. Run into a
trial folder first, then into `results/`; the two agreed byte for byte.

- ours, framing `cs_grid`, question `legacy`, `OpenRouter gpt-4o`, `t=0.3`:
  - `gridsod_default_csgrid_legacy_OpenRouter_gpt-4o_Prob_t=0.3_x3`: the three repeats of
    2026-09-11, saved then as three runs `grid-sod-r1..r3`, are one run with a `repeat`
    column - the shape every repeated run has. Repeat 1 is rebuilt from its table (its raw
    JSON was overwritten by the `grid_sod45` run); 2 and 3 are parsed from their raw JSON
  - `gridsod45_..._t=0.3` and `gridsimple_..._t=0.3`: one repeat each
  - each saved prompt equals today's (description, a space, the question) and each repeat's
    numbers equal its old table exactly; `check` passes
  - record: `saved_at` the latest repeat's, `response_schema` the schema as sent
    (`tests/fixtures/grid_schema_sent.json`), code and SDK versions empty, and `migrated`
    listing per repeat the old stem, date, commit (of `_grid_separate`'s repository) and
    whether its answers came from raw JSON or the table
  - rejected: one table per repeat, as before - a repeat is a row of one run
- the colleague's, delivered as per-profile means without prompts: runs over `grid_colleague`
  with the stored, standalone description `delivered` and the framing name `colleague`, which
  has no file - their system prompt and schema were not delivered
  - `gridcolleague_delivered_colleague_OpenRouter_gpt-4o-2024-11-20_Prob_t=0.3_x3`
    (`election_2021`): the mean of 3 valid repeats per profile, one row each, `repeat` 1 -
    the stem says x3 because three calls were made; the spread over repeats is only in the
    delivered table
  - `..._t=0.0` (`media_fees_2026`): one repeat
  - not resolvable, so written at the store's paths by hand; the record carries a `delivered`
    block (`panel.run.record.DELIVERED`: file, sha256, report, `averaged`, a note) instead of
    framing, question and schema, and `analyze.py` does not `check` such a run. The delivered
    tables are copied to `results/<topic>/delivered/`, keeping `_std`; `w_` is
    `p_<option> x p_voted` and is dropped, `Not Voted` is `1 - p_voted`
  - checked: the election run aggregates to the report's table (turnout-weighted shares,
    renormalized) to the second decimal, and turnout 72.6 %
- tests: `tests/test_grid_results.py`; `legacy_stems` now selects by the legacy framing,
  since the grid runs carry `migrated` too
- found and fixed: `effects` raised on a topic with two options (`media_fees_2026`): one
  log-ratio coordinate, and statsmodels' `MANOVA` refuses a single dependent variable. With one
  coordinate `stats.manova` now F-tests the same hypotheses on the least-squares fit, and
  Pillai's trace is the partial R squared `qF / (qF + residual df)` - what it is for one
  variable, so the table keeps its columns and meaning; checked against the one-way ANOVA
  formulas in `tests/test_stats.py`
  - rejected: leaving the MANOVA columns empty for two options - the test is valid there
- found, not fixed: `compare.py`'s `spread` raises on a dataset without an area column (every
  grid) instead of skipping; `analyze.ipynb` still checks every run, so a delivered run stops
  at that cell
- `migrate_grid.py` is spent once Martin has looked at the runs; `CoRe/_grid_separate`
  holds everything it read

## 16. Runs read as kernels (2026-11-14)

The analyses of `CoRe/llm_pushforward` ported as one slice, `panel/analysis/kernel/`, with
`kernel.py` / `kernel.ipynb`. A run is read as a matrix `M`: a row per outcome (each option's
share times P(taking part), and not taking part), a column per distinct profile. Named for the
idea - a Markov kernel from profiles to answers - not for the grid, because a survey run reads
the same way.

### 16.1 Shape

- **Profile**: the values of the columns the run's description reads; subjects with the same
  values are one column, the mean of their answers. A missing value is a level, `(missing)`.
- **Designed / sampled**: read off the kernel, not declared - designed when its profiles are
  every combination of the factors' levels (a grid), sampled otherwise.
- **Code**: pure math in `stats/` (`distributions`, `spectral`, `variance` incl. the exact
  `factorial` split, `shapley`, `sensitivity`, `surrogate`, `replicates`); `kernel/matrix.py`
  builds `Kernel` / `KernelSet` from a `RunSet`; one module per analysis,
  `(kernels, out) -> (table, figures)`, registry `KERNEL_ANALYSES`
- **Moved**: `outcome_shares` from `benchmark/ground.py` to `aggregate.py` (with `outcome_rows`,
  per call), shared by both
- **Not ported** (already here): MANOVA / PERMANOVA (`effects`), national, regional, violins,
  party covariance, CHES maps, the scoring against claims (`benchmark`), output checks

### 16.2 Decisions (settled before the port, as recommended)

- **D1**: a stored description lists the columns its prompts were built from, `"reads"`
  (`legacy` and `sod_legacy/default`: pushforward's ten; `delivered`: the grid's five);
  `reads` on a rendered description is refused
- **D2**: `"together"` in `dataset.json` names columns that share credit as one player in the
  Shapley effects (`sod`, `sod_legacy`: geography = region + district); the default is one
  player per factor
- **D3**: populations start from the run's own subjects (`weights`), uniform over profiles as an
  option (`sensitivity(uniform=True)`); for a grid they are the same
- **D4**: Shapley runs in parallel (`joblib`), no cache, `draws` a parameter (20 by default)
  - rejected: pushforward's CSV cache of replicates - outputs here are regenerable, a cache is
    state to invalidate
- **D5**: the surrogate fits a binary logit for two outcomes (the same softmax, the first
  outcome's effects at zero)

### 16.3 Choices made in the port

- **Variance**: a designed kernel splits exactly (main effects, two-way interactions, the rest,
  noise taken out of the rest); a sampled one by out-of-sample ridge R², as pushforward. Noise
  is the noise variance of one answer over the answers a column averages (pushforward: of one
  answer) - on the survey runs the two differ by under 0.001
- **Uncertainty**, two sources (`matrix.scheme_of`):
  - a sampled kernel: half-samples of its profiles (pushforward's scheme, paired across kernels
    with the same profiles); percentile intervals, Shapley's re-centred on the estimate
  - a designed kernel with repeats: its repeats resampled per subject; always re-centred - a
    resampled cell mean is noisier than the mean of the repeats, which lowers every statistic
    of the spread between cells (seen: Shapley's interval for interest missed its estimate)
  - a designed kernel with one repeat, or delivered means: no interval, said so in the table
  - rejected: the ordinary bootstrap over profiles - duplicates inflate ε² for many-level factors
- **Pairwise quantiles in `intervals`** over 20,000 random pairs, so a replicate costs what
  the estimate does; `scales` uses every pair
- **Agreement**: profiles matched by their values, not their numbers (a run missing a subject
  numbers its profiles differently); the reference is a kernel or `trained on claims`
  (`benchmark`'s logit given the run's factors, grouped into the run's profiles)
- **Notebook defaults**: three GPT legacy runs (reference gpt-4o) and `grid_sod`, so every block
  has something to work on offline

### 16.4 Checked

- **Against pushforward** (`tests/fixtures/pushforward.json`, `tests/test_kernel.py`): its
  prediction files are the legacy runs (an unlisted party empty there, 0 here; it reads empty
  as 0), and `sod_legacy`'s values are its `clean_data.csv`; profiles 3854, one to one
  - exact: noise floor, Dobrushin δ, q99, mean TV to the mean column, effective number of
    axes, minimal pairs (468, 361 of them age), Shapley effects and those of the axes (1e-6)
  - to 0.002: variance shares (district left out of the design, as pushforward did)
  - its cached `results/shapley.csv` was written by an earlier version of its code: its code
    today gives our numbers, which the fixture holds (up to 0.002 from the CSV)
- **Agreement does not reproduce pushforward**, on purpose: its "human kernel" was its best
  classic model, ours `benchmark`'s logit. The std ratio depends on how smooth the reference
  is (gpt-4o: 1.46 against pushforward's, 0.89 against ours) - read "stereotype
  amplification" against a named reference only
- **On the grids**: minimal-pair counts are what the design says; the exact split adds up; the
  colleague's grid has no noise floor (means only), `grid_simple` and `grid_sod45` none either
  (one repeat)

### 16.5 Open

- **Survey weights over grid cells**: mapping SoD's values onto each grid's levels (income cut
  points still open, §6); then `populations` gives a grid's national answer against 2021
- **Agreement across datasets**: `grid_simple` against `grid_colleague` (one design, two
  vocabularies), `grid_sod` against `grid_sod45`, the trained model at grid cells - all need
  the same level mapping
- **Repeats** for `grid_sod45` and `grid_simple`, for a noise floor; Martin sends runs
- **The colleague's joint report** (election x media fees over the same profiles) crosses two
  topics; not ported
- **Agreement's reference**: whether to add a smoother trained reference (pushforward's
  classic models) is open

## 17. Pure model answers from the Anthropic clients (2026-11-14)

- **No tools offered**: `AnthropicClient` and `AnthropicLegacyClient` send `tools=[]` on every
  call. The API runs web search, web fetch or code execution only for a request that declares
  them, so this states what was already true rather than changing it
- **Checked per call**: a response whose `usage.server_tool_use` counts any use fails the call
  (`server_tools_used`); the batch runner's retries and rerun rounds then fail the run loudly,
  so a saved run cannot hold such an answer. The record is unchanged
- **The schema stays `output_format` on `messages.parse`**: that is the SDK's typed helper, sent
  on the wire as `output_config.format` beside `effort` (checked in the SDK source, 2.7.0). What
  the API deprecated is `output_format` as a body field of `messages.create`, which is not used
  - rejected: passing `output_config={"format": ...}` ourselves - the same request, without the
    typed `parsed_output`
- **OpenRouter**: offline too - no `plugins`, no `:online` suffix; not changed
- `tests/test_llm.py` checks the request body and the refusal offline (the SDK's transport
  replaced); that the API accepts `tools: []` is not verified live yet

## 18. Framings in a folder of their own (2026-11-14)

- **A framing is `config/topics/<topic>/framings/<name>.json`**, as a question is
  `questions/<name>.json`. The topic root holds only the files a topic has one of:
  `topic.json`, `plot.json`, `polls.json`, `positions.json`
- **Why**: `framing_names` defined a framing as every `*.json` in the topic folder except a
  list of known names, so each new topic-level file had to be added to that list or it was
  listed as a framing. A framing is now found by path and the list is gone
- **Unchanged**: framing names, their `short`, the stems, the records (they store the name and
  the texts, never a path), the schema fixtures under `tests/fixtures/schemas/`
- older sections of this record name framing files at the topic root (`cs.json`); they
  describe the tree as it was

## 19. Named sets of runs belong to their topic (2026-11-14)

- **`config/topics/<topic>/comparisons.json`**, optional, replaces `config/comparisons.json`:
  `set name -> {label -> run}`. The `topic` key is gone - the folder says it
- **Why now**: a set names stems, which mean something only among one topic's runs, and each
  set already named exactly one topic. §11 rejected the per-topic file because every file in
  a topic folder was read as a framing; §18 removed that reason
- **Command line**: `--set` stands in for the runs, not for the topic -
  `python compare.py election_2021 --set haiku`, likewise `benchmark.py` and `kernel.py`. The
  topic is always named
- **One file, not a folder of sets**: a set is a few lines
  - rejected: `comparisons/<name>.json`, a file per set
- **Not covered**: a set across topics (the colleague's election x media-fees report). Nothing
  compares across topics yet; when something does, such a set needs a home above the topics
- `load_comparisons(topic, topics_dir)`; a topic without the file has no sets

## 20. Prepare does not replace a table a saved run was built from (2026-11-14)

- **The concern**: `data/sod/` and `data/sod_legacy/` (now `datasets/<name>/prepared/`, §23) are not in git, and prepare overwrote
  without asking. A saved run records the sha256 of its values and prompt tables, and
  `SavedRun` refuses values that changed since - so one re-prepare with a changed build or
  wording made the old runs unreadable, with the old tables gone
- **The rule**: prepare refuses (`FileExistsError`) when a table it wrote differs from the one
  that was there and a saved run of any topic records the old one's sha256
  (`panel.run.store.digests_in_use`, over every run record under `results/`). `--overwrite` /
  `overwrite=True` replaces it anyway
  - a table that comes out byte for byte the same, or that no run names, is written as before
  - all or nothing: on a refusal every table prepare wrote in that call is put back, and a
    table that did not exist before is removed, so values and prompts never disagree
  - it writes first and compares after, so the bytes compared are the bytes the writers
    produce; nothing is serialized twice
- **Not guarded**: `forms.json` and the description files (in git, not written by prepare);
  runs under `archive/` and trial runs outside `results/`
- rejected: an `archive/` under `data/<dataset>/`, as results have - a prepared table is
  rebuilt from its source, a run cannot be; the prompts a run sent are in its own table
- checked: `tests/test_prepare.py`; `python prepare.py grid_simple` and `regions` rewrite their
  tables unchanged

## 21. The colleague's prompts are marked as unknown (2026-11-14)

- `datasets/grid_colleague/prepared/prompts/README.md` (then under `data/`) stands where the prompt table would be and says
  what was not delivered and what goes there when it is. No `delivered.csv` is invented

## 22. A topic's reference results sit in its folder (2026-11-14)

- **`config/topics/election_2021/reference.csv`**, moved from `data/election_data.csv`; the
  `reference.path` of `topic.json` is relative to the topic's folder (`Topic.dir`), no longer
  to the repo root
- **Why**: the file belongs to the topic, not to a dataset, and `data/` is one folder per
  dataset. It is a measured fact rather than a setting, but so are `polls.json` and
  `positions.json`, which were already there: the topic folder is everything that defines
  the topic
  - rejected: `data/reference/<topic>.csv` - a second kind of thing under `data/`
- **`data/grid/` is out of the tree** (`_delete/data_grid/`): the old `grid` dataset's tables
  (`income3`, 288 profiles, the September shape with `profile.csv` and `prompts_self_v1.csv`),
  superseded by the grid datasets of §15 and read by nothing; it never had a run
- without a reader: `data/agentury.csv` (copied into `polls.json`) and `data/skop.csv`, moved to
  `_delete/data_leftovers/` with §23

## 23. A dataset's prepared data lives in its folder (2026-11-14)

- **`datasets/<name>/prepared/`** holds `values.csv` and `prompts/<description>.csv`; the
  root `data/` folder is gone, and with it `paths.DATA_DIR`. `Dataset.data_dir` is
  `dir / "prepared"`, the one place the location is stated
- **Why**: `data` and `datasets` were two names for two halves of one thing, and the half
  that is derived sat apart from what it is derived from. A dataset is now found whole by
  one path: declaration, build, wording, raw source, prepared tables
  - the raw survey files were already in `datasets/sod/`, so the folder was never code only
  - saved runs record no paths, so nothing in `results/` changed
- **Cost, accepted**: generated files inside an importable package, and the ignore rules are
  per dataset - `datasets/sod/prepared/` and `datasets/sod_legacy/prepared/` are not in git,
  the grids' and `regions`' are
- rejected: renaming `data/` to `prepared/` - removes the similar names, keeps the split
- moved with `git mv` where tracked; `data/agentury.csv` and `data/skop.csv`, which nothing
  reads, went to `_delete/data_leftovers/`

## 24. Results: a folder per run, a folder per comparison (2026-11-14)

A topic's results folder held every run's four files side by side and two folders,
`analysis/` and `plots/`, with everything derived from any run or any set of runs.

```
results/<topic>/
├── runs/<stem>/          results.csv  results.json  normalized.csv  run.json
│                         archive/  analysis/  plots/  [delivered/]
├── comparisons/<name>/   runs.json  analysis/  plots/
└── <name>_claimed_votes.csv   (legacy, read by nothing)
```

- **A run is a folder named by its stem.** Its files carry no stem: `results.csv`,
  `results.json`, `normalized.csv`, `run.json`; archived copies are
  `archive/<timestamp>.csv` and `archive/<timestamp>_run.json`
  - why short names: a stem is up to ~90 characters, and a stem twice in one path
    (`runs/<stem>/analysis/<statistic>_<stem>.csv`) passes the 260 of a Windows path
  - cost: a file copied out of its folder no longer names its run; `run.json` holds the stem
- **What was derived from one run sits in it**: `analysis/<statistic>.csv`,
  `plots/<name>.png`. A table someone else delivered sits in that run's `delivered/`; the
  record's `delivered.file` is relative to the run's folder and did not change
- **What was derived from several runs is a named comparison**:
  `comparisons/<name>/analysis|plots/`, the name being the set's (`--set`), `--name`, or the
  topic. `runs.json` there is `label -> stem`, the runs it was made from - written on every
  saving pass, so it also covers a comparison that is in no `comparisons.json`. It has the
  shape of a set and can be copied into one
  - `compare.py`, `benchmark.py` and `kernel.py` write into the same folder under one name;
    their files differ by prefix (`runs_`, `benchmark_`, `kernel_`)
  - the `_unweighted` variant is a file suffix inside the same folder, not a folder
- **Code**: `ResultStore.run_dir`, `comparison_dir`, `paths`, `save_comparison` are where
  these paths are built (`analysis()`, `plot()`, `path()` are gone). `Output(folder, save,
  suffix=)` writes into one folder; `comparison_output(store, name, runs)` makes a
  comparison's and records its runs
- **Migration**: `python migrate_results.py [--write]`, one-off - files renamed, never
  rewritten, the sha256 of every file compared before and after. 134 files moved;
  `sod_claimed_votes.csv` stayed. `comparisons/grids/` got its `runs.json` from the set;
  `comparisons/election_2021/` has none - which runs those tables were made from is not known
  - not migrated: the trial runs outside `results/` (§14) and anything under `_delete/`
  - `migrate_results.py` is spent; `migrate_grid.py` (spent too) was pointed at the new paths
- rejected: keeping `<stem>.csv` names inside the run folder - the path length above
