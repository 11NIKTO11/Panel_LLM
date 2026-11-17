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
  - framing `cs_legacy` (stem part `cs-legacy`), question `legacy`: the legacy topic's `cs.json`
    and `questions/default.json`, moved; only name, short and topic changed
  - rejected: keeping the stems by giving the legacy files the shorts `cs` and `default` - a
    new `sod` run with description `legacy` would land on the same stem with other text
- the runs: stems `sod_legacy_cs-legacy_legacy_<Client>_<model>_<Shape>_t=0.0[...]` in
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
- framing `cs_grid` (stem part `cs-grid`) for the grid runs (2026-11-14): their system prompt
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
  - `gridsod45_..._t=0.3` and `grid-simple_..._t=0.3`: one repeat each
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
  `comparisons/<name>/analysis|plots/`, the name being the set's (`--set`) or `--name`; runs named one by one need
  `--name` (2026-11-16). REJECTED: the topic as the default name - it made
  `results/election_2021/comparisons/election_2021/`, which read as a second topic. `runs.json` there is `label -> stem`, the runs it was made from - written on every
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
  `comparisons/election_2021/` had none - which runs those tables were made from was not known
  (by their columns the two Haiku `sod` trials of §14); deleted 2026-11-16
  - not migrated: the trial runs outside `results/` (§14) and anything under `_delete/`
  - `migrate_results.py` is spent; `migrate_grid.py` (spent too) was pointed at the new paths
- rejected: keeping `<stem>.csv` names inside the run folder - the path length above

## 25. The place-level tables of 2025 are runs (2026-11-14)

`LLM_preferences/`: eight tables, two per GPT model (4.1-nano, 4o-mini, 4.1, 4o). For the
country and each region the model was asked ten times at temperature 0 - as an inhabitant
(`citizen_prompt`) or for the result outright (`direct_prompt`) - and only the aggregate of
the ten answers was saved. `python migrate_latent.py <folder> [--write [--to <dir>]]`, one-off.

- **Each table is a run over `regions`**, framing `cs_legacy`, `OpenAI`, `Prob`, `t=0.0`, `x10`:
  - `regions_citizen_cs-legacy_legacy_OpenAI_<model>_Prob_t=0.0_x10`: description `citizen`,
    question `legacy` - together word for word what was sent
  - `regions_direct-legacy_cs-legacy_OpenAI_<model>_Prob_t=0.0_x10`: the stored, standalone
    description `direct_legacy`
- **`direct_legacy`** is `direct` as the notebook of 2025 worded it: the preposition never
  vocalised (`v Středočeském kraji`, `v Zlínském kraji`). Two of fifteen prompts differ from
  `direct`, so the runs could not name `direct`; its prompt table was written once by the
  migration, like `sod`'s `legacy`
- **One row per area, with a `delivered` block** (`averaged: 10`), as the colleague's means
  have: `repeat` is 1, the stem says x10 because ten calls were made. The table is copied to
  the run's `delivered/`
  - what a row is: turnout is the mean; an option's share is the mean of the answers' shares,
    each normalized and weighted by its probability of voting, then rescaled to sum to 1
    (`aggregate_results_from_df` of that time) - not a mean answer. The spread over the ten
    answers is lost
  - unlike the colleague's, these resolve: description, framing, question and schema are
    today's files. They are not checked - fifteen rows stand for a hundred and fifty calls
  - the prompts are rebuilt from the notebook (`election_preferences.ipynb`, last at commit
    `79b2189`), compared with today's rendering for all fifteen areas; no saved file held them
  - unknown: the date of the runs, and whether that commit is the notebook exactly as run
- `legacy_stems` in the tests now means the legacy framing and a `migrated` block, since these
  runs carry the legacy framing too; `tests/test_region_results.py` covers them
  (`test_latent_results.py` until 2026-11-16; "latent" was the old word for a run over `regions` - a region run)
- **Not a run**: `results/election_2021/preferences/LatentPreferences.csv`, an older table of
  one country-level row per model whose prompt is not known, with its four figures
  (`latent_<model>.png`). Its numbers are neither the `citizen` nor the `direct` row for Česko

### 25.1 What else was sorted the same day

- the figures of 2025 (`_to_sort/graphs/`, 36 of 40) sit in their run's `plots/2025/` under
  their old names; a name without a shape is the `Prob` run and `_single` the `Pick` run -
  matched by the numbers in the figure for gpt-4o and gpt-4.1-nano, by the name for the rest
- `LLM_Panel_one_party/` (the tables given to the colleague) equal the four `Pick` runs number
  for number - nothing to migrate
- `tmp/skiped` (subjects skipped by gpt-4o-mini and gpt-4.1 in 2025) belongs to no saved run:
  every legacy run has all 3880 subjects
- parked in `_delete/`, for Martin to remove: `tmp/docs/`, `tmp/skiped`, `_to_sort/unported/`
  (`sod_en` and the old `grid`, never run; last tracked at `ccea2c6`)
- `migrate_latent.py` is spent

## 26. A run can be packed (2026-11-15)

`python pack.py <topic> <run> [--zip] [--copy] [--to <dir>]` writes `packs/<topic>/<stem>/`
(gitignored): everything needed to read one run without the repo. `panel/pack.py`.

- **What a pack holds**
  - `run/`: the run's folder as saved - results, record, `archive/`, `delivered/`, old figures
  - `dataset/`, `topic/`: only the files the run names, at the paths they have under
    `datasets/<name>/` and `config/topics/<topic>/`: declaration, forms, the description, the
    claims map, values, the prompt table; `topic.json`, the framing, the question, the
    reference results, `plot.json`, `polls.json`, `positions.json`
  - `MANIFEST.json`: per file its sha256, its size and where it was copied from; `built_from`
    names the packed files whose hash the run's record carries; the commit of the code
- **Tables and figures are made again**, into the pack's `run/analysis/` and `run/plots/`, by
  every analysis of `analyze.py`; a file without `from` in the manifest was made, not copied.
  `--copy` takes the saved ones as they are. The saved run is not touched
- **Refused when the data moved**: a values table, prompt table or forms file that is not the
  one the record gives the sha256 of raises, before anything is written - a pack would set a
  run beside data it was not built from
  - a hash the record does not carry is listed as `unverified`, a named file that does not
    exist as `missing` (the colleague's prompts and framing, §21) - facts of the run, not errors
- **A pack is never written over**; an existing folder raises
- `python pack.py --verify <pack>` checks a pack's files against its manifest
- REJECTED: mirroring the repo's paths inside a pack (`results/<topic>/runs/<stem>/...`), which
  would let a pack be laid over a checkout - the stem twice in one path does not fit a Windows
  path (§24)
- REJECTED: packing the dataset's raw source and the code. The source is licensed survey data
  and the record names the commit
- not packed: the comparisons a run is a member of
- a pack of a `sod` run holds the survey's values table

### 26.1 Sorted the same day

- `tmp/` is empty: its scripts are in `_delete/tmp_scripts/`, `tmp/_artifical_matrix/` in
  `_delete/tmp_artifical_matrix/`. Of its 151 files only the two tables are here (the
  colleague's delivered runs, §15.1); all 151 are in `CoRe/_grid_separate/colleague/`
- `_to_sort/` is gone, moved into `_delete/` by Martin

## 27. `grid_colleague` is `grid_simple_en` (2026-11-15)

Martin's naming: the dataset is the simple design in English ids, beside `grid_simple` (the
same design in Czech words); that it is the colleague's grid is said in its title and note.

- `datasets/grid_simple_en/`, short `grid-simple-en`; description `delivered` and framing
  `colleague` keep their names
- **its two saved runs were renamed with it**, or they would name a dataset that does not
  exist: `grid-simple-en_delivered_colleague_OpenRouter_gpt-4o-2024-11-20_Prob_t=0.3_x3`
  (`election_2021`) and `..._Prob_t=0.0` (`media_fees_2026`). In each record `stem`,
  `spec.dataset`, `dataset.name` and `dataset.short` changed and a `renamed` block names the
  old stem; no result, table or hash changed
- `grid-simple` alone now names two runs; a run of `grid_simple` is found by `grid-simple_`
- §15, §15.1 and §21 above say `grid_colleague` - the name of their day

## 28. The first two sod grids are retired; `grid_sod` is cut at 25 and 40 thousand (2026-11-15)

Martin's decision, on reading the first grid: its incomes do not span incomes, so neither it
nor its runs say what a grid over the survey's vocabulary should.

- **`grid_sod_retired`** (was `grid_sod`): incomes `Do 15.000` / `30.001 - 40.000` /
  `Vice než 80.000 Kč` - 15 to 30 and 40 to 80 thousand are in no profile
- **`grid_sod45_retired`** (was `grid_sod45`): cut at 25 and 45 thousand; covers every income,
  superseded by the cut at 40
- **`grid_sod`** (new): `Do 25.000` / `25.001 - 40.000` / `Vice než 40.000 Kč`, otherwise
  `grid_sod45` - 360 profiles, description `default`. Prepared; **no run has been sent it**
- the retired datasets stay in `datasets/` with RETIRED in title and note: a run names its
  dataset and is read against its values, so they cannot go while their runs are kept
  - their runs were renamed with them, as in §27 (`renamed` block, nothing else changed):
    `grid-sod-retired_default_cs-grid_legacy_OpenRouter_gpt-4o_Prob_t=0.3_x3` and
    `grid-sod45-retired_default_cs-grid_legacy_OpenRouter_gpt-4o_Prob_t=0.3`
  - the `grids` set labels them `grid sod retired` and `grid sod45 retired`
- REJECTED: changing the incomes of an existing grid in place - its saved run would stand
  beside profiles it was not sent (§20)
- REJECTED: moving the retired grids and their runs to `_delete/` now -
  `grid-sod-retired_..._x3` is the only grid run of ours with repeats; the kernel notebook's
  default, the kernel examples and `tests/test_kernel.py` read it
- open: once `grid_sod` has a run (with repeats), the `grids` set, `kernel.ipynb`'s default
  and the tests move to it and the retired grids can go
- §15, §15.1 and §16 above say `grid_sod` and `grid_sod45` for what are now the retired grids

## 29. The model receives every schema text, and the record says what was sent (2026-11-15)

Found in the architecture review of `panel/`: Pydantic writes a field's description beside the
`$ref` to its model, and each path to a provider treated that differently.

- what happened to the turnout field's description (`schema.turnout.field`, which carries
  "the sum must be 1.0"):
  - OpenRouter to an OpenAI model: dropped without a word (`move_ref_descriptions` looked for
    `$defs` in the wrong dict)
  - the Anthropic SDK (`transform_schema`): dropped
  - the OpenAI SDK: kept, the definition inlined at the field
  - Gemini: sent as generated
- **the generated schema has no description beside a `$ref`** (`panel/topic/response.py`): the
  framing's `turnout.field` follows `turnout.description`, after a blank line, in the turnout
  model's own description. One place, every client; the framing files did not change
  - the five schema snapshots were regenerated, deliberately
  - `tests/test_schemas.py` refuses a schema with anything beside a `$ref`
- **`response_schema_sent`** is in the record beside `response_schema`:
  `client.schema_sent(model, response_format)`, the schema as that client's request carries it
  (the SDK's rewrite for Anthropic and OpenAI, the fix-ups for OpenRouter, the generated one
  for Gemini and Fake). `execute` asks for it before the first call
  - OpenAI's is read from `openai.lib._pydantic.to_strict_json_schema`, a private name of the
    SDK - NOT checked against a request
- `panel/llm/schema_compat.py` loses nothing any more: a description beside a `$ref` is
  appended to its definition's, and for Claude through OpenRouter a number's bounds are written
  into its description (`bounds_as_text`, as Anthropic's SDK does) instead of being removed
- `tests/test_llm.py` holds both Anthropic clients to the request as it leaves the SDK: the
  whole body, and for every framing and both shapes that the schema says every text of the
  framing, lists the labels in order and states the bounds
- the five runs sent before this carry a `schema_note`: the three grid runs through OpenRouter
  (how `CoRe/_grid_separate` built the schema then is not known) and the two Haiku `regions`
  runs (the text did not reach the model)
- REJECTED: fixing it per client - the Anthropic client would have to stop using
  `messages.parse`, and each client would keep its own handling of the same schema
- open: whether OpenAI's strict mode takes a description beside a `$ref` today - a live call

## 30. `Trikolora` is gone from the saved tables; spent code left the tree (2026-11-15)

- the two Haiku `regions` runs were the last tables under the old option id (§9 kept it as an
  alias): header and `framing.labels` rewritten to `TSS` in `results.csv`, `normalized.csv`
  and the archive, a `renamed` block in the record, no number changed. The alias `Trikolora`
  left `topic.json`; `p_Trikolora` stays for the colleague's delivered table
- moved to `_delete/`: `panel/analysis/validate.py` (only a test used it; `panel/run/check.py`
  does its work), `migrate_grid.py`, `migrate_latent.py`, `migrate_results.py` (spent)
- no saved record is of the legacy shape any more, so `load_run` and `ResultStore.load` read
  today's shape only

## 31. An analysis is two halves; review follow-ups (2026-11-15)

From the architecture review of `panel/` (its plan, `_plan_analysis_split.md`, is spent).

- **an analysis module has three functions**
  - `compute(run, ...) -> {name: table}`, the main table first; no output, no figure
  - `draw(run, tables, ...) -> {name: figure}`
  - `<name>(run, out, ...)`, the public one, unchanged for callers: compute, draw,
    `output.save`. Each of the three names the parameters it uses
- **a comparison calls `compute`** of the single-run analysis it builds on
  - gone: the throwaway `Output(..., save=False)` in four comparisons, the `_Kept` output of
    `comparison/compass.py`, and the single-run figures that were drawn and discarded
  - gone: `comparison/areas.py` rebuilding and splitting `area.py`'s column names; it asks
    `area.column(metric, source, reference)`
- **names of estimates are made in one place**: `analysis/estimate.py` (`low`, `high`,
  `bounds`, `is_end`) for `<name>_low` / `<name>_high`; `sources.versus` for a pair
- one function measures every pair of sources, `sources.pair_distances`; `distance` and
  `area` both stand on it. `PAIRS`, `TURNOUT` and `shown` moved to `sources.py`
- `RunSet.stands` (claimed label -> the run that stands for it), `RunSet.labels`,
  `SavedRun.title` replace their copies
- `metrics.py` and `bootstrap.py` are in `analysis/stats/`; `bootstrap.interval` is
  `bootstrap.intervals`, so it no longer shares a name with `replicates.interval`
- checked: every table of every analysis and comparison over three saved runs and three sets
  is byte for byte what it was before
- REJECTED: a frozen dataclass per analysis instead of a dict of tables - nine types for
  tables that already have names
- REJECTED: one generic wrapper routing `**params` to `compute` and `draw` by their
  signatures - the public signature disappears and a misspelt parameter is dropped silently.
  Martin: kept up by hand he would take a dispatcher keyed by an enum,
  `run_it(AREA, run, out, **params)`; here the explicit function costs the same
- REJECTED: making the capturing output official - the figures would still be drawn
- REJECTED (changed while executing): keying an estimate's value and ends by a column level,
  flattened on writing. The tables' column orders differ, so flattening needs a rule per
  table; naming functions give the same single place with the tables left as they are saved
- not done: the comparisons, `benchmark/` and `kernel/` keep one function each; they take
  this shape with the kernel split (review A2)
- the same review, smaller:
  - `analysis/levels.py:banded` is the one place a numeric column is cut into its bands;
    `stats/design.py:indicators` takes the column already cut; `area.ordered` is `members`
  - a topic's files are read from `topic.dir`: `load_framing`, `load_question`,
    `load_plot_style`, `load_polls`, `load_positions` take no `topics_dir`; `load_run` and
    `load_runs` take `datasets_dir` and `topics_dir`
  - `plots/style.py` has `finish` and `framed`; every title is anchored by its top now
  - `panel/config.py` has `check_keys` and `check_short`; `cli.py` at the root holds what the
    entry scripts share
- open, from the review: `kernel/matrix.py` holds five things (A2); `check` takes a `Resolved`
  though it says "from the saved files alone" (A6); `SavedRun.weighted` is an analysis
  option carried by the loaded run (A7); `store.save` is called twice and builds two records
  (A8); `prepare` imports `run.store`, `output.py` imports `panel.run` for two names (A9)

## 32. The rest of the review: the kernel in four modules, and what was left as it is (2026-11-15)

- **`analysis/kernel/` is four modules where `matrix.py` was one**
  - `matrix.py`: `Kernel`, `KernelSet`, `assembled`, `push` - the kernel itself
  - `loaded.py`: `factor_columns`, `kernel_of`, `load_kernels`, `describe`
  - `factors.py`: `design`, `level_designs`, `players` - functions of a kernel, no longer its methods
  - `replicates.py`: `scheme_of`, `replicas`, `replica_answers`; the schemes are
    `HALF_SAMPLES` and `RESAMPLED` (were `HALVES` and `REPEATS`, the second a name `noise.py` also has)
  - `assembled(label, run, calls, factors, designed, averaged)` takes what it needs; the empty
    kernel `kernel_of` built to hand it as a prototype is gone
  - `shapley.py` takes its replicates from `replica_answers` and no longer repeats the scheme's branches
- **the benchmark takes runs**: `load_ground(runs: {label: run}, ...)`; topic and style are the
  first run's. `benchmark/` no longer imports `comparison/`, and `kernel/agreement.py` no
  longer makes up a run set with empty polls
- **`check(store, stem, topic, framing)`** reads the record and the saved answers only: the
  calls are the record's subjects times its repeats, the prompts the saved ones. It no longer
  resolves the spec, so a run can be checked after its dataset was prepared again
- **`ResultStore.save` is one call**: the archived copy and the plain files from one table and
  one record. `timestamped` is gone
- **a client says whether it takes an effort**: `BaseLLMClient.TAKES_EFFORT`, true for the
  Anthropic clients; `resolve` reads it and no longer inspects a constructor
- **the experts' scale is data**: `positions.json` has `"scale": [0, 10]`, `Positions.scale`;
  the two `compass` figures read it
- `pack` takes `datasets_dir` and `topics_dir`
- named in `plots/style.py`: `AGREEMENT`, `SHIFTS`, `DISTANCES`, the colour ranges the kernel
  figures wrote as literals; `scales.py`'s `DELTA` is `DOBRUSHIN`
- `panel/digest.py` is the one place a file's sha256 is taken and a record's `<name>_sha256`
  keys are written and read; `Dataset.built_from(description)` names the files a run is built
  from, for `resolve` and `pack`. `resolve.SUBJECT` is `dataset.INDEX`; the presets read their
  number fields off `RunSpec`; `kernel/` and `benchmark/` name interval ends through `estimate.py`
- a saved `parties` is no longer read (`response.py`): no saved answer has had one since §10
- fixed on the way, both on a kernel of one factor (the `regions` runs), both now tested:
  - `kernel/variance.py` stacked a scalar zero for the interactions it does not have
  - `kernel/surrogate.py` matched a one-level MultiIndex against a plain index
- checked: every table of the kernel analyses over the grids, the two `regions` runs and one
  survey run, and of the benchmarks over two survey runs, is byte for byte what it was
- REJECTED: `compute` / `draw` (§31) for `comparison/`, `benchmark/` and `kernel/`
  - nothing calls their numbers without their figures, which was the reason for the split
  - several kernel figures draw what is not a table - every pair's distance, the noise's
    pairs, the sorted matrix - so `draw` would compute it a second time, or `compute` would
    stop returning tables only
- REJECTED: `weighted` as a parameter of the analyses instead of a field of `SavedRun`
  - it would pass through every `compute` that reads the sources, their comparisons and
    `ranked`; one reading option on the loaded run is the smaller thing
- left as it is, on purpose:
  - `panel/prepare.py` imports `run.store.digests_in_use`: prepare must not replace what a
    saved run was built from (§20), so it has to look at the results
  - `Output` reads `ANALYSIS_DIR` / `PLOTS_DIR` off `ResultStore`, which owns every results path (§24)
  - `run.py` constructs the client: it is the composition root
  - a kernel's factor columns are read from today's description file: a stored description's
    `reads` is not in the record
  - `panel/llm/runner.py` keeps its unused fallback and its prints, with the package's
    copy-as-a-unit contract
- open: the kernel and the benchmark each have an `agreement` that computes the same
  per-outcome correlation and ratio of standard deviations, per profile and per subject

## 33. Hyphens in shorts; a run that was brought in is never saved over (2026-11-16)

- **A short of several words is hyphenated**: `grid-simple`, `grid-simple-en`, `grid-sod`,
  `grid-sod-retired`, `grid-sod45-retired`, `sod-legacy` (datasets), `cs-grid`, `cs-legacy`
  (framings), `direct-legacy` (description). Formerly the same letters run together
  (`gridsimpleen`), which read as one unknown word
  - the underscore still separates the parts of a stem and is still refused in a short
  - 21 run folders renamed, their records' `stem` and `short` fields rewritten and a `renamed`
    entry added; `renamed` is a list where a record has more than one. No table was touched
  - the stems in the sections above were rewritten to today's, except where a section names
    a stem of before the `retired` rename (`gridsod_...`, `gridsod45_...`)
  - not renamed: packs already written, anything in `_delete/`
  - REJECTED: leaving the framing and description shorts alone - half a stem hyphenated
- **A run with a `migrated` or a `delivered` block is never saved over** (`ResultStore.refuse`,
  `record.KEPT`), whatever `overwrite` says; `execute` asks before the first call
  - why: such a run was brought in, not sent, so it has no `archive/` - a save over it would
    replace the only copy. Every run `save` wrote has an archived copy per save
  - a legacy or retired run sent again is a new run under a name of its own; the blocks, not
    the words `legacy` / `retired` in a name, are what marks a run as kept
  - REJECTED: archiving the brought-in files on the first save over them - it would leave a
    run under a legacy stem that is not the legacy run
- **Runs named one by one need `--name`** (§24): no comparison is called after its topic.
  `results/election_2021/comparisons/election_2021/` and `preferences/` are deleted

## 34. How finely a model tells subjects apart: `resolution` (2026-11-17)

The question: a run sends thousands of different prompts - does the model answer each, or sort
people into a few bins? gpt-4o-mini on `sod`: 3854 different prompts, 650 different answers,
no column holding more than 16 different values.

- **Two answers are the same when every topic column matches**; "distinct" is the count of
  different answers. Formerly the check counted "identical answers" - every answer with a twin,
  3591 of 3880 - a number with no scale that read as an error
  - REJECTED: "strictly distinct", two answers matching in no column - a property of a pair
    that gives no count for a run (not transitive), and nearly empty: every answer shares a
    0.05 with every other somewhere
- **The check prints one line, as a canary**: `distinct answers 650 (16.8%) of 3880; 289 given
  once, the most common 130 times` (`check.Distinct`), over all calls
- **`panel/analysis/resolution.py`**, second in a full pass and at the top of `analyze.ipynb`,
  after the subjects' table. On the answers as saved, never normalised:
  - `resolution` - per column, how many different values and the three most common with
    their share; figure: a stem at every value, as tall as the share of answers holding it
  - `resolution_values` - every value of every column with its count and share
  - `resolution_answers` - how much of the run the most common answers cover, against the
    same curve for the prompts: expected (dashed) is an answer of its own for every different
    prompt, actual (solid) is what was given
  - `resolution_columns` (figure only) - that curve per column, a line per option in its
    colour, turnout dotted (its first column; the second is its mirror)
- **A value is a stem, not a bar**: a hairline ending in a dot. A bar has a width and reads as
  a range of values; the point is that the answers hold exactly that number
- **A run that repeats is counted within each repeat** - one pass over the subjects, an answer
  each. The mean curve is drawn, shaded from the least to the most, and the count is a range
  (`310-313` of 360 for the retired `sod` grid). The values a column holds are counted over
  every call: they are what the model wrote
  - REJECTED: over all calls - the model's noise between repeats of one prompt passes for
    telling subjects apart (744 answers for 360 prompts)
  - REJECTED: repeats averaged per subject first - a mean of three answers is nearly always
    its own (360 of 360), and the numbers the model rounds to are gone
  - REJECTED: the count over all calls divided by the repeats - a model that answers a prompt
    the same every time would count a third of its answers (248 where a pass has 311)
  - the band is least-to-most, not a standard deviation: there are three repeats
- **Not for a run kept as a table** (`delivered`): its rows are means, not answers
- **`SavedRun.prompts`**: the prompt of every saved row, so answers can be held against them
- **`spread` lost its figure of the columns** (the violins; `plots.distributions.violins` is
  gone). It showed normalised shares per subject, and `resolution` shows the same columns as
  written, which says more. `spread` keeps its table and the two distance histograms
  - REJECTED: stems on the normalised shares - normalising turns a handful of round numbers
    into many nearly equal ones, so "exactly this value" means nothing there (a rug of dots)
  - the distance histograms say little within one run (0.149 mean Euclidean has no yardstick);
    kept for `compare.py`'s `spread`, which sets runs beside each other
- **`analyze.ipynb` has no "What the answers say" block** any more; `answers` is still the
  first analysis of `analyze.py`. The check cell is three lines: framing, check, report
- **`analyze.ipynb` has no "Spread of answers per subject" block** either: within one run its
  table repeats the stems and its distances have no yardstick. `spread` still runs in
  `analyze.py` and feeds `compare.py`. The single run's map (`compass`) is 7.5 inches wide, not 10
- **The turnout difference keeps its sign and says whose it is**: the measure is
  `Turnout, first - second` (`sources.TURNOUT`, formerly `Turnout difference`), a row
  `predicted vs actual` being predicted minus actual, and the figures' notes say so. Tables
  saved before carry the old column name until they are made again
  - REJECTED: the absolute difference - the sign is the finding (a model over- or
    underestimates turnout; respondents over-claim), and an interval that crosses zero
    would turn into one that never touches it
- **Not in `compare.py`**: no comparison of runs by resolution, by decision
- open: on the per-column figure an option whose colour is near-black (`PirSTAN`) differs
  from the expected line only by being solid

## 35. Effort beyond Claude: OpenRouter's vocabulary, asked of the provider per run (2026-11-17)

- **One vocabulary, OpenRouter's**: `minimal, low, medium, high, xhigh, max`
  (`panel/llm/base.py:EFFORT_ORDER` without `none`; `spec.EFFORTS`). The word in the spec is
  the word sent, and the stem keeps `e=<effort>`
  - why: the providers name different levels and each model takes a subset - Claude
    `low .. max`, OpenAI adds `none` and `minimal`, Gemini 3 stops at `high`, Gemini 2.5 takes
    a token count. OpenRouter already takes one word for all of them
  - REJECTED: a translation table of our own (`max` -> Gemini `high`, ...) - the stem would
    no longer say what the provider received, and it breaks whenever a model's levels change
  - REJECTED: `none` as an effort - `--effort none` already means "set no effort"
    (`presets.NONE`); a provider's `none` means "no reasoning". Adding it needs another
    spelling for one of the two
- **`OpenRouter` takes an effort** (`TAKES_EFFORT`), sent as `reasoning.effort`. The native
  `OpenAI` and `Gemini` clients still refuse one: no preset uses them with a reasoning model
  - how OpenRouter turns the word into the provider's setting (its docs, 2026-11-16): OpenAI
    the same word; Gemini the nearest `thinkingLevel`; Claude a thinking budget that is a
    share of `max_tokens` (`max` / `xhigh` 95%, `high` 80%, `medium` 50%, `low` 20%,
    `minimal` 10%). So `e=high` on `OpenRouter` and on `Anthropic` are different settings for
    the same Claude model; the client's short name in the stem tells them apart
  - `OpenRouterClient(max_tokens=)`: None sends none, as before; with an effort and none
    given, 16000 is sent (`EFFORT_MAX_TOKENS`, the Anthropic client's default) so that a
    budget which is a share of it has a known base. `client_parameters` records both
- **A level the model does not take is refused before the first call**
  (`panel/run/effort.py:effort_tier`, called by `execute` and by `run.py`)
  - why ours: OpenRouter moves an unsupported level to the nearest one without saying so
  - `BaseLLMClient.efforts(model)` asks the provider: lowest first, empty for a model that
    takes none, None where there is no list to ask
    - `Anthropic` / `AnthropicLegacy`: the Models API, `capabilities.effort`
    - `OpenRouter`: the model catalogue, `reasoning.supported_efforts` - no such key is a
      model without an effort setting, null one that takes every level (shape checked against
      the live catalogue, 2026-11-16)
    - a level name outside `EFFORT_ORDER` raises: its place among the others is unknown
  - a client built with another effort than the spec's is refused too; the client's is the
    one sent
  - `resolve` stays offline: it checks only that the client takes an effort at all, so
    `--show` and the tests make no request
- **The record says where the effort stood**: `"effort": {level, rank, of, levels}`, null for
  a run without one; `rank`, `of`, `levels` null where the provider has no list. `run.py`
  prints it before asking: `effort high: 3 of 5 (...)`
  - `levels` is as the provider reports it, a `none` included, so `of` counts it
  - the rank is a position, not an amount: the same rank on two models is not the same thinking
  - REJECTED: the tiers as fields of `RunSpec` - the spec is what is declared; a field the
    provider reports would break the spec comparison of `ResultStore.refuse` and the reading
    of older records
  - REJECTED: a table of tiers per model in the repo - it goes stale with every new model
- not done: no live call. The Anthropic lookup is tested against the SDK with a stand-in
  transport, OpenRouter's against a stand-in catalogue; `run.py --check` does not print the tier

## 36. A call fails in one of three ways; the runner states no default (2026-11-17)

- **Three failures of the `llm` package's own** (`panel/llm/errors.py`), and every client turns
  what its provider raised into one of them:
  - `RateLimited` - the runner waits (`2^attempt` s plus jitter) and tries again
  - `CallFailed` - tried again at once: network, timeout, 408 / 409 / 5xx, an answer that is
    not JSON or not the schema, a refusal, an answer cut off, a server tool used
  - `BatchFailed` - the batch stops: any other 4xx (wrong key, no credit, unknown model, a
    request rejected as such)
  - `status_error(status, message)` is the one mapping from an HTTP status
  - why: the runner sorted by Python type and knew only `openai.RateLimitError`, so one failure
    was handled four ways - OpenRouter's 429 retried with no wait, Anthropic's skipped, a wrong
    key retried three times per call on one client and skipped on another
- **Anything else a call raises stops the batch too**: a bug, or a failure no client foresaw.
  Calls not yet started are dropped, those under way finish, the error is raised; `execute`
  saves nothing
  - formerly such a call was skipped and sent again in a rerun round
  - so `AnthropicClient`'s refusal of a temperature stays a `ValueError`: it is a wrong run, and stops it
- **A client never returns None.** `OpenAIClient` returned `output_parsed`, None on a refusal;
  the runner kept it as an answer and the save failed on it after every call was paid for.
  Now a `CallFailed`; a client that returns None all the same stops the batch (`TypeError`)
- **`run_llm_batch` and `_process_single_item` take every argument by keyword and default
  none**; `execute` is the one place `max_workers`, `retries` and `rerun_rounds` have defaults
  - why: `execute` always passed its own, so the runner's were never in force while reading as
    if they were; a forgotten `model` went to `gpt-4.1-nano`
  - this reverses the part of §32 that kept the runner's fallback: `run_llm_batch` no longer
    builds a client by the model's name. `create_client` stays in the package; the prints stay
- not done:
  - a `Retry-After` the provider sends is not read; the wait is the runner's own
  - `GeminiClient` turns a failed connection into nothing - the SDK names no error for it -
    so it stops the batch
  - a call that fails every attempt still loses the run (`execute` raises, nothing saved) - changed in §37
  - no live call

## 37. An answer is judged before it is accepted; an incomplete run is saved under a name of its own (2026-11-17)

Seen in `regions_citizen_cs_neutral_Anthropic_claude-haiku-5-5_..._x3`: 4 of 45 answers empty
(`voted: 0, not_voted: 0, options: []`), 11 with options summing to 0.98 .. 1.20, all saved as
ordinary rows and rescaled. The schema cannot say that probabilities sum to 1.

- **`panel/run/accept.py:judge`** looks at each parsed answer; the schema sent and the reading
  of saved runs do not change
  - accepted: each group of probabilities (the options, turnout) sums to within the run's
    bounds, no option listed twice
  - flawed but usable: a sum outside the bounds (`options_sum`, `turnout_sum`), or a subject
    certain not to vote who is given no preference (`no_preference`)
  - unusable: an option twice (`duplicates`), turnout summing to zero (`turnout_zero`), no
    preference for a subject who may vote (`options_zero`)
  - a sum of zero is refused whatever the bounds: it cannot be normalised
- **A flawed answer is tried again** within the call's attempts (`retries`). If none is
  accepted, the best usable one is kept - the one whose sums are nearest 1, the earliest on
  a tie - and the record names it: `"faulty": {"<id>[:<repeat>]": [reasons]}`. An unusable
  answer is never kept; the call goes to the rerun rounds like a failed one
  - why keep: how badly a model answers is a result (gpt-4.1-nano); dropping the call would
    hide it, and analysis reads a faulty row as it read every row before
  - a kept answer is not sent again in the rerun rounds, so a weak model costs at most
    `retries` calls per subject, not `retries * (rerun_rounds + 1)`
  - REJECTED: summing a duplicated option - a guess at what the model meant
  - an answer listing fewer than all options is still filled with zeros; the sum bound
    catches the cases that matter
- **The bounds are the run's**: `RunSpec.sum_low`, `sum_high`, of the data kind in
  `presets.json` (base 0.95 and 1.05), `--sum-low` / `--sum-high` on `run.py`. One pair for
  options and turnout. None leaves a side open, which is what every earlier run did, so an
  older record reads truthfully
  - not in the stem: two runs differing only in the bounds share one, and `ResultStore.refuse`
    keeps the second from saving over the first without `overwrite`
  - so a saved run sent again with today's base is "a run of a different spec";
    `run.py --overwrite` (2026-11-17) saves over it, and `run.py` now refuses before it asks
    to send rather than after. The old run's archived copies stay; a run with a `migrated`
    or `delivered` block is still never saved over
  - specs are compared as specs (`store._same_spec`), a field the saved record lacks read at
    its default: a run saved before `sum_low` existed equals today's run with
    `--sum-low none --sum-high none`. Formerly the dicts were compared, and no command
    could match such a run
- **What came back is always saved.** A run with a call missing goes to
  `runs/<stem>_incomplete/`, its record carrying `incomplete: true` and `missing` (the calls,
  named as in `results.json`); the run's own folder only ever holds the whole run
  - when: calls left without a usable answer after the rerun rounds (`RuntimeError`, after
    the save); a batch stopped by `BatchFailed`, by an error no client foresaw or by an
    interrupt (that error, after the save). Nothing came back, nothing is saved
  - a stopped batch keeps the calls that were under way: `run_llm_batch` raises
    `BatchStopped` carrying the results, from what stopped it
  - `load_run` refuses an incomplete run, so no analysis reads one; `check` reads it and says
    "NOT the same subjects"
  - REJECTED for now: a share of calls that may be missing in a run saved under its own stem
    (5% was discussed) - zero until it proves a problem
- `check` prints `kept though flawed` with a count per reason, for a record that has the block;
  `describe` says how many
- `run_llm_batch` takes `review` and returns `(results, flawed, skipped)`; `Flaw` is the
  package's word for what a review found
- not done:
  - (done, §38) `run.py ... --missing`: send only the missing calls of an incomplete run
  - `run.py --check` does not find the `_incomplete` folder; `check(store, stem + "_incomplete", ...)` does
  - which of the saved runs hold answers that would now be flawed is not counted
  - no live call

## 38. `--missing` completes an incomplete run (2026-11-17)

- **`python run.py <the run's options> --missing`** (`execute(..., missing=True)`) sends only
  the calls `<stem>_incomplete` lacks and saves the whole run under the stem
  - `ResultStore.answered(resolved)` reads the incomplete run back as results. It raises
    unless it is the same run: the same spec (the bounds included), built from files with
    the same digests, every saved prompt the one that call would send today
  - the answers kept though flawed stay kept and marked; they are not sent again
  - still incomplete afterwards - stopped again, or calls still unanswered - it is saved as
    `<stem>_incomplete` again, with what both sendings gave, so each try can only add
- **The record says the run was put together**: `"resumed": {answers, saved_at,
  model_resolved}` - how many answers the earlier sending gave, when, and which model
  version it reported. `model_resolved` at the top is the later sending's
  - why: a run completed a week later may hold answers of two model versions; the stem
    cannot say so
- **A completed run takes the incomplete one in**: its archived copies (a CSV and a record per
  incomplete save) move into the run's `archive/` as `<timestamp>_incomplete.csv` and
  `<timestamp>_incomplete_run.json`, and the `_incomplete` folder is removed
  - why: left in place it could be completed a second time, paying for the same calls again
    and replacing answers of the whole run
  - nothing is lost: every answer is in the whole run, and each earlier state is an archived
    pair whose record says `incomplete`
  - `--missing` on a run that is whole already is refused (`FileExistsError`), which also
    covers an incomplete folder left behind by a run that was later sent again in full
- checked: a run stopped twice and then completed has the same `results.json` and
  `results.csv` as the run sent in one go (offline, the fake client)
- not done: `run.py --check` still does not look at an `_incomplete` folder; no live call

## 39. The record says what sending a run took (2026-11-17)

- **Why**: retries select answers. A call whose first answer summed to 1.20 and whose second
  was accepted left no trace, so a model that needs three tries per subject read like one
  that answers cleanly; and how many calls a run cost was not written down at all
- **`sending`** in the record, the totals of one sending:
  - `calls_sent` - every attempt: those that gave no accepted answer, and one per accepted answer
  - `answered` - the calls of this sending that have an answer in the run, kept flawed ones included
  - `refused` - attempts refused, per flaw; an attempt with two flaws is in both
  - `failed`, `rate_limited` - attempts that failed (`CallFailed`) and that met a rate limit
  - `rerun_rounds` - the rounds used; `settings` - `max_workers`, `retries`, `rerun_rounds` as given,
    recorded nowhere before
- **`attempts`** in the record, per call (`"<id>"` or `"<id>:<repeat>"`, as `faulty`): one word
  per attempt that gave no accepted answer, in order - `failed`, `rate_limited`, or the flaw's
  reasons joined by `+`. Only calls that had such an attempt; a clean call is not listed
  - why per call: the totals give the rate, this says which subjects a model stumbles on
  - REJECTED: a column in `results.csv` - its columns are the topic's, read by name and order
- **With `--missing`** both blocks describe the later sending only; `resumed` is now
  `{answers, sendings}`, one entry per earlier sending, oldest first, each with `saved_at`,
  `model_resolved`, `sending` and `attempts`
  - this replaces §38's `{answers, saved_at, model_resolved}`; no saved run has that shape
- `check` prints `calls sent`: `7 for 5 answers; refused: options_sum 2; 0 failed, 0 rate limited; 0 rerun rounds`
- `run_llm_batch` returns a `Batch` (`results`, `flawed`, `attempts`, `skipped`); `BatchStopped.batch` is one
- not counted: a call that stopped the batch, and calls dropped by the stop
- no live call

## 40. An answer read off the token distribution: a framing with a menu, a client that reads tokens (2026-11-17)

- **Why**: a model asked to write nine probabilities writes round ones (§34: few distinct
  answers, a 5 pp lattice). The probability of the answer's first token is continuous and
  comes with one call. It is a different measurement, not a sharper one: a written
  probability is what the model states, a token probability is what it would sample from
  - tried standalone in `CoRe/API_test` (`vote_openai_logprobs.py`, `openai_test.ipynb`)
- **A framing states the answer in one of two ways**, exactly one: `schema` (text of a
  structured answer) or `answer` (a menu of one-token answers). `cs_letters`, short `cs-letters`
  - `answer.tokens`: option id -> token, complete over the topic's options; the menu lists
    them in this order, so a framing with the letters permuted is another file
  - `answer.abstain` `{token, label}`: required exactly when the topic has turnout, listed last
  - `answer.line`: one menu line, `{token}` and `{label}`
  - `Resolved.prompt` puts the menu last: description, question, a line break, the menu
  - why the framing: the menu and the instruction to answer with one token are what the
    model reads beside description and question, and they replace the schema text one to one
  - REJECTED: a new shape (`Token`) - the answer is a distribution over the options, the
    saved fields are those of `Probabilities`, and framing and client already tell the stems apart
  - REJECTED: new fields in `RunSpec` - nothing is declared that a framing or a client does not say
- **Turnout is an entry of the menu** ("did not vote"), not a second call
  - `mass` = the probability on the menu's tokens together; everything else is a share of it
    (`panel/topic/response.py:TokenProbabilityResponse.from_tokens`): `voted` the options' tokens,
    `not_voted` the abstain token, each option its share of `voted` - every group sums to 1
  - a topic without turnout: each option is its token's share of `mass`
  - `mass` is a field of the saved answer (`results.json`), not a column; `check` prints
    `mass on the menu`: the mean and the lowest
  - **what went to no answer token is counted under no entry**
    - REJECTED: adding it to "did not vote" or to "other party" - it is what the model was about
      to say instead (a quote, a word, a refusal), and either entry would grow by it in every
      subject, by an amount that depends on the model and not on the subject
    - REJECTED (the first version, same day): the probabilities as reported, `voted + not_voted`
      = the mass, judged by `sum_low` - a low mass was then a flaw and the call was sent again
- **A token answer is not judged** (`execute`: no review for a `READS_TOKENS` client), so the
  run's `sum_low` / `sum_high` do nothing and nothing is sent twice for its content
  - why: its sums are 1 by construction, and the same prompt gives the same probabilities
  - a failed call (`CallFailed`, a rate limit) is still tried again
  - REJECTED: asking twice (turnout, then party) - twice the calls
  - known: the question as worded presupposes having voted; the abstain entry answers it sideways
- **`OpenAITokens`** (`panel/llm/client_openai_tokens.py`, `READS_TOKENS`): the Responses API with
  `top_logprobs=20`, `include=["message.output_text.logprobs"]`, `max_output_tokens=16`, no `text_format`
  - the answer is read at the first generated token whose candidates include an answer token;
    candidates are matched with decoration and case stripped and added up (`A`, ` A`, `"A`)
  - only the 20 likeliest candidates are reported: an answer below them reads as 0
  - no token probabilities at all is `BatchFailed`; no answer among the candidates `CallFailed`
  - `schema_sent` is None, so the record's `response_schema_sent` is null; `response_schema` is
    the unsent model's; the record's framing carries `answer`, and `schema_text` null
  - the llm package knows no topic: the response class brings `ANSWERS` and `from_tokens`
- **`resolve` refuses** a menu with a client that reads a structured answer, a schema with one
  that reads tokens, and any shape but `Prob` with a menu
- **Non-reasoning models only**: `OpenAITokens` takes no effort, and a reasoning model reports
  no token probabilities unless its reasoning is off. Data preset `sod-letters`; the client is named as a field over
  a model preset: `python run.py -d sod-letters -m nano --client OpenAITokens`, offline `--client FakeTokens`
  - REJECTED: model presets per token client (`nano-tokens`, ...) - a copy of each model preset
    that differs in the client alone; `sod-letters` without such a client is refused and says why
  - later, `TODO.md`: reasoning models with reasoning `none` - no other level makes sense here,
    and `none` is not an effort of ours (§35)
- not verified live: the request as a whole; that `max_output_tokens=16` is accepted and is the
  floor; that the reported probabilities do not depend on the temperature
- in `API_test` one saved output puts 99.95% on `I` ("Jiná strana"): whether a model's
  distribution over letters says anything about the options is the first thing to look at

---

## 41. A dataset whose subjects are wholes is never pooled (2026-11-17)

- **The problem**: a `regions` run asks about fifteen places, and each answer is already that
  place's result. The analyses that read the subjects as a sample of one population made a
  "country" out of them - a plain mean in which Karlovarský kraj counts as much as Středočeský
  and `Česko`, the model's own country answer, is one of the fifteen; the correlations, the map
  and the spread were over fifteen areas under titles that speak of subjects
- **`"aggregate": true`** in `dataset.json` (optional, `Dataset.aggregate`, False by default): a
  subject stands for a population and its answer is that population's result. Only `regions` says so
- **`panel/analysis/saved.py:pools(run, name)`** raises `NotApplicable` for such a dataset; the
  first line of `compute` in `country`, `distance`, `correlation`, `compass`, `spread`
  - `compare.py` loses `shares`, `distances`, `compass` with them, since each calls a run's `compute`
  - `comparison/distances.py` no longer rewords a run's reason into "neither claims nor a reference"
  - still run: `answers` (its mean is over the areas - known, left), `resolution`, `area`, `groups`;
    `areas` of a comparison. `area` is the analysis such a run is for
- **REJECTED for now, the goal in `TODO.md`**: aggregating on purpose - a weight per area (voters,
  not the approximate populations of `regions.csv`), the `Česko` row left out; it would say whether
  the model's regions add up to its country. It would be set up in the framing; how is not discussed
- the tables and figures the five had written for the `regions` runs and the comparisons
  `regions-citizen` / `regions-direct` were removed from `results/` (152 files)

## 42. A text edited under its name makes another run (2026-11-17)

- **The hole**: a spec holds names, the record copies the texts behind them. Sending a run
  again compared specs only, so an edited system prompt under an unchanged framing name
  saved over the earlier run without a word - against the rule that new wording is a new
  file (`CLAUDE.md` §6), which nothing enforced
- **`record.changed(record, today)`** sets the record's `description`, `framing` and
  `question` blocks against today's files and names what differs as `block.key`
  (`framing.instructions`). `record.texts(resolved)` builds the three blocks, for the record
  and for the comparison alike
  - a key the record's block lacks is passed over: it did not exist when the run was saved
    (`framing.answer`, §40). A key it has is compared whole, so schema text that gained a
    line since counts as changed - the run was sent something else
  - a `note` is not copied into a record, so editing one changes nothing
- **Where it is asked**
  - `ResultStore.refuse`: a run of the same spec that read other texts is refused like a run
    of a different spec, before the first call; `overwrite` (`run.py --overwrite`) replaces it
  - `ResultStore.answered`: `--missing` does not complete a run with texts it was not sent with
  - `check`: a `framing` line, `as recorded` or `NOT as recorded: framing.instructions`.
    Only the framing - `check` reads nothing of the dataset, so not the description or the
    question; the refusal covers all three
- on 2026-11-17 every saved run of `election_2021` whose framing has a file reads `as recorded`
- not done: `check` still parses the saved answers with today's framing; a changed label
  therefore fails there before the `framing` line is printed

## 43. A framing for a place, `cs_area`; `grid_sod` worded as the survey words it (2026-11-17)

- **`cs_area`** (short `cs-area`) frames a run whose subject is an area and nothing else, the
  `direct` description of `regions`; the `direct` preset names it
  - why: `cs` speaks of a respondent throughout - "na základě profilu respondenta", "pravděpodobnost,
    že se respondent voleb zúčastnil" - and `direct` sends "Účast a výsledek voleb ... v Praze ...
    budou:". For a place the numbers are shares of voters and of votes
  - built sentence for sentence on `cs` (as reworded in §44): the same opening, the same rule
    about sums, the same schema blocks; class titles `TurnoutShare`, `PartyShare`. The two differ only in
    what is estimated
  - `citizen` keeps `cs`: its prompt is an inhabitant speaking
  - REJECTED: one shorter, more general system prompt for people and places - vaguer for the
    survey and grid runs, and `cs` is already recorded by saved runs
  - the old direct runs were sent with `cs_legacy`, no system prompt; they stay as they are
  - the Czech is unreviewed
- **`grid_sod` forms**: income and education now have forms, the ones of `datasets/sod/forms.json` -
  "do 25.000 Kč", "více než 40.000 Kč" (the level is spelled `Vice`), "mé vzdělání je
  vysokoškolské" instead of "... je vysokoškolské vzdělání". No run had been sent from this
  grid, so its prompt table was re-rendered freely
  - the retired grids keep the levels spliced as they stand: their runs were sent so

## 44. `cs` reworded in place, before any full run (2026-11-17)

- **`cs` was edited under its name**, against the rule that new wording is a new file (§42,
  `CLAUDE.md` §6), by decision: the only runs that recorded it are trials to be sent again
  or discarded - the two Haiku `regions_citizen_cs_neutral_...` runs and the Haiku 4.5
  `sod ... subset=200` sample. Their check now reads `framing NOT as recorded`, and sending
  one again needs `--overwrite`. From here on the rule holds for `cs` again
- **The system prompt**
  - the sum rule is one sentence for every set of probabilities ("Každé rozdělení
    pravděpodobností, které uvedeš, musí mít součet přesně 1.0"), where it used to name
    turnout only and left the parties to the schema text. It holds for both shapes: a
    `Pick` answer has the one set
  - "Analyzuj VŠECHNY poskytnuté informace" is "Vycházej ze všech informací, které o
    respondentovi máš": no capitals for emphasis
  - turnout is asked as both halves, going and not going, as the answer has them
  - "profilu respondenta" is "popisu respondenta": what is sent is a description
- **The schema text says "pokud k volbám šel" with every party probability** (`option`,
  `probabilities.options`, `pick.choice`); formerly only the system prompt did. The analysis
  reads the party shares as given that the subject voted, and a subject sure to stay home
  should still be given a preference
- **`cs_area` follows**: the same four changes where they apply, shares for probabilities
- left as it is: the English class titles (`TurnoutProbability`, `PartyProbability`) - they
  become keys of the schema, and a non-ASCII key is a risk with a provider's strict mode for
  no gain
- not done: `en` is a translation of the former `cs` and no longer matches it; `cs_grid` and
  `cs_legacy` are what old runs were sent and are not touched
- **`cs_individual`** (short `cs-individual`) is `cs` with one rule more: "Odhaduj právě tohoto
  respondenta, ne celkový výsledek voleb." Everything else is `cs` word for word
  - why: the sample's answers hugged the national shares (the highest option averaged 0.28)
  - why a framing of its own and not a line in `cs`: the rule steers the result, so what it
    does has to be seen - a run with it beside a run with `cs`
- **`election_2025` has both**, `cs` and `cs_individual`, the 2021 texts with the year changed.
  Nothing had been run on that topic. No `cs_area` there: the `direct` description names 2021
- REJECTED for now: the system prompt as a name of its own beside the question, so that a
  variant need not copy the labels and the schema text
  - the system prompt and the schema text have to speak of the same thing - a respondent's
    probabilities or an area's shares - and one file keeps them together
  - it adds a field to the spec and a part to the stem, days before the full runs
  - worth it once a third variant of the system prompt alone appears; `cs_individual` is the first
- the Czech is unreviewed

## 45. A digest is of the file's text, not of its bytes (2026-11-17)

- **Seen**: after the checkout of `main`, runs refused to pack and four tests of
  `tests/test_pack.py` failed with "built from other files than today's". Nothing had changed
  but line endings: `core.autocrlf` is `true` on this machine, so a tracked file is LF in git
  and CRLF on disk after a checkout, and the record's digest was the sha256 of the bytes
- **The saved records were mixed**, which is why keeping the files byte for byte could not
  work: every values and prompt table had been hashed with CRLF (prepare writes them so on
  Windows), the forms of the grids and `regions` with LF, the forms of `sod` and everything
  recorded since the checkout with CRLF. No one state of the files on disk matched them all
- **`panel/digest.py:content_sha256`**: the sha256 of the file with every line ending as a
  bare line feed.
  `resolve` records it, `load_run`, `pack` and `--missing` compare with it, and prepare's
  guard (§20) looks tables up by it and calls a table changed only when its text changed
  - the record's keys stay `values_sha256`, `prompts_sha256`, `forms_sha256`
  - `file_sha256` is the plain one, used for a pack's `MANIFEST.json`: a pack's files are
    copies that must not change at all. `built_from` in the manifest carries the record's
    digests, so it is checked with `content_sha256`
- **76 digests in the 31 saved run records were rewritten** (13 were already the text's). A
  digest was rewritten only where it was exactly the sha256 of today's file with CRLF line
  endings, which proves the content is the one the run was built from; none was left
  unexplained. Each record has a `restamped` block: the date, the reason, and the digests as
  they were. The copies in each run's `archive/` are as they were; so are packs written earlier
- REJECTED: a `.gitattributes` keeping these files byte for byte, the first thing tried
  - it needs one byte form per file, and the records disagree on which
  - it leaves a trap: an editor that saves `forms.json` with other line endings orphans every
    run built from it, silently
- REJECTED: writing prepared tables with a fixed line ending - not needed once the digest
  does not see it; the tables stay as the system writes them
- `tests/test_kernel.py` named its grid runs by a prefix (`grid-simple_`), which two runs match
  since the Haiku 4.5 grid run was saved; it names the full stems now
