# Baseline fixtures

What the code produced before the architecture overhaul touched it. The tests under `tests/`
recompute each of these and compare; a difference is drift, not a new truth.

- taken by `python -m tests.baseline` from the repo root, on `e615307` (phase 0); `manifest.json`
  says when and from which commit. Retaking needs `--overwrite` and a commit message that says
  why.
- the inputs behind them (`data/profiles/sod.csv`, `results/`) are gitignored; the tests that
  need them skip, naming the path, on a machine that lacks them.

| file | holds | recomputed by |
|---|---|---|
| `prompt_tables.json` | rows and digest of each stored prompt table, keyed `<dataset>/<description>`, taken after re-rendering reproduced it | `test_prompts.py` |
| `schemas/<topic>/<framing>.json` | `model_json_schema()` of both response classes under that framing; for `election_2021_legacy` the frozen copy from `results/election_2021_legacy/config.json`, not a generation | `test_schemas.py` |
| `legacy_aggregates.json` | country-level aggregation of every legacy run, from its plain and its normalized CSV, for all four `weighted` x `normalized` combinations, keyed by the native names the legacy CSVs use | `test_legacy_results.py` |
| `legacy_reparse.json` | per legacy run: response model, raw-JSON shape (`bare` or `{prompt, result}`) and a digest of the reparsed frame; `null` for the four `VPick` runs, which have no JSON | `test_legacy_results.py` |

- `prompt_tables.json` keys were renamed in phase 3, digests unchanged: `self_description_v1`
  became `sod/self_v1` and `self_description_v2` became `sod/self_v2` when the templates moved
  to `config/descriptions/` and the tables under `data/prompts/<dataset>/`. The digest covers
  `id<TAB>text` lines only, so the column rename does not touch it.
- `grid/self_v1` was added in phase 3 from the grid's own pre-overhaul render
  (`grid_description_v1` in `data/grid_income3_sod.csv`, 288 rows). The shared `self_v1`
  template has to reproduce it, which is what retires the grid's separate template.
- the schema snapshots moved from `schemas/<topic>.json` to `schemas/<topic>/cs.json` in
  phase 4, when a topic gained one framing per language. Three are unchanged; the legacy one
  is now reproduced by the code and its test no longer expects failure.
- `schemas/media_fees_2026/cs.json` was regenerated in phase 4, deliberately: the topic's two
  options were shown to the model as the colleague's English ids (`abolish_fees_state_budget`,
  `keep_fees`) inside a Czech prompt; they are Czech phrases now. The topic has never been run.
- the four non-legacy schema snapshots were regenerated on 2026-11-11, deliberately
  (`DECISIONS.md` §8): the classes are `Probabilities` / `Pick`, the list is `options`, and
  the schema text speaks the topic's words. `schemas/election_2021_legacy/cs.json` is left as
  the frozen copy; the legacy framing no longer reproduces it and its two tests fail until
  the legacy migration.
- `schemas/election_2021/en.json` was added in phase 8 as the first snapshot of the English
  framing: English labels for the parties that translate, `ANO 2011` kept, English schema text.
  Built offline; no run has been sent with it.
