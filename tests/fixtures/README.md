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
| `schemas/<topic>.json` | `model_json_schema()` of both response classes; for `election_2021_legacy` the frozen copy from `results/election_2021_legacy/config.json`, not today's generation | `test_schemas.py` |
| `legacy_aggregates.json` | country-level aggregation of every legacy run, from its plain and its normalized CSV, for all four `weighted` x `normalized` combinations | `test_legacy_results.py` |
| `legacy_reparse.json` | per legacy run: response model, raw-JSON shape (`bare` or `{prompt, result}`) and a digest of the reparsed frame; `null` for the four `VPick` runs, which have no JSON | `test_legacy_results.py` |

- `prompt_tables.json` keys were renamed in phase 3, digests unchanged: `self_description_v1`
  became `sod/self_v1` and `self_description_v2` became `sod/self_v2` when the templates moved
  to `config/descriptions/` and the tables under `data/prompts/<dataset>/`. The digest covers
  `id<TAB>text` lines only, so the column rename does not touch it.
- `grid/self_v1` was added in phase 3 from the grid's own pre-overhaul render
  (`grid_description_v1` in `data/grid_income3_sod.csv`, 288 rows). The shared `self_v1`
  template has to reproduce it, which is what retires the grid's separate template.
- `schemas/election_2021_legacy.json` differs from what the code generates today: the topic
  refactor replaced the election-specific schema text with generic wording. Its test is
  `xfail(strict=True)` until phase 4 framings carry that text again.
