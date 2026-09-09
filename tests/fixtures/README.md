# Phase 0 baseline fixtures

What the code produced on `e615307`, before the architecture overhaul touched it. The tests
under `tests/` recompute each of these and compare; a difference is drift, not a new truth.

- taken once, by `python -m tests.baseline` from the repo root; `manifest.json` says when and
  from which commit. Retaking needs `--overwrite` and a commit message that says why.
- the inputs behind them (`data/clean_data.csv`, `results/`) are gitignored; the tests that
  need them skip, naming the path, on a machine that lacks them.

| file | holds | recomputed by |
|---|---|---|
| `prompt_tables.json` | rows and digest of each stored `data/prompts/<variant>.csv`, taken after re-rendering reproduced it 3880 / 3880 | `test_prompts.py` |
| `schemas/<topic>.json` | `model_json_schema()` of both response classes; for `election_2021_legacy` the frozen copy from `results/election_2021_legacy/config.json`, not today's generation | `test_schemas.py` |
| `legacy_aggregates.json` | country-level aggregation of every legacy run, from its plain and its normalized CSV, for all four `weighted` x `normalized` combinations | `test_legacy_results.py` |
| `legacy_reparse.json` | per legacy run: response model, raw-JSON shape (`bare` or `{prompt, result}`) and a digest of the reparsed frame; `null` for the four `VPick` runs, which have no JSON | `test_legacy_results.py` |

- `schemas/election_2021_legacy.json` differs from what the code generates today: the topic
  refactor replaced the election-specific schema text with generic wording. Its test is
  `xfail(strict=True)` until phase 4 framings carry that text again.
