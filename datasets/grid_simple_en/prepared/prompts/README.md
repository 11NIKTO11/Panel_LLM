# Prompts of `grid_simple_en`: unknown

There is no prompt table here because the colleague's prompts were not delivered.

- **What is missing:** the prompt sent per profile, the system prompt, the question and the
  response schema. Their bundle (`CoRe/_grid_separate/colleague/`) holds a report, figures and
  two tables of per-profile means - no prompt column and no raw answers.
- **What stands in for it:** the stored description `delivered`
  (`datasets/grid_simple_en/descriptions/delivered.json`), which has no table. Their two runs
  carry a `delivered` block in the record and can be read, not resolved or sent again
  (`DECISIONS.md` section 15.1).
- **When the prompts arrive:** `delivered.csv` goes here, one row per profile of `values.csv`,
  and the system prompt and schema become a framing file `colleague`.
