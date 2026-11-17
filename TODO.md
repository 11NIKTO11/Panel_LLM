# TODO - ideas not acted on

Things worth doing that nobody decided to do yet. What was decided, and what was rejected, is
in `DECISIONS.md`; an entry leaves this file when it is done or dropped, with a line there.

## Runs whose subjects are places (`regions`)

- **Goal: aggregate them on purpose** - a country-level result, correlations, a map and distances that
  mean something for a run that asks about places
  - today they are skipped: `regions` declares `"aggregate": true` and `country`, `distance`,
    `correlation`, `compass`, `spread` raise `NotApplicable` (`panel/analysis/saved.py:pools`), which
    takes `shares`, `distances`, `compass` of `compare.py` with them
  - why: a subject there is an area and its answer is already that area's result; pooling read the
    fifteen as a sample of voters - a plain mean in which Karlovarský kraj counts as much as
    Středočeský, and `Česko`, the model's own country answer, is one of the fifteen
  - what means something meanwhile: `area` (each area against its own result), the `Česko` row alone
  - an aggregate needs a weight per area (voters, not the approximate populations of `regions.csv`)
    and the `Česko` row left out; it then says whether the model's regions add up to its country
- **Where it would be set up: the framing** (Martin, 2026-11-17) - what an answer is the result of, and
  so how answers combine, is said where the question is framed. How, structurally, is NOT discussed yet
- `answers` still averages over the areas; `effects` skips them for being too few
