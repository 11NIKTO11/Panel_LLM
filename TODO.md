# TODO - ideas not acted on

Things worth doing that nobody decided to do yet. What was decided, and what was rejected, is
in `DECISIONS.md`; an entry leaves this file when it is done or dropped, with a line there.

## Token probabilities (`DECISIONS.md` §40)

- **Reasoning models, with reasoning `none` only**
  - no other level makes sense: once a model reasons, the answer token is conditioned on the
    reasoning and its probability is no longer the model's first reading of the subject
  - in the way: `none` is not an effort here (`--effort none` means "set no effort", §35) and
    `OpenAITokens` takes no effort; it needs a spelling of its own or a client that always sends it
  - `CoRe/API_test/openai_test_new.py` does it for `gpt-5.4`
- **Do letters measure the options at all**
  - one output in `API_test` puts 99.95% on `I`; a framing with the letters permuted
    (`cs_letters_*`, another order in `answer.tokens`) tells the option from its letter and place
  - the "random respondent of this profile" wording of `openai_test.ipynb`, as a framing
- **Answers of several tokens** (party acronyms, full names): match on the first token; `SPD`
  and `SPOLU` may share it
- **A low mass on the menu** is only printed by `check`; nothing excludes or marks such an answer
- **A second call for turnout** instead of the abstain entry, if the entry turns out to bend
  the party shares (rejected for now: twice the calls)
- **Other providers**: OpenRouter forwards `logprobs` for some routes (`API_test/vote_logprobs.py`)
- verify live: `max_output_tokens=16`, and that the temperature does not change the reported probabilities

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
