# riichi-decision-v1

One riichi mahjong decision per task, graded by Mortal (deep-RL, ~expert strength).

Each task is a real board position from a played hanchan, rendered from one seat's
point of view with its legal actions numbered. The model answers `{"choice": N}`.
Reward is Mortal's normalised Q-advantage for the option chosen: 1.0 for Mortal's own
choice, 0.0 for its worst, linear between. This is the per-decision term of the rating
jongbench reports for a full game, so this taskset and a played hanchan measure the
same quantity. The difference is that here every model sees byte-identical prompts on
identical boards, at one call per graded decision instead of about 1,000 per hanchan.

## Tasks

A 128-position sample bank from Mortal self-play ships with the package and is the
default, so the taskset runs out of the box. Reference points on it:

| policy                                | reward | match  | answered | cost    |
|---------------------------------------|--------|--------|----------|---------|
| always first option                   | 0.341  | —      | —        | —       |
| uniform random                        | 0.367  | 18.9%  | —        | —       |
| openai/gpt-5.6-luna                   | 0.776  | 51.6%  | 100%     | $0.075  |
| deepseek/deepseek-v4-flash-0731       | 0.834  | 63.8%  | 93.7%    | $0.428  |
| google/gemini-3.5-flash-lite          | 0.844  | 57.0%  | 100%     | $0.279  |
| Mortal's own choice                   | 1.000  | 100%   | 100%     | —       |

All three at `--sampling.reasoning-effort low`, one pass over the 128 positions,
cost as metered by OpenRouter. deepseek agrees with Mortal most often and still
scores below gemini: 6% of its replies never arrive, and an unanswered task scores
0. mean 8.8 legal options per position; `jongbench positions` prints the
non-model baselines for any bank it builds.

Two earlier numbers (gpt-5-mini 0.727, deepseek 0.649) are withdrawn, not
restated. They were measured against a board renderer that pasted the whole match
into every prompt, and against a 16k completion cap that truncated a third of
deepseek's replies mid-reasoning. Both are fixed. The caps in `engines.py` are now
sized to catch runaway loops rather than to bound normal output.

State hints are on by default and account for 23% of a full-game prompt budget.
They were worth about +0.09 reward to gpt-5-mini on the pre-fix render, so they
stay the default. The size of that gap has not been remeasured.

For a serious ranking, build a bigger bank from Mortal self-play or grade an
existing mjai log — banks are one `jongbench.positions.Position` JSON per line,
plain or gzipped:

```console
$ jongbench positions --out bank.jsonl --games 4          # ~600 positions per game
$ jongbench positions --out bank.jsonl --from-log runs/<stamp>/logs/g0.json.gz
```

Grading is baked in at build time, so evaluation needs neither a runtime nor the
Mortal checkpoint — scoring is pure trace.

## Run

From the repo root (the package imports `jongbench`, which runs from the checkout):

```console
$ PYTHONPATH=".:environments/riichi_decision_v1" .venv/bin/eval riichi_decision_v1 \
    -m anthropic/claude-sonnet-5 \
    --client.base-url https://openrouter.ai/api/v1 \
    --client.api-key-var OPENROUTER_API_KEY \
    --no-push
```

`eval` is the verifiers v1 CLI — call it by path, the shell builtin shadows the
name. `--no-push` keeps the run local instead of uploading it to the Prime
platform. The package bundles verifiers' plain chat harness as its default, so
no harness or runtime flags are needed.

The offline integration check (no model, no key) is:

```console
$ PYTHONPATH=".:environments/riichi_decision_v1" .venv/bin/validate riichi_decision_v1 \
    --runtime.type subprocess
```

## Config

| key           | default             | meaning                                              |
|---------------|---------------------|------------------------------------------------------|
| `bank`        | shipped sample bank | path to a position bank (`.jsonl` or `.jsonl.gz`)     |
| `state_hints` | `true`              | include rule-derived shanten/waits/furiten in prompts |

## Rewards and metrics

- `q_advantage` (reward, weight 1.0) — Mortal's rating of the chosen action. An
  unparseable or out-of-range reply scores 0.0; answering in the required form is
  part of the task.
- `matched_mortal` (metric) — the choice was exactly Mortal's.
- `answered` (metric) — the reply parsed to a legal choice. Use it to separate
  disagreement from malformed output when reading a low score.

## Notes

- verifiers does not force Prime inference: point `client.base_url` and
  `client.api_key_var` at any OpenAI-compatible endpoint (jongbench itself uses
  OpenRouter).
- A bank drawn from random self-play contains mostly poor boards. For ranking
  strong models, generate source games from Mortal self-play (the default) or from
  real logs.
