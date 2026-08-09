# riichi-decision-v1

One riichi mahjong decision per task, graded by Mortal (deep-RL, ~expert strength).

Each task is a real board position from a played hanchan, rendered from one seat's
point of view with its legal actions numbered. The model answers `{"choice": N}`.
Reward is Mortal's normalised Q-advantage for the option chosen: 1.0 for Mortal's own
choice, 0.0 for its worst, linear between. That is the per-decision term of the rating
jongbench reports for a full game, so this taskset and a played hanchan measure the
same quantity — here with byte-identical prompts on identical boards across models,
at one call per graded decision instead of ~1,000 per hanchan.

## Tasks

A 128-position sample bank from Mortal self-play ships with the package and is the
default, so the taskset runs out of the box. Reference points on it:

| policy                | reward | match  |
|-----------------------|--------|--------|
| uniform random        | 0.367  | 18.9%  |
| always first option   | 0.341  | —      |
| Mortal's own choice   | 1.000  | 100%   |

(mean 8.8 legal options per position; `jongbench positions` prints the same
baselines for any bank it builds)

For a serious ranking, build a bigger bank from Mortal self-play or grade an
existing mjai log — banks are one `jongbench.positions.Position` JSON per line,
plain or gzipped:

```console
$ jongbench positions --out bank.jsonl --games 4          # ~600 positions per game
$ jongbench positions --out bank.jsonl --from-log runs/<stamp>/logs/g0.json.gz
```

Grading is baked in at build time, so evaluation needs neither a runtime nor the
Mortal checkpoint — scoring is pure trace.

## Config

| key           | default             | meaning                                              |
|---------------|---------------------|------------------------------------------------------|
| `bank`        | shipped sample bank | path to a position bank (`.jsonl` or `.jsonl.gz`)     |
| `state_hints` | `true`              | include rule-derived shanten/waits/furiten in prompts |

## Rewards and metrics

- `q_advantage` (reward, weight 1.0) — Mortal's opinion of the chosen action.
  An unparseable or out-of-range reply scores 0.0: failing to answer in the required
  form is a real failure at this task, not a broken sample.
- `matched_mortal` (metric) — chose exactly what Mortal would have.
- `answered` (metric) — reply parsed to a legal choice; separates disagreement
  from malformed output when reading a low score.

## Notes

- verifiers does not force Prime inference: point `client.base_url` and
  `client.api_key_var` at any OpenAI-compatible endpoint (jongbench itself uses
  OpenRouter).
- A bank drawn from random self-play is a distribution of bad boards. For ranking
  strong models, generate source games from Mortal self-play (the default) or real
  logs.
