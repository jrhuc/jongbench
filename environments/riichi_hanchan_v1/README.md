# riichi-hanchan-v1

Four models play a full Tenhou-rules hanchan against each other. Multi-agent
verifiers `Env`: one episode is one hanchan refereed by the vendored `libriichi`
arena, each seat a live interaction, so every trace is a real rollout.

Reward is placement — what riichi is actually played for — and genuinely zero-sum:
the four placements are a permutation of 1–4, so the rewards always sum to 2.0
however the models play.

This is the expensive way to measure a model (~1,000 calls per episode, and every
seat sees a board the other three steered). For a cheap, separable, byte-identical
comparison use `riichi-decision-v1`, which grades single positions against Mortal.

## Agents

`seat0`..`seat3`, each a `vf.AgentConfig`. A seat holds one conversation per kyoku:
an opening board, then per-decision deltas — the same prompt path `jongbench run`
uses, including the furo toggle and invalid-reply retries.

## Run

From the repo root (the package imports `jongbench`, which runs from the checkout):

```console
$ PYTHONPATH=".:environments/riichi_hanchan_v1" .venv/bin/eval riichi_hanchan_v1 \
    --env.seat0.model anthropic/claude-sonnet-5 \
    --env.seat1.model openai/gpt-5.2 \
    --env.seat2.model google/gemini-3-pro \
    --env.seat3.model deepseek/deepseek-v4 \
    --env.log-dir episodes \
    --client.base-url https://openrouter.ai/api/v1 \
    --client.api-key-var OPENROUTER_API_KEY \
    -n 1 --no-push
```

`eval` is the verifiers v1 CLI — call it by path, the shell builtin shadows the
name. `--no-push` keeps the run local instead of uploading it to the Prime
platform. A seat with no pinned model plays the run's `-m` model — the policy
under evaluation.

## Config

| key           | default | meaning                                               |
|---------------|---------|-------------------------------------------------------|
| `state_hints` | `true`  | rule-derived shanten/waits/furiten in prompts          |
| `log_dir`     | `None`  | persist each episode as a jongbench run dir (below)    |

## Rewards and metrics

Per seat trace:

- `placement` (reward) — 1st → 1.0, 4th → 0.0, zero-sum across the table.
- `final_score`, `decisions`, `fallbacks`, `calls_declined` (metrics).
- `trace.info["hanchan"]` — seat, placement, score, seed, seat order.

## Post-hoc Mortal grading

With `log_dir` set, each episode lands as `hanchan-<idx>/` containing the mjai log,
per-seat decision logs and a `config.json` — a normal jongbench run dir:

```console
$ jongbench review episodes/hanchan-00000     # Mortal rating per seat + report.html
$ jongbench reasoning episodes/hanchan-00000  # each seat's reasoning joined to Mortal's verdict
```

So placement (the env reward) and decision quality (Mortal's grading) come from the
same rollout without spending a second one.

## Notes

- Hard to reward-hack: placement is computed by the Rust referee from final scores,
  every action is validated against the legal menu, and post-hoc grading uses a
  frozen external model. There is no tool surface or environment state to game.
- One hanchan per episode, deliberately: a seat holds one conversation, and driving
  several games through one engine would interleave their turns into it.
- The taskset is an infinite seeded generator; episode `i` uses seed `20260000 + i`,
  so a run is reproducible from its indices.
