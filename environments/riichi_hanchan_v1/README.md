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

`seat0`..`seat3`, each a `vf.AgentConfig`. A seat holds one conversation — and one
verifiers interaction — per kyoku: an opening board, then per-decision deltas, the
same prompt path `jongbench run` uses, including the furo toggle and invalid-reply
retries. Finished kyoku never ride along in later requests (dragging them measured
at ~3x the input tokens), and every kyoku trace is a bounded training sample
carrying the seat's final placement.

## Run

From the repo root (the package imports `jongbench`, which runs from the checkout):

```console
$ PYTHONPATH="$PWD:$PWD/environments/riichi_hanchan_v1" .venv/bin/eval riichi_hanchan_v1 \
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

| key                   | default | meaning                                                       |
|-----------------------|---------|---------------------------------------------------------------|
| `state_hints`         | `true`  | rule-derived shanten/waits/furiten in prompts                  |
| `auto_pass_reactions` | `false` | pass pure chi/pon/kan reactions without a model call (~15% of decisions; the seats then never call on discards) |
| `tools`               | `false` | board-query tools instead of inline hints (below)              |
| `log_dir`             | `None`  | persist each episode as a jongbench run dir (below)            |

## Tool-using seats

`--env.tools true` inverts the prompt design: turns stay minimal — delta and menu, no
inline state hints — and each seat gets MCP tools it calls when it actually wants
information. `board()` re-renders the full table, `discards(player)` one discard row,
`waits()` the seat's own shape, `simulate(tile)` the shape after a legal discard. On
top of those, `note(text)`/`notes()` is a private scratchpad that survives kyoku
resets — the one thing the per-kyoku design otherwise throws away — so opponent reads
and standings plans can outlive the board. `notes_saved` joins the metrics.

The tools answer from exactly the information the hints path may use — rule-derived
only, nothing from Mortal — precomputed at each decision point and served from the
rollout's state channel. This measures something different on purpose: information
seeking and memory management become part of the skill, and a tool call is a full model
roundtrip, so an over-querying seat costs more than always-on hints would have.

The toolset launches as its own process from a different working directory, so tools
mode needs the `PYTHONPATH` entries to be absolute (`$PWD` as above, not `.`).

## Rewards and metrics

A seat produces one trace per kyoku; every one of them carries the same seat-level
signals:

- `placement` (reward) — 1st → 1.0, 4th → 0.0. A seat's mean reward is exactly its
  placement reward, and the four seats' rewards sum to 2.0 — zero-sum however the
  models play.
- `final_score`, `decisions`, `fallbacks`, `calls_declined` (metrics, seat totals).
- `trace.info["hanchan"]` — seat, placement, score, seed, seat order, kyoku index.

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
