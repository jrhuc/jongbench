# riichi-hanchan-v1

Four models play a full Tenhou-rules hanchan against each other. Multi-agent
verifiers `Env`: one episode is one hanchan refereed by the vendored `libriichi`
arena, each seat a live interaction, so every trace is a real rollout.

Reward is placement. The four placements are a permutation of 1–4, so the rewards
sum to 2.0 whatever the models do; the episode is zero-sum.

This is the expensive way to measure a model: about 1,000 calls per episode, and
every seat sees a board the other three steered. For a cheaper comparison on
byte-identical prompts, use `riichi-decision-v1`, which grades single positions
against Mortal.

## Agents

`seat0`..`seat3`, each a `vf.AgentConfig`. A seat holds one conversation, and one
verifiers interaction, per kyoku: an opening board, then per-decision deltas, on the
same prompt path `jongbench run` uses, including the furo toggle and invalid-reply
retries. Finished kyoku are not carried into later requests; dragging them measured
at about 3x the input tokens. Each kyoku trace is a bounded training sample that
carries the seat's final placement.

## Run

Install the package with the command shown on its Environments Hub page, then run
one episode at a time. The package pins its `jongbench` engine dependency to an
immutable Git revision.

```console
$ .venv/bin/eval riichi_hanchan_v1 \
    --env.seat0.model anthropic/claude-sonnet-5 \
    --env.seat1.model openai/gpt-5.2 \
    --env.seat2.model google/gemini-3-pro \
    --env.seat3.model deepseek/deepseek-v4 \
    --env.log-dir episodes \
    --client.base-url https://openrouter.ai/api/v1 \
    --client.api-key-var OPENROUTER_API_KEY \
    -n 1 --no-push
```

For local development from this repository:

```console
$ uv run --with verifiers==0.3.0 --with ./environments/riichi_hanchan_v1 \
    eval riichi_hanchan_v1 -n 1 --no-push
```

`eval` is the Verifiers v1 CLI; calling its virtual-environment path avoids the
POSIX shell builtin of the same name. `--no-push` keeps the run local instead of
uploading it to Prime. A seat with no pinned model plays the run's `-m` model—the
policy under evaluation.

### Metered cost

Cost is the provider's own number, carried on the response's usage and recorded per
decision, per trace and per leaderboard row. Nothing is reconstructed from a price
table. OpenRouter reports it on ordinary completions without being asked (1051/1053
calls in the rotated batch, 635/641 in the tools batch); the missing ones are
responses that failed upstream. A provider that meters nothing leaves the column
`n/a` rather than `$0.00`.

## Config

| key                   | default | meaning                                                       |
|-----------------------|---------|---------------------------------------------------------------|
| `state_hints`         | `true`  | rule-derived shanten/waits/furiten in prompts                  |
| `auto_pass_reactions` | `false` | pass pure chi/pon/kan reactions without a model call (~15% of decisions; the seats then never call on discards) |
| `tools`               | `false` | board-query tools instead of inline hints (below)              |
| `max_tool_calls`      | `32`    | tool calls one decision may spend before the seat must commit (`0` lifts the cap) |
| `seat_rotation`       | `false` | episode *i* seats `seat0` at table position *i*, so a batch of 4 gives every agent each position (below) |
| `log_dir`             | `None`  | persist each episode as a jongbench run dir (below)            |
| `weights`             | `auto`  | verified, cached Mortal checkpoint for a `mortal` control seat (below) |

## Mortal as a control seat

Setting a seat's model to the bare spec `mortal` seats the Mortal NN itself:

```console
    --env.seat3.model mortal
```

The control seat plays deterministically and locally: no API calls, no
interactions, no traces, and an empty decision log. It anchors the table. The
LLM seats' placements are measured against a known-strength opponent instead of
only each other, and `jongbench review` grades everyone with the same model that
is sitting at the table. With a control seat in play the 2.0 reward sum no longer
holds, because the control absorbs whatever placement it wins.

Crash recovery still works. The journal records only the bridged seats, and the
control seat recomputes its choices live on replay, reproducing the identical
game.

## Seat rotation

Table position is not neutral: the dealer wins more, and a fixed seating measures a
model's placements from one chair. `--env.seat-rotation true` moves agent *i* to
table position `(i + episode) % 4`, so a four-episode batch sits every agent in every
chair exactly once and the position cancels out of the pooled result.

Rewards, decision logs and `trace.info["hanchan"]` follow the agent, not the chair:
`seat0`'s reward is `seat0`'s placement wherever it sat, and `table_position` records
where that was. Each episode's `config.json` names the `rotation` it used and the
resulting `table`, so a run dir still reads as a normal jongbench game.

## Tool-using seats

`--env.tools true` inverts the prompt design. Turns stay minimal (delta and menu, no
inline state hints) and each seat gets MCP tools it calls when it wants information.
`board()` re-renders the full table, `discards(player)` one discard row, `waits()` the
seat's own shape, `simulate(tile)` the shape after a legal discard. `note(text)` and
`notes()` are a private scratchpad that survives kyoku resets, so opponent reads and
standings plans can outlive the board. The per-kyoku design otherwise discards them.

Tools mode adds four metrics: `notes_saved`, `tool_calls`, `tool_turns` (decisions
where the seat queried anything at all) and `budget_spent` (decisions that used up
their whole tool budget). `trace.info["hanchan"]["tools"]` holds a per-tool breakdown.
Together they measure how much a seat pays for information alongside how well it
plays.

`--env.max-tool-calls` (default 32, `0` lifts the cap) bounds one decision. Past it
every tool answers "budget spent". If the seat ignores that result and queries again,
the env stops that interaction, plays the engine's safe fallback, and opens a fresh
interaction at the next decision. This hard edge matters because every tool turn
resends the decision's whole conversation: the first live tools run burned 2.5M input
tokens on one discard before it was killed. The remaining budget survives engine
retries and resets only at the next decision. A normal interaction still spans one
kyoku; the forced fallback can split a pathological kyoku into another trace rather
than aborting the hand for all four players.

The tools answer from the same information the hints path may use: rule-derived only,
nothing from Mortal, precomputed at each decision point and served from the rollout's
state channel. Tools mode measures something different from hints mode. Information
seeking and memory management become part of the task, and a tool call is a full model
roundtrip, so an over-querying seat costs more than always-on hints would have.

The toolset launches as its own process from a different working directory, so tools
mode needs the `PYTHONPATH` entries to be absolute (`$PWD` as above, not `.`).

## Rewards and metrics

A seat normally produces one trace per kyoku; a forced tool-budget fallback may split
that kyoku so the next decision can continue. Every trace carries the same seat-level
signals:

- `placement` (reward) — 1st → 1.0, 4th → 0.0. A seat's mean reward is exactly its
  placement reward, and the four seats' rewards sum to 2.0.
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

## Measured

A four-episode flash-tier batch (2026-08-09; seeds 20260000–3, auto-pass-reactions
on, reasoning effort low, 16k max completion tokens — $2.73 of API for all four
hanchan, reviewed post-hoc as above):

| model                           | placements | avg  | Mortal rating | match | cost  |
|---------------------------------|------------|------|---------------|-------|-------|
| qwen/qwen3.7-flash              | 1 3 1 3    | 2.00 | 69.2          | 54.5% | $0.32 |
| google/gemini-3.5-flash-lite    | 3 2 3 1    | 2.25 | 65.7          | 50.3% | $0.96 |
| deepseek/deepseek-v4-flash-0731 | 2 1 4 4    | 2.75 | 73.5          | 59.3% | $1.30 |
| openai/gpt-5.6-luna             | 4 4 2 2    | 3.00 | 63.4          | 51.7% | $0.15 |

The two signals disagree at this sample size, which is why both are reported. Mortal
grades deepseek's decisions best in every game, but zero-sum placement over four
hanchan is noisy enough to leave it third; the per-decision rating converges orders
of magnitude faster than the outcome. Spend tracks reasoning depth rather than rank.
At effort `low` luna answered in about 190 completion tokens per decision while
deepseek spent a median of about 6.8k, and produced the batch's only fallbacks: 10
decisions truncated at the 16k cap.

### Against a Mortal control, rotated

Same seeds and settings, but two of the four chairs are `mortal` and
`seat_rotation` is on, so each agent played every table position exactly once
(2026-08-11; $1.04 of API as metered by the provider, no upstream errors, no
fallbacks):

| model                        | placements | avg  | Mortal rating | match | in/decision | out/decision | cost  |
|------------------------------|------------|------|---------------|-------|-------------|--------------|-------|
| mortal (2 seats)             | 4 1 1 2 / 2 2 2 4 | 2.25 | 100.0 | 100.0% | — | — | — |
| google/gemini-3.5-flash-lite | 1 3 3 3    | 2.50 | 70.1          | 58.5% | 4.6k        | 255          | $0.90 |
| openai/gpt-5.6-luna          | 3 4 4 1    | 3.00 | 68.5          | 52.5% | 5.6k        | 211          | $0.13 |

The control finished mid-table on placement (2.25) against two flash-tier models,
so the LLM seats are measured against a known strength rather than only each
other. Both models' Mortal ratings came in about 4 points above the all-LLM batch
on the same seeds, which is the effect of a stronger, more orthodox table on the
boards a seat has to read. Placement ordering was unchanged. `jongbench
leaderboard runs/<batch>/ --review` pools the episodes by spec and reports the
`table_positions` each spec occupied, so a rotated batch reads as one row per
model with the chair effect averaged out.

### Tools versus hints, one seed

Seed 20260000 replayed four ways with the same two agents and two `mortal` chairs
(2026-08-11 and 12, effort low, auto-pass on): hints inline, hints off with no tools, tools
at a 12-call budget, and tools at 4. Per-seat totals, de-duped across the seat's
kyoku traces:

| arm         | seat                         | decisions | tool calls | queried | budget out | in/dec | out/dec | cost    |
|-------------|------------------------------|-----------|------------|---------|------------|--------|---------|---------|
| hints, on   | openai/gpt-5.6-luna          | 107       | —          | —       | —          | 675    | 208     | $0.028  |
| hints, on   | google/gemini-3.5-flash-lite | 97        | —          | —       | —          | 3,391  | 279     | $0.170  |
| hints, off  | openai/gpt-5.6-luna          | 110       | —          | —       | —          | 576    | 199     | $0.025  |
| hints, off  | google/gemini-3.5-flash-lite | 105       | —          | —       | —          | 3,507  | 386     | $0.212  |
| tools, 12   | openai/gpt-5.6-luna          | 131       | 173        | 23.7%   | 1.5%       | 631    | 208     | $0.035  |
| tools, 12   | google/gemini-3.5-flash-lite | 128       | 350        | 83.6%   | 3.1%       | 19,281 | 69      | $0.763  |
| tools, 4    | openai/gpt-5.6-luna          | 127       | 91         | 18.9%   | 17.3%      | 647    | 222     | $0.034  |
| tools, 4    | google/gemini-3.5-flash-lite | 126       | 232        | 84.9%   | 24.6%      | 13,476 | 59      | $0.528  |

The games diverge after the first differing choice, so the decision counts differ and
placements are not comparable across arms. What is comparable is what each seat did
with the same offer of information, and the two models answer it differently.

Turning hints off saves less than the 23% of prompt tokens they occupy, and only for
one of the two seats. Luna's input fell 675 → 576 per decision, a 15% saving. Gemini's
rose 3,391 → 3,507, because its output per decision grew 279 → 386 and its own replies
re-enter the per-kyoku transcript. Removing hints made it reason more in tokens, and
that cost more than the hints did.

In tools mode luna queries on about a fifth of its decisions and its prompt cost stays
flat against hints mode. Gemini queries on five sixths of them, and because every tool
turn resends the decision's whole conversation, its input per decision goes 3.4k to
19.3k, a 5.7x increase that shows up as 4.5x the cost. Its output per decision falls
the other way, 279 to 69: it stops reasoning in tokens and reasons by calling
`simulate` instead.

`simulate` accounts for nearly all of both seats' usage: 338 of gemini's 350 calls and
172 of luna's 173. `board` and `waits` were occasional and `discards` was never called
in either arm. Two notes were saved all match. Both seats swept their own discard
options and ignored the opponents. That is a finding about the models, not the tools.

The cap sizing follows from the same table. At 4, a quarter of gemini's decisions and
a sixth of luna's hit the cap and had to commit early, which changes how the seat
plays; at 12 that is 3.1% and 1.5%. An exhaustive sweep is about 20 calls (each of up
to 14 discards, plus board and waits), so the default is 32: above what a thorough
seat needs, and far below the runaway that motivated the cap. Gemini's 8 fallbacks at
the 12-call budget were provider-side rather than truncation: one `ProviderError` plus
six calls that came back with no finish reason, the same malformed-upstream family
seen on the bank runs.

## Crash recovery

With `log_dir` set, every decision is also journaled to `hanchan-<idx>/journal.jsonl`
as it happens. The arena is deterministic given seed + actions, so if a run dies or is
killed, rerunning the same command replays each episode's complete hands from its
journal without a single model call and goes live from the first unrecorded hand. Only
the hand that was in flight is paid for twice. A journal whose header names a
different game (seed, models, or prompt-shaping config changed) fails without
overwriting the old journal. A journal from a finished episode replays the whole
hanchan for free, reproducing its
artifacts with zero API cost. Delete the episode's `journal.jsonl` to force a fresh
game. This composes with the verifiers `--resume` flag, which skips episodes whose
results were already accepted; the journal covers the ones it re-runs.

In tools mode the seats' notes are not journaled: replayed hands re-derive actions,
not conversations, so a resumed seat starts its live hands with an empty scratchpad.

## Notes

- Hard to reward-hack: placement is computed by the Rust referee from final scores,
  every action is validated against the legal menu, and post-hoc grading uses a
  frozen external model. There is no tool surface or environment state to game.
- One hanchan per episode. A seat holds one conversation, and driving several games
  through one engine would interleave their turns into it.
- The taskset is an infinite seeded generator; episode `i` uses seed `20260000 + i`,
  so a run is reproducible from its indices.
