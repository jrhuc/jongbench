# Scoping: tool-using seats (riichi as an agentic env)

Status: stages 1, 2 and 3 done (`tools: true` on riichi-hanchan-v1 — toolset,
minimal turns, notes persistence, measured against both hint arms). The design
applies the prime-agent approach (https://www.primeintellect.ai/blog/prime-agent)
to the hanchan env.

## The design

A seat is currently fed everything each turn: narrated event deltas, its hand,
always-on state hints, an annotated menu. Measured over a real hanchan, hints are
23% of input tokens, and most turns use almost none of them. Tools mode gives the
seat a minimal turn instead (delta and menu, no hints) plus board-query tools it
calls when it wants information. Across kyoku it carries its own notes rather than
transcript; the per-kyoku reset otherwise discards the model's opponent reads along
with everything else.

This measures something different: information seeking and memory management become
part of the task. Tools mode is a separate benchmark arm, not a cheaper version of
the existing one. Both are kept.

## Tool surface (rule-derived only, nothing hidden, nothing Mortal)

Every tool answers from the seat's own `PlayerState` + public events — exactly the
information the prompt path may use today, so no new information leaks:

- `board()` — the full render `render_state` produces today, on demand.
- `discards(player)` — one player's discard row, with riichi tile marked.
- `simulate(tile)` — resulting shanten / waits / furiten for one legal discard
  (today's per-option hint, priced at one tool call instead of every menu line).
- `waits()` — current shape: shanten, tiles that advance, furiten status.
- `note(text)` / `notes()` — a per-seat scratchpad that SURVIVES kyoku resets;
  the continual-harness "memory" component, model-managed.

Deliberately absent: safety oracles, EV tables, anything Mortal — same rule as
state hints today. `safety(tile)` derived purely from public discards (genbutsu)
is defensible but starts to coach; decide when building.

## Architecture (as built)

- verifiers `Task.toolsets()` returns FastMCP-backed `vf.Toolset`s; the null
  harness already speaks MCP (`SUPPORTS_MCP`), advertises the tools and runs the
  call loop. No harness work. `SeatToolsTask` in the env carries the toolset.
- Correction from the original sketch: `colocated=True` means "same runtime", not
  "same process" — the framework always launches a toolset as `python -m
  riichi_hanchan_v1.tools` in its own process, so tools can never touch the live
  Rust `PlayerState`. Instead the engine precomputes every possible answer at the
  decision point (`prompts.decision_snapshot`: board render, discard rows, shape
  summary, per-discard simulate results — all rule-derived) and publishes it
  before the model call; the env writes it to `trace.state`, which the
  interception serves to the toolset over the rollout's state channel
  (GET/PUT `/state`). `simulate()` is a dict lookup by the time the model calls it.
- Notes live in `SeatState.notes`: `note()` pushes them back through the state
  channel, the env reads them off `trace.state` after each turn and seeds them
  into the next kyoku's rollout state. Retrieval only (`notes()`): notes are never
  injected into the prompt, so the model has to ask for them. Automatic injection
  stays an unbuilt A/B arm.
- Per-kyoku interactions stay: tool results append to the kyoku transcript, so
  keeping results terse matters more than today.
- A config knob on the existing env (`tools: true` switches prompts to minimal and
  turns toolsets on) rather than a separate env package.
- One toolset process per seat per kyoku (~48 launches per episode). Fine for a
  spike; measure the overhead before scaling episode counts.

## What it costs

A tool call is a full model roundtrip carrying the transcript, so a seat that
queries often costs more than always-on hints. That is a result to measure, not a
defect. The readouts, all cheap to log per seat:

- tool calls per decision, and which tools,
- input tokens per decision against the hints-on and hints-off arms,
- cost per decision against both arms,
- whether `note()` is used at all.

Mortal rating is not one of them. Hanchan traces carry `final_score`, `decisions`,
`fallbacks` and `calls_declined` per seat; rating exists only after the separate
Mortal review pass (`jongbench summarize`).

## Staged plan

1. **Spike** — DONE (seed 20260000, cap 12, luna + gemini-flash-lite vs two Mortal
   controls). Both exit questions came back positive. The seats queried without
   being told to: gemini on 84% of its decisions (350 calls over 128), luna on 24%
   (173 over 131). The choice-JSON contract held through tool-call turns; luna had
   zero fallbacks and gemini's 8 trace to provider-side failures (one
   `ProviderError` and six replies with no finish reason) rather than a broken
   contract. `simulate()` accounts for almost every call (172/173 and 338/350) and
   `discards()` was never called. No separate "answer only after tools" phase was
   needed.
2. **Notes** — BUILT (`note()`/`notes()`, persistence across kyoku, retrieval
   arm); the injection A/B is unbuilt and waits on evidence retrieval is used.
3. **The comparison** — DONE, one seed (20260000) replayed four ways: hints-on,
   hints-off, tools at budget 12, tools at budget 4. Same lineup each time
   (luna seat 0, gemini-flash-lite seat 1, two Mortal controls). The table is in
   the hanchan env README. The result is a cost and query-pattern comparison, not
   a strength ranking: hanchan traces carry no Mortal rating, and placement at one
   episode is noise. A strength comparison needs the review pass over several
   seeds.
4. **RL tie-in (later)** — a tool-surface env is the more interesting RL target:
   the policy learns what to look up, not just what to discard. Decision-bank
   training stays the cheap first rung (see DESIGN).

## Open questions

- ~~Does the strict `{"choice": N}` output contract fight the tool-call loop?~~
  Answered: no, on both models tried (stage 1).
- Toolset lifecycle vs per-kyoku rollouts: one toolset per seat per kyoku is the
  shape as built. Measured launch overhead is ~2-3s from rollout start to the
  MCP server answering — negligible against a kyoku's model calls, but it is
  paid ~48 times an episode, so revisit if episode counts go up.
- The per-decision budget is the main cost lever. With the cap at 12, the
  heavier-querying seat spent 22x what its table-mate did on the same board. At 4
  the cap cut short 25% of gemini's decisions and 17% of luna's; at 12 about 3%
  and 2%. The default is now 32, which clears an exhaustive ~20-call sweep and
  still stops a runaway loop.
- Does a minimal turn plus tools hurt weak models enough that the comparison sits
  at the floor? The only measurement is from the withdrawn pre-fix render
  (gpt-5-mini, 0.638 hints-off against 0.727 hints-on): a real drop, but well
  above the then-current 0.367 random floor (now 0.364 on the schema-v3 sample).
  It has not been remeasured since the render and cap fixes.
