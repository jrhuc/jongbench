# Scoping: tool-using seats (riichi as an agentic env)

Status: stages 1 and 2 built (`tools: true` on riichi-hanchan-v1 — toolset, minimal
turns, notes persistence). Follows the prime-agent idea — programs over data instead
of tokens over data (https://www.primeintellect.ai/blog/prime-agent) — mapped onto
the hanchan env.

## The idea

Today a seat is fed everything: narrated event deltas, its hand, always-on state
hints, an annotated menu. Measured over a real hanchan, hints alone are 23% of
input tokens and most turns need almost none of it. The inversion: give the seat a
*minimal* turn (delta + menu, no hints) and a set of board-query tools it calls
when it actually wants information. What persists across kyoku is not transcript
but the seat's own notes — the one thing the current design deliberately throws
away (per-kyoku reset drops the model's opponent reads with everything else).

This changes what is measured, on purpose: information-seeking and memory
management become part of the skill. That makes it a different benchmark, not a
cheaper version of the current one — both stay.

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
  into the next kyoku's rollout state. Retrieval-only (`notes()`), no injection —
  the honest-agency arm; injection remains an A/B if retrieval never fires.
- Per-kyoku interactions stay: tool results append to the kyoku transcript, so
  keeping results terse matters more than today.
- Config knob on the existing env (`tools: true` prompts minimal + toolsets on),
  not a separate env package, until it earns one.
- One toolset process per seat per kyoku (~48 launches per episode). Fine for a
  spike; measure the overhead before scaling episode counts.

## What it costs

A tool call is a full model roundtrip carrying the transcript, so an
over-querying seat costs MORE than always-on hints — that is a finding, not a
bug. The interesting readouts, all cheap to log per seat:

- tool calls per decision (and which tools),
- Mortal rating vs the hints-on and hints-off baselines,
- input tokens per decision vs both baselines,
- whether `note()` gets used at all, and whether notes correlate with rating.

## Staged plan

1. **Spike** — DONE: `board()`/`discards()`/`simulate()`/`waits()` toolset,
   `tools: true` knob, minimal turn prompt, offline tests plus a live MCP
   launch probe. Remaining exit question needs a live episode: do models call
   tools at all under the strict output contract, and does the choice-JSON
   contract survive tool-call turns?
2. **Notes** — BUILT (`note()`/`notes()`, persistence across kyoku, retrieval
   arm); the injection A/B is unbuilt and waits on evidence retrieval is used.
3. **The comparison** — same lineup, 4-8 episodes each: hints-on vs hints-off vs
   tools. Rating, cost, query patterns. This is the writeup artifact.
4. **RL tie-in (later)** — a tool-surface env is the more interesting RL target:
   the policy learns what to look up, not just what to discard. Decision-bank
   training stays the cheap first rung (see DESIGN).

## Open questions

- Does the strict `{"choice": N}` output contract fight the tool-call loop on
  some models (tool call + JSON in one turn)? May need "answer only after tools".
- Toolset lifecycle vs per-kyoku rollouts: one colocated toolset per seat per
  kyoku is the default shape; confirm launch overhead is negligible.
- Does minimal-turn + tools hurt weak models so much the comparison is all floor?
  Bounded, measured: gpt-5-mini scores 0.638 reward hints-off vs 0.727 hints-on
  on the decision bank — a real drop, nowhere near the 0.367 random floor. Tools
  have ~0.09 reward of headroom to win back at lower steady-state cost.
