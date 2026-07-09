# jongbench — design

A benchmark that has 4 LLMs play riichi mahjong against each other, then grades every
decision with the Mortal deep-RL model (same scoring as mjai.ekyu.moe).

## Data flow

```
model specs ──> engines (LLM adapters, mjai-log protocol)
                    │
                    ▼
        libriichi.arena.FourEngines        (vendored Rust, runs full Tenhou-rules hanchan)
                    │
        ┌───────────┴────────────┐
   mjai logs (.json.gz)   decision logs (.jsonl, per-LLM)
        │
        ▼
   evaluate.py  ── Mortal weights (weights/mortal.pth, v4 b40c192)
        │            per-decision Q-values → rating (port of mjai-reviewer/src/review/mortal.rs)
        ▼
   report.py → report.html (summary table + per-game per-decision review)
```

## Repo layout

```
libriichi/                 vendored from Mortal, + src/arena/four_engines.rs (4 distinct engines, seat rotation)
jongbench/                 python package
  libriichi.so             built cdylib (cargo build --release --lib, PYO3_PYTHON=.venv python)
  mortal_model.py          vendored Mortal/mortal/model.py   (Brain, DQN, GRP)
  mortal_engine.py         vendored Mortal/mortal/engine.py  (MortalEngine)
  tiles.py  actions.py  engines.py  providers.py  prompts.py
  arena.py  evaluate.py  spectator.py  report.py  cli.py
assets/pai.svg             tile sprite sheet (from mjai-reviewer)
weights/mortal.pth         gitignored, from https://huggingface.co/VoidShine/mortal-298k
runs/<stamp>/              per-run artifacts: config.json, logs/*.json.gz, decisions/*.jsonl,
                           review/*.json, report.html
```

## mjai cheat sheet

Tile notation: `1m..9m 1p..9p 1s..9s E S W N P F C`, red fives `5mr 5pr 5sr`.
Events (JSON, one per line): `start_game start_kyoku tsumo dahai chi pon daiminkan
kakan ankan reach reach_accepted dora hora ryukyoku end_kyoku end_game none`.
A player's reaction is one event JSON; `{"type":"none"}` passes.
Arena logs are God-view (all `tehais` filled). Engines only ever see their own POV.

## Module contracts

### tiles.py
- `TILES`: canonical 37-tile order used by libriichi labels 0..36 (0..8 = 1m..9m, 9..17 = 1p..9p,
  18..26 = 1s..9s, 27..33 = E S W N P F C, 34/35/36 = 5mr/5pr/5sr).
- `tile_to_label(s) -> int`, `label_to_tile(i) -> str`.
- `fmt_tile(s, glyphs=False) -> str`: ascii (`5pr` shown as `0p`, honors as letters) or Unicode
  mahjong glyph (U+1F000 block). `fmt_hand(list, ...)` sorts and groups by suit.
- `sort_tiles(list) -> list` in canonical order.

### actions.py
- `build_menu(state: PlayerState) -> list[MenuItem]` where
  `MenuItem = {label: str, event: dict | {"type":"none"}, kind: str}`.
- Enumerates from `state.last_cans` + getters: discards (one per distinct hand tile, aka distinct,
  `tsumogiri` flag set when the tile is `state.last_self_tsumo()`), `reach`, chi low/mid/high
  (consumed reconstruction copied from mjai-reviewer `src/review/mortal.rs::to_event`, akaize when
  holding the red 5), pon (aka handling ditto), daiminkan, ankan (per `state.ankan_candidates()`),
  kakan (per `state.kakan_candidates()`), hora (tsumo/ron via `can_tsumo_agari`/`can_ron_agari`,
  target = `last_cans.target_actor`), ryukyoku, and none/pass when reacting to others.
- Every candidate is filtered through `state.validate_reaction(json)`; invalid ones dropped.
  This guarantees the menu is exactly the legal action set (e.g. riichi discard restrictions).
- After `reach` is chosen, the next `can_discard` turn only offers tenpai-keeping discards —
  handled automatically by the validate filter.

### engines.py
- `BaseEngine`: implements the mjai-log protocol (`engine_type='mjai-log'`, `name`,
  `set_player_ids`, `react_batch(game_states)`, `start_game/end_kyoku/end_game` no-ops).
  `react_batch` maps each game_state through `self.decide(player_id, state, events)` and
  runs decisions concurrently (ThreadPoolExecutor) since games are independent.
- `RandomEngine`: takes hora when offered, otherwise uniform random legal action (pass-biased
  when reacting). For testing/baselines; deterministic given seed.
- `LLMEngine(spec, name)`: builds menu; if menu is single-item, auto-picks (no API call);
  else prompts the model (prompts.py), expects `{"choice": <int>}` (tolerate reasoning text
  around the JSON — extract last JSON object). Invalid → one retry with the error appended.
  Final fallback: tsumogiri/none. Records every decision to a .jsonl sink:
  `{game_seed, kyoku, honba, player_id, menu, choice, fallback, raw_response, usage, latency_ms}`.
- Engines never see Mortal, shanten helpers, or EV tables — the LLM plays unassisted.
  Prompt state is built only from public info + own hand (see prompts.py).
- The arena's `events_json` is GOD-VIEW; `sanitize_events(events, player_id)` masks other
  seats' `tehais` and drawn tiles, and every engine sanitizes before `decide()`. Raw events
  go only to the spectator (which legitimately renders the full table).
- `HumanEngine(BaseEngine)`: a human fills the seat. `decide()` delegates to a blocking
  `HumanIO.ask(player_id, state, events, menu) -> event` — events are already POV-sanitized
  by BaseEngine, so a human UI can only ever show the player's own hand.
  * `TerminalHumanIO`: prints `prompts.render_state` + numbered menu, reads a choice from
    stdin (re-asks on invalid input).
  * `WebHumanIO`: exposes the pending decision to the web server (`pending() -> {menu,
    state_text, seat}`), blocks on a threading.Event until `choose(index)` is called by the
    POST handler. One decision pending at a time; a generation counter guards stale POSTs.
- Seat spec `human` (accepted wherever a model spec is): `make_engine` returns a HumanEngine
  wired to the right HumanIO for the frontend in use. Human games get the same Mortal
  review as models — the human's rating is directly comparable.
- Human mode in the web UI restricts the board to the player's POV (opponents' hands
  face-down): the server masks `snapshot()` hands for seats != pov before sending.

### prompts.py
- `SYSTEM`: concise riichi rules reminder + "you are playing a benchmark game" + strict
  output contract (JSON `{"choice": N}` as the LAST line; brief reasoning allowed above).
- `render_state(player_id, state, events) -> str`: compact text block built from PlayerState
  getters + parsed kyoku events: round/honba/kyotaku, seat winds, scores, dora indicators,
  own hand (sorted, aka marked) + drawn tile, own melds, per-player discard rows (riichi tile
  marked with `*`), riichi declarations, tiles left, shanten is NOT shown (no coaching).
- `render_menu(menu) -> str`: numbered options with human labels.

### providers.py
- `parse_spec(s)`: `anthropic:<model>`, `openai:<model>`, `google:<model>`,
  `xai:<model>` and `deepseek:<model>` (presets over the OpenAI-compatible client,
  base urls https://api.x.ai/v1 and https://api.deepseek.com, keys `XAI_API_KEY` /
  `DEEPSEEK_API_KEY`), `compat:<base_url>:<model>` (any OpenAI-compatible), `random`.
- `Provider.complete(system, user, *, max_tokens=1200, temperature=0.6) -> (text, usage_dict)`
  using official SDKs (`anthropic`, `openai`, `google-genai`), SDK retries enabled,
  60s timeout. Fail with a clear message when a key is missing.
- Keys come from env by default but every provider accepts an explicit `api_key`
  override (`make_provider(spec, api_key=None)`) — the web UI passes visitor-supplied
  keys this way; they live only in that game's memory and are never logged or persisted.

### arena.py
- `run_games(engines, games, seed=(N,key), log_dir) -> list[GameSummary]` wrapping
  `libriichi.arena.FourEngines.py_4p`. Seat rotation is built into the Rust side
  (engine i sits seat (i+g)%4 in game g). Returns names/scores/seed per game;
  computes placements.

### evaluate.py — faithful port of mjai-reviewer/src/review/mortal.rs
- `load_engine(weights_path)`: torch checkpoint → Brain+DQN (version from ckpt config) →
  `MortalEngine(..., enable_quick_eval=False, enable_rule_based_agari_guard=True, device=cpu)`.
- `review_player(events: list[dict], player_id, engine) -> Review` using `libriichi.mjai.Bot`:
  feed each event; when Mortal reacts with meta and `popcount(mask_bits) > 1`, find the
  player's actual next action (`next_action` port), map to label (`to_label`/`to_kan_label`
  ports incl. kan_select sub-metadata), pop q_values against descending mask bits, softmax
  (temperature 1) for display probs, and accumulate
  `raw_rating += 1 if matched else (q_actual - q_min)/max(q_max - q_min, 1e-6)`.
  Final `rating = (raw_rating / total_reviewed)**2` (0..1; display ×100).
  Match uses aka-insensitive comparison for consumed sets (port `equal_ignore_aka_consumed`).
- Review entries carry: kyoku/honba/junme/tiles_left, actual event, Mortal's expected event,
  is_equal, ranked candidate list [{event, q_value, prob}], shanten, at_furiten, actual_index.
- `review_game(events, engine) -> dict[player_id -> Review]`, plus aggregates:
  match_rate, mean prob-loss, worst N decisions (by prob(best) - prob(actual)), breakdown by
  decision kind (discard/call/riichi/agari/pass).
- No GRP (rank-probability curves omitted from reports).

### spectator.py (watch mode)
- `Spectator`: subscribed by engines; each `decide()` first publishes that seat's newly seen
  events (diff by index per kyoku). Public events are deduped (they appear in all 4 POVs);
  own-draw events supply the hidden info, so merging 4 POVs reconstructs the full table.
- `TableState`: the merged model (scores, round/honba, dora indicators, per-seat hand
  (sorted), melds, discard rows, riichi state, last action, event ticker). Shared by the
  terminal renderer, the web UI, and the replay recorder. Also emits a serializable
  event feed: `{seq, event, table_snapshot?}` for SSE/replay.
- Terminal renderer: draws the table as a SQUARE board — self seat at the bottom, opponents
  on right/top/left; top row horizontal, left/right discards as vertical tile stacks
  (one tile per line); ascii tiles with suit colors, aka in red; `--glyphs` for Unicode
  glyphs. Event ticker below the board. Throttle/step delay configurable.

### webui.py (watch/serve mode)
- Stdlib `http.server.ThreadingHTTPServer`, no framework. Single self-contained HTML page
  (inline CSS/JS, TEXT tiles — monospace ascii tiles, red fives colored; no image assets).
  The board is a square: viewer-selected seat at the bottom, other three seats' tile rows
  rotated 90/180/270 with CSS transforms around the center (discard ponds in the middle,
  hands/melds on the edges, scores + winds in the corners).
- Endpoints: `GET /` app page; `POST /api/start` {models: [4 specs], keys: {provider: key},
  seed} → starts a game thread, returns run_id (visitor keys are held in RAM for that run
  only, never logged); `GET /api/events/<run_id>` SSE stream of spectator feed (replays
  the backlog on connect, so reconnect/late-join works); `GET /api/state/<run_id>` snapshot;
  `GET /api/review/<run_id>` review JSON once evaluation finishes; `GET /api/demo` a
  recorded bundle (see replay below). Front-end has Live and Replay modes — Replay steps
  through a recorded feed client-side with a speed slider, needing no keys.
- `jongbench watch --ui web` starts the server for a local one-game run (auto-opens
  browser); `jongbench serve --host 0.0.0.0 --port N` runs it as a shareable app where
  visitors bring their own keys.
- Replay bundle (post-MVP): `jongbench record <run_dir>` packs the spectator feed +
  decisions + review of one game into `demo.json` served at `/api/demo`.

### report.py
- `write_report(run_dir)`: single self-contained `report.html` (inline CSS/JS, embedded
  review JSON, inline pai.svg sprite for tile rendering).
- Summary: one row per model — games, avg placement, avg final score, Mortal rating (×100),
  match rate, fallback rate (LLM errors), tokens/latency if available.
- Per game (collapsible): seats/placements/scores, per-player rating; per-kyoku decision
  table like mjai.ekyu.moe: board context, actual action vs Mortal's ranked candidates with
  probability bars; mistakes highlighted by prob-loss; strong/weak aggregates dropdown.

### cli.py  (`python -m jongbench`, script `jongbench`)
- `run    --models A B C D --games N [--seed S] [--label X] [--no-eval] [--concurrency K]`
  (`human` not allowed in batch runs)
- `watch  --models A B C D [--seed S] [--delay MS] [--glyphs] [--ui term|web]`
  (games=1 + spectator + report; any seat may be `human` — the UI then switches to that
  seat's POV and prompts for choices; terminal uses TerminalHumanIO, web uses WebHumanIO)
- `review <run_dir | log.json.gz> [--out FILE]`  (re-evaluate + regenerate report)
- `selfcheck`  (4 random engines, 1 game, evaluate, report — no API keys needed)
- Models: provider specs from providers.py; `random` allowed for filling seats.

## Testing
- `selfcheck` is the end-to-end gate (arena → eval → report) with no network.
- `tests/test_actions.py`: run N seeded random games with an engine that asserts, at every
  decision point, every menu item passes `validate_reaction` and the menu is non-empty.
- `tests/test_evaluate.py`: review a tsumogiri game; assert rating in (0,1), entries present,
  and a hand-checked decision matches Mortal's meta.

## Conventions
- Python 3.12, venv at .venv (uv). Deps: torch, numpy, anthropic, openai, google-genai, rich.
- AGPL-3.0 (inherited from vendored Mortal/libriichi code).
- The benchmark is offline-only; weights are the community VoidShine/mortal-298k checkpoint.
