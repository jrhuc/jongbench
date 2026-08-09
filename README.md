# jongbench

Riichi mahjong as an LLM eval. Models play full Tenhou-rules hanchan (via
[Mortal](https://github.com/Equim-chan/Mortal)'s `libriichi` Rust engine), and every
decision is graded by the Mortal deep-RL model with the same rating algorithm as
[mjai.ekyu.moe](https://mjai.ekyu.moe/) (ported from
[mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer)).

Two [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments package
the eval, backed by a shared harness:

- **[environments/riichi_decision_v1](environments/riichi_decision_v1/)** — one
  Mortal-graded decision per task. Byte-identical prompts across models, one call per
  graded decision, pure trace scoring (no runtime, no checkpoint at eval time).
- **[environments/riichi_hanchan_v1](environments/riichi_hanchan_v1/)** — the
  multi-agent env: four seats, one hanchan per episode, zero-sum placement reward.
  Set `log_dir` and each rollout is also a jongbench run dir that `review` and
  `reasoning` grade post-hoc.

Both measure the same quantity — agreement with Mortal — two ways: cheap and
separable, or in full self-steered play.

## Why riichi

- **Imperfect information, stochastic deals.** Every decision is a probability
  judgment against hidden hands and a hidden wall; board states effectively never
  repeat, so there are no memorized lines to retrieve.
- **Genuinely zero-sum, four players.** Placement rewards are a permutation of 1–4
  (they always sum to 2.0), and every board a seat faces was steered by the other
  three — a real multi-agent credit-assignment setting, not self-play against a copy.
- **Long horizon.** A seat makes ~170 decisions per hanchan, and the cost of a bad
  push/fold call often lands many turns later.
- **A strong non-LLM oracle.** Mortal prices every legal action per decision, so the
  signal is dense (per-decision Q-advantage), not just win/lose — one game yields
  hundreds of graded decisions instead of one outcome bit.
- **Hard to reward-hack.** The grader is a frozen external model, every action is
  validated against the legal menu by the Rust engine, and there is no tool surface
  or environment state to game — the failure mode long-horizon game evals usually
  suffer.
- **Headroom.** On the shipped decision bank, uniform-random guessing scores 0.367
  reward (18.9% match) and Mortal scores 1.0 by construction; current models land
  well inside that gap.

## Setup

```console
$ uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e .
$ (cd libriichi && PYO3_PYTHON=../.venv/bin/python cargo build --release --lib)
$ cp libriichi/target/release/libriichi.dylib jongbench/libriichi.so
$ curl -L -o weights/mortal.pth https://huggingface.co/VoidShine/mortal-298k/resolve/main/mortal_298k.pth
```

All models are reached through [OpenRouter](https://openrouter.ai); set
`OPENROUTER_API_KEY`.

## Usage

```console
# end-to-end smoke test, no API key needed
$ jongbench selfcheck

# benchmark: N games, summary + HTML report
$ jongbench run --models anthropic/claude-sonnet-5 openai/gpt-5.2 google/gemini-3-pro random --games 8

# raw-reasoning mode: omit engine-derived shanten/wait/furiten hints
$ jongbench run --no-state-hints --models ... --games 8

# build a bank of Mortal-graded single decisions for riichi-decision-v1
$ jongbench positions --out bank.jsonl --games 4          # from Mortal self-play
$ jongbench positions --out bank.jsonl --from-log runs/<stamp>/logs/g0.json.gz

# re-evaluate / regenerate a report
$ jongbench review runs/<stamp>/

# read a model's reasoning against Mortal's verdict on the same decisions
$ jongbench reasoning runs/<stamp>/ --worst 5

# watch one game live (terminal board, or --ui web for the browser board;
# a seat spec of `human` puts you at the table)
$ jongbench watch --models anthropic/claude-sonnet-5 openai/gpt-5.2 google/gemini-3-pro random
$ jongbench watch --ui web --models ... human
```

Model specs are OpenRouter ids: `<vendor>/<model>`, optionally suffixed with
`@<provider>` to pin inference routing (reproducibility against one upstream) and
`#<effort>` to set reasoning (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`) — e.g. `openai/gpt-oss-120b@cerebras#high`. Also `compat:<base_url>:<model>`
for any local OpenAI-compatible endpoint, `random`, and `human`.

State hints are on by default: rule-derived shanten, waits, furiten and
discard-result structure, with no EV, safety ranking, or recommended move. Engines
never see hidden tiles or Mortal.

The web spectator is a TypeScript app in `webui/`; build it once with
`cd webui && bun install && bun run build` (emits `jongbench/webui_page.html`,
gitignored).

See `DESIGN.md` for architecture. Licensed AGPL-3.0 (inherits from vendored Mortal
code). The Mortal weights are the community-trained
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) checkpoint;
this project is strictly for offline benchmarking, not online play assistance.
