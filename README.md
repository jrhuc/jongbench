# jongbench

Benchmark LLMs at riichi mahjong. Four models play full hanchan against each other
under Tenhou rules (via [Mortal](https://github.com/Equim-chan/Mortal)'s `libriichi`
Rust engine), then every decision is graded by the Mortal deep-RL model using the same
rating algorithm as [mjai.ekyu.moe](https://mjai.ekyu.moe/) (ported from
[mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer)).

## Setup

```console
$ uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e .
$ (cd libriichi && PYO3_PYTHON=../.venv/bin/python cargo build --release --lib)
$ cp libriichi/target/release/libriichi.dylib jongbench/libriichi.so
$ curl -L -o weights/mortal.pth https://huggingface.co/VoidShine/mortal-298k/resolve/main/mortal_298k.pth
```

API keys via env: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## Usage

```console
# test, no API keys needed
$ jongbench selfcheck

# benchmark: N games, summary + HTML report
$ jongbench run --models anthropic:claude-sonnet-5 openai:gpt-5.2 google:gemini-3-pro random --games 8

# raw-reasoning mode: omit engine-derived shanten/wait/furiten hints
$ jongbench run --no-state-hints --models anthropic:claude-sonnet-5 openai:gpt-5.2 google:gemini-3-pro random --games 8

# watch one game live in the terminal
$ jongbench watch --models anthropic:claude-sonnet-5 openai:gpt-5.2 google:gemini-3-pro random

# watch in the browser (add `human` as a model spec to take a seat yourself)
$ jongbench watch --ui web --models anthropic:claude-sonnet-5 openai:gpt-5.2 google:gemini-3-pro human

# host the web UI (visitors configure seats and bring their own keys)
$ jongbench serve --host 0.0.0.0 --port 8642

# re-evaluate / regenerate a report
$ jongbench review runs/<stamp>/
```

The web UI is a TypeScript app in `webui/`; the built page is committed at
`jongbench/webui_page.html`, so running it needs no JS toolchain.

State hints are enabled by default. They add rule-derived shanten, waits, furiten, and
discard-result structure without exposing hidden tiles, EV, safety, or a recommended move.
Use `--no-state-hints` in the CLI or clear the web setup checkbox for raw reasoning.

Win events in the live log and saved mjai log include ron/tsumo, base hand points,
fu/han or yakuman count, and the engine-calculated yaku list.

Model specs: `anthropic:<model>`, `openai:<model>`, `google:<model>`,
`compat:<base_url>:<model>` (any OpenAI-compatible endpoint), `random` (baseline).

See `DESIGN.md` for architecture. Licensed AGPL-3.0 (inherits from vendored Mortal
code). The Mortal weights are the community-trained
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) checkpoint;
this project is strictly for offline benchmarking, not online play assistance.
