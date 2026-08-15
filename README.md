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

Both measure agreement with Mortal. The decision env does it cheaply, one position at
a time; the hanchan env does it in full self-steered play.

## Publishing to the Environments Hub

Both packages target the Verifiers v1 `Taskset` API. Use the v1 `eval` and
`validate` CLIs; the legacy `verifiers.load_environment(...)` entry point is not
implemented.

The checked-in environment versions are release candidates. Publish them privately
before consuming `0.1.0`:

```console
$ git push origin <release-branch>
$ gh pr create --base main --head <release-branch>
$ gh pr checks --watch
$ gh pr merge --merge --delete-branch
$ prime upgrade
$ prime login
$ prime whoami
$ prime env push --path environments/riichi_decision_v1 --visibility PRIVATE --plain
$ prime env push --path environments/riichi_hanchan_v1 --visibility PRIVATE --plain
```

Merging Git first is required: `riichi-hanchan-v1` pins `jongbench` to an
immutable Git commit so the Hub's clean build container does not depend on an
unpublished registry package. For each private package:

1. Wait for its Hub Action to report `SUCCESS`.
2. Install it in a fresh Python 3.12 environment using the command on its Hub page.
3. Run `validate riichi_decision_v1 --runtime.type subprocess` or
   `validate riichi_hanchan_v1 --runtime.type subprocess -n 4`.
4. Run a four-task decision smoke evaluation and one complete hanchan evaluation.

Only after those gates pass, change each package version from `0.1.0rc1` to `0.1.0`
and push it with `--visibility PUBLIC`. The hanchan smoke is intentionally last: one
episode costs roughly 1,000 model calls.

A trained Phoenix reviewer checkpoint is an external model artifact, not environment
package data. Configure a private hanchan environment's Hub **Variables** with a
public checkpoint URL, its required digest, and policy mode:

```text
JONGBENCH_WEIGHTS_URL=https://.../reviewer-phoenix.pth
JONGBENCH_WEIGHTS_SHA256=<64-character SHA-256>
JONGBENCH_WEIGHTS_USE_POLICY=1
```

Environment Actions and hosted evaluations receive those variables automatically.
The checkpoint is downloaded into the normal verified cache; an incomplete pair,
bad digest, or checkpoint without a reviewer policy head fails closed. Without the
variables, `auto` remains the pinned Mortal 298k checkpoint and Q policy. The
decision environment is unaffected because its Mortal rewards are frozen into the
published bank.

Local v1 validation does not require a Prime account:

```console
$ uv run --with verifiers==0.3.0 --with ./environments/riichi_decision_v1 \
    validate riichi_decision_v1 --runtime.type subprocess
$ uv run --with verifiers==0.3.0 --with ./environments/riichi_hanchan_v1 \
    validate riichi_hanchan_v1 --runtime.type subprocess -n 4
```

## Why riichi

- **Imperfect information, stochastic deals.** Every decision is a probability
  judgment against hidden hands and a hidden wall; board states effectively never
  repeat, so there are no memorized lines to retrieve.
- **Zero-sum, four players.** Placement rewards are a permutation of 1–4 and always
  sum to 2.0, and every board a seat faces was steered by the other three. It is a
  multi-agent credit-assignment setting, not self-play against a copy.
- **Long horizon.** A seat makes ~170 decisions per hanchan, and the cost of a bad
  push/fold call often lands many turns later.
- **A strong non-LLM oracle.** Mortal prices every legal action per decision, so the
  signal is dense (per-decision Q-advantage) rather than win/lose. One game yields
  hundreds of graded decisions instead of one outcome bit.
- **Hard to reward-hack.** The grader is a frozen external model, every action is
  validated against the legal menu by the Rust engine, and there is no tool surface
  or environment state to game.
- **Headroom.** On the shipped decision bank, uniform-random guessing scores 0.367
  reward (18.9% match), always picking the first option scores 0.341, and Mortal
  scores 1.0 by construction. Measured models land inside that gap: gpt-5.6-luna
  0.776, deepseek-v4-flash-0731 0.834, gemini-3.5-flash-lite 0.844.

## Setup

Python 3.12–3.13 and a Rust toolchain are required when installing from source.
The benchmark extra installs provider and Mortal dependencies; setuptools-rust
builds `libriichi` as part of the package:

```console
$ uv sync --python 3.12 --extra benchmark
```

Prefix the commands below with `uv run`, or run them from an activated project
environment.

The AGPL-3.0 Mortal checkpoint is downloaded from
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) on first use,
verified by SHA-256, and cached under `${XDG_CACHE_HOME:-~/.cache}/jongbench`.
Set `JONGBENCH_CACHE_DIR` to override the cache root, or pass `--weights PATH` for
a local checkpoint.

Standalone model calls use [OpenRouter](https://openrouter.ai); set
`OPENROUTER_API_KEY`. The Hub environments use the Verifiers client and default to
Prime Inference instead.

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

# pool a directory of runs into one table, keyed by model spec
$ jongbench leaderboard runs/<batch>/ --review

# generate open training logs, fit a Phoenix-policy head, then run a duplicate duel
$ jongbench selfplay --games 256 --out training/selfplay
$ jongbench train --logs training/tenhou/2026 --out weights/reviewer.pth \
    --data-provenance https://github.com/NikkeTryHard/tenhou-to-mjai/releases/tag/v2.0.0
    --data-sha256 c37af299d9c382cc45608e6a253a0c966d038335493a55bfcc06d6fdf2674816
$ jongbench duel --challenger weights/reviewer.pth --challenger-policy --games 256

# iterate on-policy self-play, clipped policy updates, and duplicate-match promotion
$ jongbench improve --init weights/reviewer.pth --out training/policy-league \
    --rounds 4 --rollout-games 256 --duel-games 512 --device cuda

# watch one game live (terminal board, or --ui web for the browser board;
# a seat spec of `human` puts you at the table)
$ jongbench watch --models anthropic/claude-sonnet-5 openai/gpt-5.2 google/gemini-3-pro random
$ jongbench watch --ui web --models ... human

# replay a finished game on the web table, results overlay included
$ jongbench replay runs/<stamp>/
$ jongbench replay runs/<stamp>/ --out replay.json   # static bundle for hosting
```

Reviewer checkpoints keep Mortal's Q network as the outcome-value axis and add a
masked policy distribution, next-rank head, and confidence head. Training starts the
policy exactly at Mortal's temperature-scaled action distribution, then fits expert
actions while regularizing back to that teacher. Validation is file-disjoint and its
metrics and data-provenance string are stored in the checkpoint. Plain or gzip MJAI
logs are accepted. The shipped training path uses no NAGA reports, outputs, or code.
The legacy `prob` field is a softmax display weight over Mortal Q-values, not a
calibrated probability; trained reports expose `policy_prob` separately.

`jongbench improve` keeps Mortal's encoder and Q head frozen. Each round samples
one stochastic current-policy seat against three frozen greedy copies, trains only
from that seat's decisions using centered final-placement returns and a clipped
behavior-policy ratio, and regularizes toward the original expert-trained policy.
A candidate is promoted only when its duplicate
1-vs-3 point estimate clears `promotion_margin` after subtracting
`promotion_z * paired_standard_error`; four seat rotations of each seed form one
paired sample. The final manifest also records direct duplicate matches against the
initial policy and Mortal Q control.

Model specs are OpenRouter ids: `<vendor>/<model>`, optionally suffixed with
`@<provider>` to pin inference routing (reproducibility against one upstream) and
`#<effort>` to set reasoning (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`) — e.g. `openai/gpt-oss-120b@cerebras#high`. Also `compat:<base_url>:<model>`
for any local OpenAI-compatible endpoint, `random`, `human`, and `mortal` — the
Mortal NN itself as a deterministic control seat.

State hints are on by default: rule-derived shanten, waits, furiten and
discard-result structure, with no EV, safety ranking, or recommended move. Engines
never see hidden tiles or Mortal.

The web table is a TypeScript app in `webui/`; build it once with
`cd webui && bun install && bun run build` (emits `jongbench/webui_page.html`,
gitignored). It is both the live spectator and the replay viewer, and the built
page is a single self-contained file: host it anywhere static (GitHub Pages)
with a `replay.json` beside it and it opens on a landing page with that game
ready to watch — or open it with `#replay` to skip straight to the replay,
which doubles as a file picker for any `--out` bundle.
`.github/workflows/pages.yml` deploys exactly that: the built page plus the
showcase game in `assets/showcase-replay.json.gz`.

See `DESIGN.md` for architecture. Licensed AGPL-3.0 (inherits from vendored Mortal
code). The Mortal weights are the community-trained
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) checkpoint;
this project is strictly for offline benchmarking, not online play assistance.
