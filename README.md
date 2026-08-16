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
  multi-agent env: one evaluated model plays a full hanchan against three fixed
  Mortal controls. One placement reward is reported per episode; four rotated
  episodes balance table position. Set `log_dir` and each rollout is also a
  jongbench run dir that `review` and `reasoning` grade post-hoc.

The decision env measures agreement with Mortal cheaply, one position at a time. The
hanchan env measures game outcome against fixed Mortal controls in self-steered play;
its optional post-hoc review reports decision-level agreement separately.

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

`riichi-hanchan-v1` pins versioned CPython 3.12 and 3.13 x86-64 manylinux
wheels, including SHA-256 fragments, because the Hub build image does not include
Rust. Those requirements select the `mortal` extra so a published environment can
run a `mortal` control seat, and platform markers keep the Linux wheels out of
macOS and arm64 source-checkout resolution. If a release changes the core package,
merge that change first, run the manual `Release verification` workflow, publish
its `jongbench-manylinux-wheels` artifact under the matching
`jongbench-v<version>` tag, then update both wheel URLs and digests in the hanchan
package before pushing either environment. For each private package:

1. Wait for its Hub Action to report `SUCCESS`.
2. Install it in a fresh Python 3.12 environment using the command on its Hub page.
3. Run `validate riichi_decision_v1 --runtime.type subprocess` or
   `validate riichi_hanchan_v1 --runtime.type subprocess -n 4`.
4. Run a four-task decision smoke evaluation and one complete, four-episode
   chair-balanced hanchan evaluation.

Only after those gates pass, change each package version from `0.1.0rcN` to `0.1.0`
and push it with `--visibility PUBLIC`. The hanchan smoke is intentionally last: the
four-episode rotated batch costs roughly 1,000 model calls and is a smoke test, not
the headline comparison. A real hanchan eval needs a multiple of four episodes and
must report seed-block standard error beside the mean.

Phoenix reviewer v1 is a separate release asset and a **control/opponent**, not the
grader. In-env and CLI grading always use Mortal 298k. Use Phoenix only with a
private hanchan environment. Set these Hub **Variables**:

```text
JONGBENCH_WEIGHTS_URL=https://github.com/jrhuc/jongbench/releases/download/reviewer-phoenix-2026-v1/reviewer-phoenix-2026-v1.pth
JONGBENCH_WEIGHTS_SHA256=1ba7f63a2ae0555ce1a99c76fed45d44c20162689015afe3568b2befabe693ab
```

The loader requires both the URL and digest and verifies them before loading.
Pass `--env.control-use-policy true` to opt a hanchan control into Phoenix's policy
head; deterministic Mortal-Q control remains the default. The resolved source,
digest, path, and policy mode are recorded in episode artifacts.

In an independent 1,024-game duplicate match, Phoenix v1 scored
`avg_pt=1.36` with `standard_error=1.069` against its parent policy. Against
Mortal Q, it scored `avg_pt=0.57` with `standard_error=2.070`. The Mortal Q
result is not statistically significant. Keep Phoenix v1 private while
evaluating model seats. The decision environment does not load this checkpoint.

Local v1 validation does not require a Prime account:

```console
$ uv run --with verifiers==0.3.0 --with ./environments/riichi_decision_v1 \
    validate riichi_decision_v1 --runtime.type subprocess
$ uv run --extra mortal --with verifiers==0.3.0 \
    --with ./environments/riichi_hanchan_v1 \
    validate riichi_hanchan_v1 --runtime.type subprocess -n 4
```

## Why riichi

- **Imperfect information, stochastic deals.** Every decision is a probability
  judgment against hidden hands and a hidden wall; board states effectively never
  repeat, so there are no memorized lines to retrieve.
- **Constant-sum, four players.** Normalized placement rewards always sum to 2.0,
  and every board a seat faces was steered by the other three. The standard eval
  scores one policy against fixed controls rather than pooling the invariant
  four-seat total or evaluating a model only against copies of itself.
- **Long horizon.** A seat makes ~170 decisions per hanchan, and the cost of a bad
  push/fold call often lands many turns later.
- **A strong non-LLM oracle.** Mortal prices every legal action per decision, so the
  signal is dense (per-decision Q-advantage) rather than win/lose. One game yields
  hundreds of graded decisions instead of one outcome bit.
- **Hard to reward-hack.** The grader is a frozen external model, every action is
  validated against the legal menu by the Rust engine, and there is no tool surface
  or environment state to game.
- **Headroom.** On the shipped decision bank, uniform-random guessing scores 0.364
  reward (16.5% match), always picking the first option scores 0.373, and Mortal
  scores 1.0 by construction. Model numbers from the previous 13-hand sample are
  withdrawn; they do not apply to this bank.

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
a local checkpoint. Run configs record the resolved path, digest, source, and selected
head for every Mortal gameplay seat and for the reviewer, including `--no-eval` runs.

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

# create training logs, train reviewer heads, and run a duplicate match
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

### Reviewer architecture

A reviewer checkpoint uses the Mortal v4 encoder and Q head. It adds three
modules:

- A policy head maps the shared 1,024-value encoder output to the 46 Mortal
  actions. Illegal actions are masked.
- A rank head predicts the player's position after the current hand.
- A confidence head predicts whether the policy's first choice matches the
  logged action.

The Q head remains available for standard Mortal review. Policy mode uses the
policy head to select actions. The rank and confidence heads add fields to
review output. They do not select actions. Reports keep the Q-based `prob`
field and add `policy_prob`, `policy_confidence`, and `next_rank_probs`.

### Reviewer training

`jongbench train` initializes the policy from Mortal's Q head. The default
teacher temperature is 0.1. The encoder and Q head are frozen by default.
Training fits the logged action, the next rank, and policy confidence. It also
penalizes divergence from the initial Mortal policy. Files are assigned to the
training or validation set by a stable hash of the file name. Metrics, data
provenance, and the data digest are stored in the checkpoint. `--data-provenance` and
a 64-character lowercase `--data-sha256` must be supplied together.

`jongbench improve` keeps the encoder and Q head frozen and updates only the
policy head. Each round collects stochastic self-play from one policy seat
against three copies of the current policy. The update uses final-placement
returns, a clipped policy ratio, anchor-policy regularization, and entropy
regularization. Promotion uses duplicate 1-vs-3 games. Four seat rotations for
one seed form one paired sample. A candidate passes when
`avg_pt - promotion_z * standard_error` is greater than
`promotion_margin`.

Plain and gzip MJAI logs are supported. The training path does not use NAGA
reports, outputs, or code. See `DESIGN.md` for module dimensions and runtime
behavior.

Model specs are OpenRouter ids: `<vendor>/<model>`, optionally suffixed with
`@<provider>` to pin inference routing (reproducibility against one upstream) and
`#<effort>` to set reasoning (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`) — e.g. `openai/gpt-oss-120b@cerebras#high`. Also `compat:<base_url>:<model>`
for any local OpenAI-compatible endpoint, `random`, `human`, and `mortal` — the
Mortal NN itself as a deterministic control seat.

State hints are on by default: rule-derived shanten, waits, furiten and
discard-result structure, with no EV, safety ranking, or recommended move. Engines
never see hidden tiles or Mortal.

The web table is a TypeScript app in `webui/`. Its self-contained generated
page, `jongbench/webui_page.html`, is versioned and included in source and wheel
distributions so an ordinary PEP 517 build is complete without a hidden frontend
step. After changing the UI, run
`cd webui && bun install --frozen-lockfile && bun run build` and commit the result;
`bun run build --check` verifies that it is current. It is both the live spectator
and the replay viewer: host the page anywhere static (GitHub Pages)
with a `replay.json` beside it and it opens on a landing page with that game
ready to watch — or open it with `#replay` to skip straight to the replay,
which doubles as a file picker for any `--out` bundle.
`.github/workflows/pages.yml` deploys exactly that: the built page plus the
showcase game in `assets/showcase-replay.json.gz`.

See `DESIGN.md` for architecture. Licensed AGPL-3.0 (inherits from vendored Mortal
code). The Mortal weights are the community-trained
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) checkpoint;
this project is strictly for offline benchmarking, not online play assistance.
