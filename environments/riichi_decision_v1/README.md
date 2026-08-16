# riichi-decision-v1

One real riichi-mahjong decision per task, graded against a frozen Mortal review.
The model receives a board rendered from one seat, a numbered legal-action menu, and
replies with `{"choice": N}`.

The primary reward is Mortal's normalised Q-advantage for the chosen option: 1.0 for
Mortal's highest-Q option, 0.0 for the lowest-Q option, linear between. This is a
**reviewer-relative decision diagnostic**, not ground-truth game value. Every model
sees byte-identical prompts and menus, so comparisons are much denser and less noisy
than live hanchan outcomes, but they inherit the reviewer's blind spots.

## Shipped bank

A schema-v3 sample bank ships with the environment. Each bank begins with an immutable
manifest and stores, per position:

- source-game and measurement-profile provenance;
- `board_id`, `prompt_id`, and full position identity;
- both frozen prompt variants;
- the complete legal menu;
- raw reviewer Q values and normalised rewards;
- reviewer checkpoint digest and competence tags.

The standalone loader validates hashes, dimensions, reward ranges, duplicate IDs,
finite Q values, and `best_index`. Evaluation does not need jongbench, a native
runtime, Torch, or the Mortal checkpoint after the bank has been built.

Build a larger bank from Mortal self-play or an existing MJAI log:

```console
$ jongbench positions --out bank.jsonl --games 4
$ jongbench positions --out bank.jsonl --from-log runs/<stamp>/logs/g0.json.gz
```

Banks sampled from strong self-play are generally more useful for ranking strong
models than random-policy boards. Public results must name the bank digest,
measurement profile, and missing/invalid-answer rate.

## Run

From the Environments Hub:

```console
$ uvx --from verifiers==0.3.0 eval <owner>/riichi-decision-v1 \
    -m MODEL -n 128 -r 1 -c 16 --no-push
```

From this repository:

```console
$ uv run --with ./environments/riichi_decision_v1 eval riichi_decision_v1 \
    -m MODEL -n 128 -r 1 -c 16 --no-push
```

Offline package validation needs no model or API key:

```console
$ uv run --with ./environments/riichi_decision_v1 validate riichi_decision_v1 \
    --runtime.type subprocess
```

## Config

| key | default | meaning |
|---|---:|---|
| `bank` | shipped sample | rendered `.jsonl` or `.jsonl.gz` bank |
| `state_hints` | `true` | select the frozen prompt with rule-derived hints |
| `tags` | empty | keep positions having any comma-separated tag |
| `permute_seed` | `None` | experimental deterministic menu-order arm |
| `both_prompt_variants` | `false` | emit paired hints-on/off arms |
| `probes` | `false` | append experimental rule-comprehension items |
| `min_confidence` | `None` | unsupported; non-`None` is rejected |
| `confidence_weight` | `false` | unsupported; `true` is rejected |

The available Phoenix confidence head predicts policy-imitation correctness. It is
not uncertainty in Mortal's Q estimate, so it must not filter or weight the reward.
It may be retained as explicitly named metadata for reviewer research.

## Rewards and metrics

- `q_advantage` — primary 0–1 reviewer-relative reward. Malformed or out-of-range
  replies score zero and remain in every applicable competence slice.
- `matched_mortal` — exact agreement with Mortal's highest-Q option.
- `answered` — reply parsed to a legal menu index.
- `q_loss` — raw `max(Q) - Q(choice)` in the checkpoint's own units.
- `normalised_q_loss` — Q-loss divided by the legal menu's Q span.
- `q_span` — `max(Q) - min(Q)`, reported so low-stakes and high-stakes menus are not
  silently treated as equivalent.
- `choice_index` — parsed menu index, or `-1` for a malformed answer.
- `tag_<name>` — `q_advantage` on that tagged slice, including malformed answers.

Raw Q units are not automatically points. Any points calibration must be fitted on
held-out games and evaluated with whole-game or whole-wall cluster resampling.

## Experimental arms are paired, not pooled

Menu permutation, hints on/off, notation changes, and comprehension probes are
robustness experiments. They share a board identity and must be reduced as paired
choice-flip, validity, and regret deltas. Do not append them to the ordinary task
stream and average one scalar; that destroys the intervention each arm was designed
to estimate.

The current `tiles_left` probe is mainly a format/retrieval sanity check. Stronger
comprehension probes should require derived facts such as shanten, furiten, live waits,
or genbutsu sets and should grade perception separately from action quality.

## Interpretation

Use this environment for controlled, byte-identical decision comparison. Use
`riichi-hanchan-v1` for live trajectory effects, and use a future branch-continuation
reducer to test whether reviewer Q-loss predicts realised downstream regret. The
repository's evaluation architecture and merge gates are described in
`docs/eval-program.md`.
