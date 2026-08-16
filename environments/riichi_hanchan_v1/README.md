# riichi-hanchan-v1

One evaluated model plays a full Tenhou-rules hanchan against three fixed Mortal
controls in the vendored `libriichi` arena. The default four episodes rotate the
evaluated role through every chair while reusing one wall seed; this balances chair
assignment but remains a **smoke run**, not a ranking-quality sample.

Live play measures something a frozen decision bank cannot: the model changes the
future states on which it is judged. It is correspondingly expensive and noisy. Keep
placement, final score, reviewer diagnostics, cost, and failure behavior separate
rather than compressing them into one headline.

## What one run estimates

- `placement` is the standard bounded environment reward: first = 1.0, fourth = 0.0.
- `final_score` is a descriptive outcome in game points.
- `table_score_margin` (historically `score_differential`) is own final score minus the
  other seats' mean. At a constant-sum table it is only an affine transform of own
  score. It does **not** cancel deal luck or add an independent outcome signal.
- Rotation controls chair assignment. It does not create a matched all-control
  counterfactual by itself.
- In-environment Mortal grading supplies reviewer-relative decision diagnostics from
  the same trajectory, without another model rollout.

A genuine common-random-number outcome comparison requires a separate all-control
baseline on the same wall and chair, followed by a reducer that computes policy-seat
minus matched-control-seat deltas and aggregates complete four-chair wall blocks.
That reducer is proposed in `docs/eval-program.md`; do not label the current table
margin as duplicate-adjusted performance.

## Run

A four-episode smoke run uses the run model as `seat0` and deterministic Mortal
controls in the other roles:

```console
$ .venv/bin/eval riichi_hanchan_v1 \
    -m MODEL \
    --env.log-dir episodes \
    -n 4 --no-push
```

For local development from the repository:

```console
$ uv run --extra mortal --with-editable . --with verifiers==0.3.0 \
    --with ./environments/riichi_hanchan_v1 \
    eval riichi_hanchan_v1 -n 4 --no-push
```

`--with-editable .` is important: the environment imports branch-core modules and
must be tested against the same core revision. Published environment wheels may pin a
released core, but pull-request validation must not substitute an older release for
the code under review.

Seats may be overridden with `--env.seatN.model`. Set `evaluated_agent=None` only for
a tournament whose per-agent outputs will be reduced separately. Pooling all four
placement rewards is invariant because their sum is constant.

## Config

| key | default | meaning |
|---|---:|---|
| `seat0.model` | run model | policy under evaluation |
| `seat1..3.model` | `mortal` | fixed deterministic control opponents |
| `evaluated_agent` | `seat0` | sole role contributing standard eval reward |
| `state_hints` | `true` | include rule-derived shanten/waits/furiten |
| `auto_pass_reactions` | `false` | pass pure reaction menus without a model call |
| `tools` | `false` | expose board-query tools instead of inline hints |
| `max_tool_calls` | `32` | per-decision tool budget; `0` removes the cap |
| `seat_rotation` | `true` | rotate the evaluated role through four chairs |
| `log_dir` | `None` | persist each episode as a jongbench run directory |
| `weights` | `auto` | verified checkpoint used by local control seats |
| `control_use_policy` | `false` | use a checkpoint policy head for controls |
| `control_boltzmann_epsilon` | `0` | optional control-policy mixture |
| `control_boltzmann_temp` | `1` | temperature for that mixture |
| `grade` | `true` | run Mortal review, fingerprint, scorecard, and Q-loss |

## Seat rotation and sampling

With rotation enabled, `-n` must be a multiple of four. Episodes are grouped into
four-chair wall blocks. Report uncertainty over complete wall blocks, not over kyoku
or individual decisions. Four episodes provide one wall block and therefore no useful
empirical standard error.

`trace.info["hanchan"]` follows the evaluated role and records its physical chair,
policy role, placement, kyoku count, and whether the trace carries the evaluation
reward. Each persisted episode records the wall seed, rotation, table assignment, and
measurement profile.

## Evaluation versus training aggregation

A live seat usually creates one trace per kyoku; a tool-budget fallback can split a
kyoku. During evaluation, every trace records the outcome but exactly one trace for
the evaluated role carries weight 1, so each hanchan contributes once regardless of
length. Training clients may apply the bounded placement reward to each evaluated-role
kyoku trace.

`final_score`, `decisions`, `fallbacks`, `calls_declined`, token usage, provider cost,
and tool use are seat totals. They are diagnostics, not replacements for a
cluster-aware outcome reducer.

## Mortal control and reviewer roles

The default `mortal` opponent is local and makes no API calls. Mortal 298k is also the
frozen grader. These are conceptually distinct roles even when they use related model
code.

Phoenix may be selected as an experimental **control opponent** through the verified
`JONGBENCH_WEIGHTS_URL` and `JONGBENCH_WEIGHTS_SHA256` pair plus
`--env.control-use-policy true`. Its confidence head predicts policy-imitation
correctness; it is not Q-value uncertainty and must not weight Mortal grading.
Checkpoint source, digest, and policy mode are recorded in the run artifact.

## Tools versus hints

`--env.tools true` removes inline state hints and exposes board, discard, wait,
simulation, and private-note tools. Tool use changes both information access and cost:
each tool turn is another model round trip. The environment records tool calls,
decisions that queried, exhausted budgets, and saved notes.

Hints-on, hints-off, and tools arms must be run on matched walls and reduced as paired
interventions. A one-seed demonstration is useful for debugging, not for a model
claim. Do not pool the arms into the ordinary placement or reviewer score.

## Persisted artifacts and post-hoc review

With `log_dir` set, each episode contains the MJAI log, per-seat decisions, prompts,
configuration, and measurement profile:

```console
$ jongbench review episodes/hanchan-00000
$ jongbench reasoning episodes/hanchan-00000
```

These artifacts are the durable product. A separate reducer should own paired outcome
deltas, cluster bootstrap intervals, competence slices, invariance matrices, and
missing/failure accounting.

## Interpretation and merge gate

A credible public comparison should include multiple complete wall blocks, an
explicit target precision, failure coverage, and a matched all-control baseline when
claiming causal outcome improvement. Placement remains human-readable; Mortal grading
remains dense; neither is ground truth by itself.

The next high-value evaluation is counterfactual branch continuation: snapshot a real
decision, force every legal action, and let fixed controls finish the kyoku under
common random numbers. That directly tests whether reviewer disagreement predicts
realised downstream regret and uses the harness's deterministic state machinery more
fully than another static bank.
