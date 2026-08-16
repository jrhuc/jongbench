# Evaluation program: from reviewer agreement to causal competence

## Decision

Do not merge the competence-profile PR as a finished benchmark.  It contains useful
instrumentation, but several proposed metrics overclaim what they measure and the
current taskset reducers cannot produce the paired or clustered estimates the README
promises.

The project should treat the harness as the product and the initial evals as probes of
that harness.  The distinctive capability is not another frozen multiple-choice bank:
it is a deterministic Mahjong runtime that can reproduce a state, enumerate every
legal action, preserve the wall and chair assignment, continue a branch under fixed
policies, and retain the model's prompt, response, tools, and downstream trajectory.
The evaluation program should be built around those capabilities.

## Immediate corrections

### Keep

- Immutable board, prompt, grading, checkpoint, source-log, and measurement-profile
  identities.
- Raw Q values and normalized rewards, but report them as reviewer diagnostics until
  calibration establishes what the Q scale predicts.
- Competence tags as sampling and slicing labels, not as ground-truth diagnoses.
- Exact rule checks where the runtime can compute the counterfactual directly.
- Rotated common-wall blocks, clustered uncertainty, and explicit smoke-run labeling.
- Behavioral fingerprints as descriptive context.

### Change

- Rename `score_differential` to **table score margin**.  At a constant-sum table,
  own score minus the opponents' mean is an affine transform of own score; it neither
  cancels luck nor creates an independent signal.  The outcome comparison should be a
  policy seat minus an all-control seat on the same wall and chair.
- Compare same-shanten discards when computing ukeire loss.  A larger acceptance count
  at a worse shanten is not a better alternative.
- Define the existing “fold rate” narrowly as exact genbutsu against every declared
  riichi opponent.  Do not label all other discards as pushes.
- Treat closed, non-riichiable, ron-yakuless tenpai as a diagnostic condition, not as
  “cannot win”: menzen tsumo remains a possible yaku.
- Compute reviewer style against its probability distribution, not only its argmax.
- Record failures in every competency slice.  A malformed answer must score zero in the
  relevant tag metrics rather than disappearing from them.
- Reduce hint, menu-order, notation, and comprehension arms as paired experiments.
  Appending them to the ordinary task stream and averaging one scalar destroys the
  effect the experiment was meant to estimate.

### Remove or quarantine

- **Reviewer-confidence weighting.**  The available Phoenix confidence head is trained
  to predict whether Phoenix's policy matches logged play.  It is not uncertainty in
  Mortal's Q estimate, and weighting Mortal scores by it is circular.  Preserve it only
  as explicitly named policy-imitation metadata.
- Any claim that the environment already emits a seed-block standard error.  A
  per-episode `Env` cannot manufacture a batch statistic; the reducer must group the
  finished episode artifacts.
- The current board-comprehension “tiles left” item as evidence of Mahjong perception.
  It is mostly direct text retrieval.  Comprehension probes should require reconstructing
  hidden rule facts from the rendered board.
- Calibration claims from in-sample ordinary least squares.  Fit on held-out games and
  bootstrap whole game or wall clusters.

## Measurement architecture

### 1. Immutable rollout artifacts

The environment should emit facts, not final benchmark claims.  One episode artifact
must include:

- wall seed, rotation, table position, control checkpoint, and complete measurement
  profile;
- every prompt variant and legal menu;
- the parsed action and raw response;
- a serializable decision-state handle sufficient to reproduce or branch the state;
- rule diagnostics and reviewer outputs as separate namespaces;
- final outcomes and the exact cluster/pair identifiers used by reducers.

A reducer then owns all aggregation: paired deltas, tag profiles, bootstrap intervals,
invariance gaps, and missing/failure accounting.  This prevents UI averaging rules from
silently changing the estimand.

### 2. Counterfactual branch regret — primary novel eval

At selected decisions, replay the exact state and force each legal action once.  From
that point, let fixed controls finish the kyoku, using common random numbers across
branches.  No additional call to the evaluated model is required after its original
choice.

For decision \(d\), action \(a\), and continuation replicate \(k\):

\[
R_{d,a,k}=\text{realized seat point delta under forced action }a.
\]

Report:

\[
\text{branch regret}_d = \max_a \bar R_{d,a} - \bar R_{d,a_{model}},
\]

with uncertainty across continuation replicates and game clusters.  Also retain the
full action-value vector, because close choices and catastrophic mistakes should not be
compressed into the same 0–1 stretch.

This is the strongest use of the harness:

- it is consequential rather than imitation-based;
- it can validate or falsify Mortal Q on the exact positions being scored;
- it works on states the evaluated model caused, including off-distribution states;
- it separates model-call cost from simulator compute;
- and it exposes whether reviewer disagreement matters in realized play.

Implementation order:

1. Add a stable `DecisionSnapshot` serialization containing PlayerState, wall cursor,
   scores, honba/kyotaku, turn owner, and legal events.
2. Add `arena.branch(snapshot, forced_event, continuation_policy, continuation_seed)`.
3. Pilot only on discard decisions with 2–6 materially distinct options and high Q
   span; continue to end of kyoku.
4. Use 4–16 common-random-number continuations per option, selected adaptively until
   the best-action ordering is stable or the budget is exhausted.
5. Compare branch regret with reviewer Q-loss on held-out games.  A weak relationship is
   a benchmark result, not an implementation failure.

### 3. Matched control intervention — primary live outcome

For every wall seed, run an all-control baseline once and cache it.  Compare the policy
seat's final score with the baseline control occupying the same chair on the same wall.
Average the four chair deltas in a rotated block:

\[
\Delta_s = score(policy\ replacement, s) - score(all\ control, s).
\]

This is a real counterfactual intervention.  It does not pretend that the policy and
baseline follow identical trajectories; trajectory divergence is the effect being
measured.  Controls must be deterministic, or their randomness must be explicitly
coupled.

Placement remains a human-readable secondary result.  Table score margin remains a
single-episode descriptive metric only.

### 4. Perception → judgment decomposition

Use one structured response on the same prompt:

```json
{
  "facts": {
    "shanten": 1,
    "furiten": false,
    "genbutsu_against": {"P1": ["3p", "E"]},
    "live_wait_count": 0
  },
  "choice": 4
}
```

Grade rule facts exactly, then grade the action separately.  Report four cells:

- facts correct / action strong;
- facts correct / action weak;
- facts wrong / action strong;
- facts wrong / action weak.

This directly tests whether inline hints and tools repair perception, judgment, or
both.  Probes must require derived board facts; copying a visible counter is only a
format sanity check.

### 5. Paired invariance suite

Each original board receives controlled transformations with stable pair IDs:

- menu permutation;
- equivalent tile notation or rendering;
- hints on/off;
- tools versus inline facts;
- irrelevant wording perturbations.

Primary metrics are choice-flip rate, branch-regret delta, and answer-validity delta.
Never average transformed tasks into the ordinary decision score.  Report an invariance
matrix and inspect the largest-regret flips.

### 6. Compounding-error profile

Maintain two banks with identical schema and reducers:

- **reference distribution:** positions reached by strong self-play or high-level human
  logs;
- **policy-induced distribution:** positions reached by the evaluated model in live
  games.

Match or reweight by observable state features before comparing scores.  Report both
performance and the state-distribution shift: shanten, score pressure, calls, furiten,
remaining tiles, opponent riichi, and action-menu composition.  The gap reveals where a
model's earlier mistakes create later situations the reference bank systematically
misses.

### 7. Reviewer disagreement, not fictitious confidence

Build banks with two independently frozen reviewers.  Align rows by `board_id` and
report:

- consensus argmax positions;
- close-value positions;
- contested positions;
- reviewer-versus-branch-regret agreement.

Consensus is a useful high-precision slice.  Contested positions are an audit set for
human analysis, not examples to force into one supposedly correct label.

## Statistical contract

- The independent unit is a game, wall, or paired board—not a decision row.
- Confidence intervals bootstrap those clusters and preserve paired arms.
- A result always names the measurement profile and missing/failure rate.
- `n=4` is a smoke run only.
- Public comparisons state a target precision before collecting data and may stop only
  at a cluster boundary.
- Raw and normalized reviewer losses are both shown, together with Q span.
- Calibration uses held-out games and reports predictive error, not only in-sample
  \(R^2\).

## Framework changes

The current Verifiers task abstraction is suitable for issuing calls, but not for every
estimator.  Keep it as the execution layer and add a jongbench reducer layer that reads
immutable artifacts.  The reducer should be the only component allowed to publish a
benchmark summary.

Recommended interfaces:

```text
jongbench collect <profile> ...        # prompts, actions, rollouts, snapshots
jongbench branch <run> ...             # counterfactual continuation artifacts
jongbench reduce <run-or-bank> ...     # paired/clustered estimates and report
jongbench audit <report> ...            # worst-regret and disagreement examples
```

The Hub environment may still expose lightweight per-task metrics, but it should link
to the reducer report rather than treating `avg_reward` as the competence profile.

## Merge sequence

### Gate A — feature branch validity

- Run environment integration against the branch's core wheel, not an older released
  wheel that cannot contain the branch's modules.
- Keep the published-install check separate.
- Reject confidence filtering and weighting.
- Fix exact scorecard semantics and failure accounting.
- Ensure all unit, native, wheel, and environment-source tests pass.

### Gate B — publication validity

- Merge the release-audit base first.
- Publish and verify 0.1.1 manylinux wheels for Python 3.12 and 3.13.
- Replace every 0.1.0 direct URL and digest in the hanchan package.
- Install the environment from its own wheel in a clean process and run four smoke
  episodes.
- Regenerate any artifact whose schema or metric semantics changed.

### Gate C — benchmark claim

- Ship the reducer and clustered/pair-aware report.
- Publish no score-differential or confidence-weighted headline.
- Run the branch-regret pilot and the Q-loss calibration on held-out games.
- Demonstrate that at least one proposed slice separates models or explains a live
  failure mode.  Remove slices that do neither.

## Work started in this PR

- Correct same-shanten ukeire comparison and stream the scorecard state in one pass.
- Make fingerprint rates hand-aware and multi-ron-safe.
- Compare style with the reviewer's action distribution.
- Expose raw and normalized Q-loss plus Q span.
- Count malformed responses in tag slices.
- Reject the invalid reviewer-confidence weighting path.
- Validate the hanchan environment against the branch's 0.1.1 core wheel in CI.

These changes make the current diagnostics honest.  They do not, by themselves, turn
the PR into the final benchmark; the causal branch evaluator and reducer are the next
high-value implementation work.
