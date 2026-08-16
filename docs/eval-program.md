# Evaluation program: causal Mahjong experiments, not one benchmark number

## Research decision

Jongbench should not compete as another frozen multiple-choice benchmark. Its useful
asset is a deterministic Mahjong runtime with legal-action enumeration, exact rule
state, fixed controls, full prompts, decision journals, and replayable walls. The
research program should use those properties to run interventions that ordinary eval
harnesses cannot run.

The execution layer emits immutable facts. A separate reducer owns every paired or
clustered estimate. No environment-level `avg_reward` is allowed to masquerade as the
competence profile.

## Measurement hierarchy

### 1. Matched control replacement — primary live outcome

For each wall seed and physical chair:

1. collect the policy episode against fixed controls;
2. run the same wall once with four copies of the fixed control and cache the
   physical-chair outcome;
3. reuse that baseline for every rotated policy episode on the wall;
4. subtract the matched control chair's score from the policy seat's score;
5. average the four chair deltas into one independent wall observation;
6. bootstrap whole walls, never kyoku or decision rows.

This estimator is valid only when the policy episode has exactly one live policy
seat and three controls matching the baseline identity. The two games are expected to
diverge; that divergence is the intervention. Placement remains a readable secondary
outcome. `table_score_margin` remains descriptive only.

Controls must be deterministic. A stochastic control is admissible only after its RNG
stream is explicitly coupled across policy and baseline runs.

### 2. Exact-wall action branching — consequential audit

A finished episode already contains a stronger state handle than a private Rust object:

- arena seed and key;
- physical table order;
- measurement profile and control identity;
- the globally ordered decision journal.

The arena is deterministic given the seed and action history. A content-addressed
**replay capsule** therefore reconstructs a target decision by replaying the factual
prefix. At the target, an experiment forces a legal alternative, preserves every other
model reaction made to the same event prefix, and then hands every seat to fresh fixed
controls. Preserving those co-temporal reactions avoids changing two interventions at
once. This requires no additional model call.

The first implementation reports the target kyoku's realized point delta for every
forced action. Its estimand is named `factual_wall`: it answers what each action caused
on the hidden wall that was actually dealt.

That is useful for:

- auditing catastrophic consequential disagreements;
- comparing the model's action with a frozen reviewer's action on the same realization;
- finding policy-induced states where reviewer scores and realized
  consequences conflict;
- selecting examples for human analysis.

It is **not** conditional expected value. Taking the best result on the already-known
future wall creates hindsight selection and must not become a headline skill score.
`hindsight_regret` is an audit statistic only. The more defensible aggregate is
the model-versus-reference branch delta across held-out wall clusters.

### 3. Information-set branching — next research milestone

Expected action value requires resampling hidden tile allocations conditional on the
information available to the evaluated seat. Replaying the original wall repeatedly
does not create independent continuations.

The next native-runtime project is therefore an information-set resampler that:

- fixes all public events and the evaluated seat's private hand;
- preserves the legal state and visible tile counts;
- resamples opponents' concealed tiles and the remaining wall from compatible unseen
  tiles;
- couples each sampled hidden world across all forced actions;
- finishes the kyoku under fixed controls.

Only this estimator may be called expected branch value or expected branch regret.
Its validation must include reconstruction invariants, tile conservation, legal-action
stability at the branch point, and reviewer-Q calibration on held-out games.

### 4. Perception-to-judgment decomposition

A model response should be able to include derived rule facts and its action in one
structured answer:

```json
{
  "facts": {
    "shanten": 1,
    "furiten": false,
    "live_wait_count": 0,
    "genbutsu_against": {"P1": ["3p", "E"]}
  },
  "choice": 4
}
```

Facts are graded exactly by the runtime; action quality is graded separately. Reports
must preserve the four cells: correct/incorrect facts crossed with strong/weak action.
Visible-text retrieval such as copying `tiles_left` is a format check, not a Mahjong
perception result.

### 5. Paired invariance experiments

Menu permutation, notation changes, hints, tools, and irrelevant wording are paired
interventions on the same board. They are never appended to the primary task stream and
pooled into one mean.

Every observation carries a stable `pair_id`, `cluster_id`, profile, arm, metric, value,
and validity flag. Reducers report within-pair differences, dropped/malformed pairs,
and whole-cluster bootstrap intervals. High-consequence choice flips are retained as
auditable examples.

### 6. Reviewer diagnostics and disagreement

Mortal Q remains a frozen diagnostic, not ground truth. Reports retain raw Q loss,
normalized loss, Q span, and action probabilities. Phoenix policy confidence remains
policy-imitation metadata and cannot weight Mortal grading.

Two independently frozen reviewers should define consensus, close-value, and contested
sets. Exact-wall and information-set branches then test which reviewer disagreements
matter in play.

## Architecture

### Immutable experiment records

The experiment package defines content-addressed records in focused schema modules:

- `experiments.identity.ControlPolicyIdentity` — checkpoint and policy settings;
- `experiments.capsule.ReplayCapsule` — one episode's causal replay contract;
- `experiments.capsule.ScriptedDecision` — one globally ordered action;
- `experiments.records_control.AllControlBaseline` — a reusable chair-indexed
  wall outcome;
- `experiments.records_branch.BranchResult` — one forced legal action;
- `experiments.records_control.MatchedControlResult` — one policy/baseline
  chair join;
- `experiments.records_paired.PairedArmObservation` — one intervention arm.

Source filesystem paths are metadata and are excluded from capsule identity.

### Execution

`jongbench.experiments.runtime`, `branch`, and `matched_control` own one
explicit state transition:

```text
factual replay prefix -> forced target action -> fixed continuation controls
```

A `ReplayThenEngine` reproduces recorded decisions and otherwise delegates to an
ordinary control seat. For live seats, optional reactions absent from the journal are
auto-passed through the global intervention cut—even when that seat has no later row
of its own. Fixed controls are recomputed throughout. Once the cut is crossed, the
continuation policy owns every action.

Mortal controls are built through one canonical `MortalControlPool`: resolve one
checkpoint, load one template, and share frozen modules across fresh arena-seat
wrappers. Runtime, branching, matched baselines, and self-play should converge on this
construction path.

### Reduction

`jongbench.experiments.reduce` owns:

- complete four-chair wall checks;
- matched-control wall means;
- whole-wall bootstrap intervals;
- complete legal-action checks for branch audits;
- model-versus-reference factual deltas;
- paired-arm joins and dropped-pair accounting.

The reducer equal-weights independent clusters. It never treats the hundreds of
correlated decisions from one hanchan as hundreds of independent samples.

## Statistical contract

- Independent units are walls, games, or stable paired boards.
- Pairing is preserved before aggregation.
- Four rotated episodes are one smoke block, not a confidence interval.
- Missing arms, incomplete chairs, invalid responses, and replay divergence are visible
  outcomes, not silently dropped rows.
- Every result names its measurement profile and estimand.
- Public collection declares a target precision and stops only at a cluster boundary.
- Calibration uses held-out games and reports predictive error, not only in-sample
  fit.

## Implementation status

Implemented in the current research branch:

- honest behavioral and exact-rule diagnostics;
- raw and normalized reviewer loss with Q span;
- rejection of invalid reviewer-confidence weighting;
- branch-core CI validation against the branch wheel;
- content-addressed replay capsules;
- deterministic replay-then-control execution;
- exact-wall action branches with explicit `factual_wall` semantics;
- cached one-per-wall all-control baselines reused across chair rotations;
- complete-block matched-control reduction;
- pair-aware invariance reduction;
- whole-cluster bootstrap intervals.

Still required before a benchmark claim:

1. collect a multi-wall matched-control study on at least two model families;
2. run the new command surface against real saved episodes and publish the
   immutable artifacts;
3. run a held-out model-versus-Mortal branch study and test Q/consequence correlation;
4. implement and validate information-set hidden-tile resampling;
5. replace retrieval-style comprehension probes with derived-fact responses;
6. remove diagnostic slices that neither separate models nor explain failures;
7. merge the release-audit base and publish matching core/environment artifacts.

The branch remains a research draft until those gates are met.
