"""Command-line entry point for replay, causal branching, and paired reduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .capsule import ReplayCapsule
from .observations import (
    AllControlBaseline,
    BranchResult,
    MatchedControlResult,
    PairedArmObservation,
    read_jsonl,
)
from .reduce import (
    reduce_factual_branches,
    reduce_matched_controls,
    reduce_paired_arms,
)


def _capsule(source: str) -> ReplayCapsule:
    path = Path(source)
    return ReplayCapsule.from_episode(path) if path.is_dir() else ReplayCapsule.read(path)


def _baseline(source: str) -> AllControlBaseline:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("baseline file must contain a JSON object")
    return AllControlBaseline.from_dict(value)


def _write_json(value: object, destination: str | None) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination is None or destination == "-":
        sys.stdout.write(payload)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _choice_values(values: list[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    if len(set(values)) != len(values):
        raise ValueError("--choice values contain duplicates")
    return tuple(values)


def _load_rows(paths: list[str]) -> list[object]:
    rows: list[object] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jongbench-experiment",
        description=(
            "Build replay capsules, run exact-wall interventions, and reduce paired "
            "Mahjong experiments without relying on task-level average reward."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capsule = commands.add_parser(
        "capsule", help="freeze an episode's deterministic replay contract"
    )
    capsule.add_argument("episode")
    capsule.add_argument("--out")

    branch = commands.add_parser(
        "branch", help="force legal actions on the episode's factual hidden wall"
    )
    branch.add_argument("source", help="episode directory or replay-capsule JSON")
    branch.add_argument("--decision", type=int, required=True)
    branch.add_argument("--choice", type=int, action="append")
    branch.add_argument("--reference-choice", type=int)
    branch.add_argument("--weights", default="auto")
    branch.add_argument("--device", default=None)
    branch.add_argument("--out", required=True)
    branch.add_argument("--no-checkpoint-match", action="store_true")

    baseline = commands.add_parser(
        "control-baseline",
        help="cache one all-control outcome for a wall, indexed by chair",
    )
    baseline.add_argument("source", help="episode directory or replay-capsule JSON")
    baseline.add_argument("--weights", default="auto")
    baseline.add_argument("--device", default=None)
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--no-checkpoint-match", action="store_true")

    match = commands.add_parser(
        "match-control", help="join an episode to an existing all-control baseline"
    )
    match.add_argument("source", help="episode directory or replay-capsule JSON")
    match.add_argument("baseline")
    match.add_argument("--seat")
    match.add_argument("--out")

    control = commands.add_parser(
        "matched-control",
        help="cache the all-control wall and join one episode to its chair",
    )
    control.add_argument("source", help="episode directory or replay-capsule JSON")
    control.add_argument("--seat")
    control.add_argument("--weights", default="auto")
    control.add_argument("--device", default=None)
    control.add_argument("--out", required=True)
    control.add_argument("--no-checkpoint-match", action="store_true")

    branch_reduce = commands.add_parser(
        "reduce-branches", help="cluster-reduce factual-wall audit records"
    )
    branch_reduce.add_argument("inputs", nargs="+")
    branch_reduce.add_argument("--confidence", type=float, default=0.95)
    branch_reduce.add_argument("--bootstrap-samples", type=int, default=4000)
    branch_reduce.add_argument("--seed", type=int, default=0)
    branch_reduce.add_argument("--out")

    control_reduce = commands.add_parser(
        "reduce-controls", help="reduce complete four-chair matched-control walls"
    )
    control_reduce.add_argument("inputs", nargs="+")
    control_reduce.add_argument("--confidence", type=float, default=0.95)
    control_reduce.add_argument("--bootstrap-samples", type=int, default=4000)
    control_reduce.add_argument("--seed", type=int, default=0)
    control_reduce.add_argument("--allow-incomplete-walls", action="store_true")
    control_reduce.add_argument("--out")

    paired_reduce = commands.add_parser(
        "reduce-paired", help="reduce prompt/tool interventions within pair IDs"
    )
    paired_reduce.add_argument("inputs", nargs="+")
    paired_reduce.add_argument("--baseline-arm", required=True)
    paired_reduce.add_argument("--treatment-arm", required=True)
    paired_reduce.add_argument("--confidence", type=float, default=0.95)
    paired_reduce.add_argument("--bootstrap-samples", type=int, default=4000)
    paired_reduce.add_argument("--seed", type=int, default=0)
    paired_reduce.add_argument("--out")
    return parser


def _require_rows(rows: list[object], expected: type[Any]) -> list[Any]:
    unexpected = sorted(
        {type(row).__name__ for row in rows if not isinstance(row, expected)}
    )
    if unexpected:
        raise TypeError(
            f"expected only {expected.__name__} rows, found {', '.join(unexpected)}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "capsule":
        capsule = ReplayCapsule.from_episode(args.episode)
        destination = args.out or str(Path(args.episode) / "replay-capsule.json")
        capsule.write(destination)
        _write_json({"capsule_id": capsule.capsule_id, "path": destination}, None)
        return 0

    if args.command == "branch":
        from .branch import run_factual_branches

        capsule = _capsule(args.source)
        results = run_factual_branches(
            capsule,
            args.decision,
            choices=_choice_values(args.choice),
            reference_choice=args.reference_choice,
            weights=args.weights,
            device=args.device,
            output_dir=args.out,
            require_checkpoint_match=not args.no_checkpoint_match,
        )
        _write_json(
            {
                "capsule_id": capsule.capsule_id,
                "opportunity_id": capsule.opportunity_id(args.decision),
                "decision_id": capsule.decision_id(args.decision),
                "estimand": "factual_wall",
                "results": [result.as_dict() for result in results],
            },
            None,
        )
        return 0

    if args.command == "control-baseline":
        from .matched_control import run_control_baseline

        result = run_control_baseline(
            _capsule(args.source),
            weights=args.weights,
            device=args.device,
            output_dir=args.out,
            require_checkpoint_match=not args.no_checkpoint_match,
        )
        _write_json(result.as_dict(), None)
        return 0

    if args.command == "match-control":
        from .matched_control import match_control

        result = match_control(
            _capsule(args.source), _baseline(args.baseline), seat=args.seat
        )
        _write_json(result.as_dict(), args.out)
        return 0

    if args.command == "matched-control":
        from .matched_control import run_matched_control

        result = run_matched_control(
            _capsule(args.source),
            seat=args.seat,
            weights=args.weights,
            device=args.device,
            output_dir=args.out,
            require_checkpoint_match=not args.no_checkpoint_match,
        )
        _write_json(result.as_dict(), None)
        return 0

    rows = _load_rows(args.inputs)
    if args.command == "reduce-branches":
        typed = _require_rows(rows, BranchResult)
        reductions = reduce_factual_branches(
            typed,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.command == "reduce-controls":
        typed = _require_rows(rows, MatchedControlResult)
        reductions = reduce_matched_controls(
            typed,
            require_complete_walls=not args.allow_incomplete_walls,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.command == "reduce-paired":
        typed = _require_rows(rows, PairedArmObservation)
        reductions = reduce_paired_arms(
            typed,
            baseline_arm=args.baseline_arm,
            treatment_arm=args.treatment_arm,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    else:
        raise AssertionError(f"unhandled command {args.command!r}")

    _write_json([reduction.as_dict() for reduction in reductions], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
