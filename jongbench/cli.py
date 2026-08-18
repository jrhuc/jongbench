from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import arena, console, providers, report
from .arena import GameSummary
from .artifacts import decision_filename, file_sha256, load_mjai_log
from .engines import RandomEngine, TerminalHumanIO, make_engine
from .run_artifacts import (
    build_replay_bundle,
    create_run_dir,
    engine_names,
    log_sort_key,
    reconstruct_summary,
    record_gameplay_checkpoints,
    record_reviewer_checkpoint,
    review_log,
    reviews_missing,
    run_log_paths,
    seed_from_events_or_path,
    seed_label,
    summary_by_seed,
    write_review,
    write_run_config,
)
from .spectator import Spectator, TerminalRenderer
from .weights import AUTO_MORTAL_WEIGHTS


def _mortal_evaluate():
    try:
        from . import evaluate
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "Mortal features require `pip install 'jongbench[mortal]'`"
            ) from exc
        raise
    return evaluate


def _decision_sink(path: str | Path) -> Callable[[dict[str, Any]], None]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def sink(record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with lock, output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return sink


def _evaluate_run(
    run_dir: str | Path,
    weights: str | Path,
    temperature: float = 0.1,
    progress: bool = True,
    summaries: Sequence[GameSummary]
    | Mapping[tuple[int, int], GameSummary]
    | None = None,
) -> None:
    evaluate = _mortal_evaluate()
    run = Path(run_dir)
    log_paths = run_log_paths(run)
    if not log_paths:
        raise ValueError(f"no logs found in {run / 'logs'}")

    summaries_by_seed = summary_by_seed(summaries)
    mortal = evaluate.load_engine(str(weights), use_policy=False)
    checkpoint = getattr(mortal, "checkpoint", None)
    if checkpoint is not None:
        record_reviewer_checkpoint(run, checkpoint)

    for log_path in log_paths:
        events = load_mjai_log(log_path)
        seed = seed_from_events_or_path(events, log_path)
        game_summary = summaries_by_seed.get(seed) or reconstruct_summary(
            events, log_path
        )
        payload = review_log(
            log_path,
            game_summary,
            mortal,
            temperature=temperature,
        )
        if progress:
            for player_id in range(4):
                player_review = payload["players"][str(player_id)]["review"]
                rating = float(player_review.get("rating") or 0.0) * 100.0
                print(
                    f"review {seed_label(seed)} P{player_id} "
                    f"{game_summary.names[player_id]} rating {rating:.2f}"
                )
        write_review(run, payload)


def _cmd_run(args: argparse.Namespace) -> int:
    specs = list(args.models)
    if any(_is_human_spec(spec) for spec in specs):
        raise ValueError("run does not support human seats")

    run_dir = create_run_dir(args.runs_root, args.label)
    names = engine_names(specs)
    write_run_config(
        run_dir,
        label=args.label,
        models=specs,
        names=names,
        games=int(args.games),
        seed_start=(int(args.seed), 1),
        state_hints=bool(args.state_hints),
    )

    engines = [
        make_engine(
            name,
            spec,
            decision_log=_decision_sink(
                run_dir / "decisions" / decision_filename(name)
            ),
            concurrency=int(args.concurrency),
            temperature=float(args.temperature),
            state_hints=bool(args.state_hints),
            auto_pass_reactions=bool(args.auto_pass_reactions),
            weights=args.weights,
        )
        for name, spec in zip(names, specs, strict=True)
    ]
    record_gameplay_checkpoints(run_dir, engines)
    summaries = arena.run_games(
        engines,
        int(args.games),
        seed_start=(int(args.seed), 1),
        log_dir=str(run_dir / "logs"),
        disable_progress_bar=False,
    )

    if args.no_eval:
        print(f"run: {run_dir}")
        return 0

    _evaluate_run(run_dir, args.weights, summaries=summaries)
    summary = report.summarize(str(run_dir))
    report_path = report.write_report(str(run_dir), summary)
    console.print_summary(summary)
    print(f"report: {report_path}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    specs = list(args.models)
    if args.ui == "web":
        try:
            from . import webui
        except ImportError:
            print("web UI not built yet", file=sys.stderr)
            return 1
        run_dir = webui.run_watch_server(
            model_specs=specs,
            seed=(int(args.seed), 1),
            delay=float(args.delay),
            runs_root=args.runs_root,
            weights=args.weights,
            no_eval=bool(args.no_eval),
            state_hints=bool(args.state_hints),
            label=args.label,
        )
        print(f"run: {run_dir}")
        return 0

    human_seats = [seat for seat, spec in enumerate(specs) if _is_human_spec(spec)]
    if len(human_seats) > 1:
        raise ValueError("watch supports at most one human seat")
    human_seat = human_seats[0] if human_seats else 0

    run_dir = create_run_dir(args.runs_root, args.label)
    names = engine_names(specs)
    write_run_config(
        run_dir,
        label=args.label,
        models=specs,
        names=names,
        games=1,
        seed_start=(int(args.seed), 1),
        state_hints=bool(args.state_hints),
        human_seat=human_seat,
        no_eval=bool(args.no_eval),
    )

    renderer = TerminalRenderer(
        glyphs=bool(args.glyphs),
        reveal=not human_seats,
        pov=human_seat,
    )
    spectator = Spectator(delay=float(args.delay), on_update=renderer, names=names)
    engines = []
    for name, spec in zip(names, specs, strict=True):
        kwargs: dict[str, Any] = {
            "spectator": spectator,
            "state_hints": bool(args.state_hints),
            "weights": args.weights,
        }
        if _is_human_spec(spec):
            kwargs["human_io"] = TerminalHumanIO()
        else:
            kwargs["decision_log"] = _decision_sink(
                run_dir / "decisions" / decision_filename(name)
            )
        engines.append(make_engine(name, spec, **kwargs))

    record_gameplay_checkpoints(run_dir, engines)
    summaries = arena.run_games(
        engines,
        1,
        seed_start=(int(args.seed), 1),
        log_dir=str(run_dir / "logs"),
        disable_progress_bar=True,
    )
    if summaries:
        spectator.finish(summaries[0].names, summaries[0].scores)
        renderer.on_update(spectator.table)

    if args.no_eval:
        print(f"run: {run_dir}")
        return 0

    _evaluate_run(run_dir, args.weights, summaries=summaries)
    summary = report.summarize(str(run_dir))
    report_path = report.write_report(str(run_dir), summary)
    console.print_summary(summary)
    console.print_seat_ratings(run_dir)
    print(f"report: {report_path}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from . import webui

    bundle = build_replay_bundle(Path(args.run_dir), args.game)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"{len(bundle['frames'])} frames -> {out}")
        return 0
    webui.run_replay_server(
        bundle, host=args.host, port=args.port, open_browser=not args.no_open
    )
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if args.force:
        for path in (run_dir / "review").glob("*.json"):
            path.unlink()
    if args.force or reviews_missing(run_dir):
        _evaluate_run(run_dir, args.weights)
    summary = report.summarize(str(run_dir))
    report_path = report.write_report(str(run_dir), summary)
    console.print_summary(summary)
    print(f"report: {report_path}")
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    batch_dir = Path(args.batch_dir)
    runs = (
        [batch_dir]
        if (batch_dir / "config.json").exists()
        else sorted(path.parent for path in batch_dir.glob("*/config.json"))
    )
    if not runs:
        print(f"no finished episodes under {batch_dir}")
        return 1

    if args.review:
        for run in runs:
            if reviews_missing(run):
                print(f"reviewing {run.name}")
                _evaluate_run(run, args.weights, progress=False)

    board = report.leaderboard(str(batch_dir))
    console.print_leaderboard(board)
    print(f"leaderboard: {batch_dir / 'leaderboard.json'}")
    return 0


def _cmd_reasoning(args: argparse.Namespace) -> int:
    from jongbench import reasoning as reasoning_module

    run_dir = Path(args.run_dir)
    review_paths = sorted((run_dir / "review").glob("*.json"), key=log_sort_key)
    if not review_paths:
        print(f"no reviews in {run_dir / 'review'}; run `jongbench review` first")
        return 1

    decisions_by_name: dict[str, list[dict[str, Any]]] = {}
    for path in (run_dir / "decisions").glob("*.jsonl"):
        with path.open() as handle:
            decisions_by_name[path.stem] = [
                json.loads(line) for line in handle if line.strip()
            ]
    if not decisions_by_name:
        print(f"no decision logs in {run_dir / 'decisions'} (were all seats random?)")
        return 1

    joined_any = False
    for path in review_paths:
        data = _read_json(path)
        for player_id, player in sorted((data.get("players") or {}).items()):
            name = str(player.get("name", ""))
            records = decisions_by_name.get(
                decision_filename(name).removesuffix(".jsonl")
            )
            if not records:
                continue
            joined = reasoning_module.join(
                records, player.get("review") or {}, player_id=int(player_id)
            )
            if not joined.decisions:
                continue
            joined_any = True
            summary = joined.summary()
            print(f"\n{path.stem} P{player_id} {name}")
            print(
                f"  joined {summary['joined']}/{summary['logged']} decisions "
                f"({summary['coverage']:.1%}); {summary['with_reasoning']} carry reasoning"
            )
            if summary["with_reasoning"]:
                print(
                    f"  reasoning chars: {summary['mean_reasoning_chars']:.0f} mean; "
                    f"{summary['mean_reasoning_chars_when_matching_mortal']:.0f} when "
                    f"agreeing with Mortal, "
                    f"{summary['mean_reasoning_chars_when_not']:.0f} when not"
                )
            for decision in joined.worst(args.worst):
                print(
                    f"    kyoku {decision.kyoku} junme {decision.junme:2d}  "
                    f"played {decision.choice_label[:28]:28s} "
                    f"lost {decision.prob_loss:.2f}"
                )
                if decision.reasoning:
                    snippet = " ".join(decision.reasoning.split())[: args.chars]
                    print(f"      {snippet}")

    if not joined_any:
        print("no decisions could be joined to a review")
        return 1
    return 0


def _cmd_positions(args: argparse.Namespace) -> int:
    import gzip
    import io
    import tempfile

    from jongbench import evaluate as evaluate_module
    from jongbench import positions as positions_module

    engine = evaluate_module.load_engine(args.weights, use_policy=False)
    checkpoint = engine.checkpoint

    logs: list[list[dict[str, Any]]] = []
    artifacts: list[positions_module.SourceArtifact] = []
    for value in args.from_log:
        path = Path(value)
        logs.append(load_mjai_log(path))
        artifacts.append({"name": str(path), "sha256": file_sha256(path)})

    if logs:
        source: positions_module.SourceProvenance = {
            "kind": "mjai_logs",
            "description": "User-provided completed MJAI logs",
            "games": len(logs),
            "artifacts": artifacts,
        }
        source_label = "logs"
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            if args.source == "mortal":
                seats = [
                    positions_module.MortalArenaEngine(f"mortal-{i}", engine)
                    for i in range(4)
                ]
            else:
                seats = [
                    RandomEngine(f"random-{i}", seed=args.seed + i) for i in range(4)
                ]
            arena.run_games(
                seats, args.games, seed_start=(args.seed, 1), log_dir=tempdir
            )
            for path in sorted(Path(tempdir).glob("*.json.gz")):
                logs.append(load_mjai_log(path))
                artifacts.append({"name": path.name, "sha256": file_sha256(path)})
        source = {
            "kind": f"{args.source}_self_play",
            "description": f"Games generated by four {args.source} seats",
            "seed": int(args.seed),
            "games": len(logs),
            "artifacts": artifacts,
        }
        source_label = args.source

    extracted = [
        position
        for events in logs
        for position in positions_module.extract_positions(
            events,
            engine,
            temperature=float(args.temperature),
        )
    ]
    rows = [position.to_task_dict() for position in extracted]
    manifest = positions_module.bank_manifest(
        reviewer_checkpoint=checkpoint.source,
        reviewer_checkpoint_sha256=checkpoint.sha256,
        reviewer_temperature=float(args.temperature),
        source=source,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    try:
        if out.suffix == ".gz":
            with (
                temporary.open("wb") as raw,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0
                ) as compressed,
                io.TextIOWrapper(compressed, encoding="utf-8") as handle,
            ):
                count = positions_module.dump_bank(handle, manifest, rows)
        else:
            with temporary.open("w", encoding="utf-8") as handle:
                count = positions_module.dump_bank(handle, manifest, rows)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"wrote {count} positions from {len(logs)} game(s) ({source_label}) to {out}")
    for line in _bank_baselines(extracted):
        print(line)
    return 0


def _bank_baselines(extracted: Sequence[Any]) -> list[str]:
    """Reference points that make a taskset score on this bank readable: a model's
    q_advantage only means something relative to guessing."""
    if not extracted:
        return []
    count = len(extracted)
    mean_options = sum(len(p.rewards) for p in extracted) / count
    random_reward = sum(sum(p.rewards) / len(p.rewards) for p in extracted) / count
    random_match = sum(1 / len(p.rewards) for p in extracted) / count
    first_reward = sum(p.rewards[0] for p in extracted) / count
    return [
        f"  mean legal options      {mean_options:.1f}",
        f"  uniform-random baseline {random_reward:.3f} reward, {random_match:.1%} match",
        f"  always-first-option     {first_reward:.3f} reward",
        "  Mortal's own choice     1.000 reward by construction",
    ]


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    run_dir = create_run_dir(args.runs_root, "selfcheck")
    specs = ["random", "random", "random", "random"]
    names = engine_names(specs)
    write_run_config(
        run_dir,
        label="selfcheck",
        models=specs,
        names=names,
        games=1,
        seed_start=(777, 1),
        state_hints=False,
    )
    engines = [
        RandomEngine(name, seed=seed)
        for name, seed in zip(names, [1, 2, 3, 4], strict=True)
    ]
    record_gameplay_checkpoints(run_dir, engines)
    summaries = arena.run_games(
        engines,
        1,
        seed_start=(777, 1),
        log_dir=str(run_dir / "logs"),
        disable_progress_bar=True,
    )
    _evaluate_run(run_dir, args.weights, summaries=summaries)
    summary = report.summarize(str(run_dir))
    report_path = report.write_report(str(run_dir), summary)
    console.print_summary(summary)
    print(f"report: {report_path}")
    print("SELFCHECK OK")
    return 0


def _cmd_selfplay(args: argparse.Namespace) -> int:
    from .selfplay import selfplay

    summaries = selfplay(
        weights=args.weights,
        out_dir=args.out,
        games=args.games,
        seed=args.seed,
        batch_games=args.batch_games,
        device=args.device,
        use_policy=args.policy,
        boltzmann_epsilon=args.epsilon,
        boltzmann_temp=args.temp,
    )
    print(f"wrote {len(summaries)} games to {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    stats = train(
        TrainConfig(
            logs=args.logs,
            init=args.init,
            out=args.out,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            lr=args.lr,
            freeze_encoder=not args.unfreeze_encoder,
            file_batch_size=args.file_batch_size,
            teacher_temperature=args.teacher_temperature,
            validation_ratio=args.validation_ratio,
            validation_batches=args.validation_batches,
            data_provenance=args.data_provenance,
            data_sha256=args.data_sha256,
        )
    )
    print(f"train done: {stats}")
    print(f"checkpoint: {args.out}")
    return 0


def _cmd_policy_rl(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_PTS
    from .selfplay import DUPLICATE_PTS
    from .train import PolicyRLConfig, train_policy_rl

    stats = train_policy_rl(
        PolicyRLConfig(
            logs=args.logs,
            init=args.init,
            out=args.out,
            anchor=args.anchor,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            lr=args.lr,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            anchor_kl_weight=args.anchor_kl_weight,
            entropy_weight=args.entropy_weight,
            sampling_temperature=args.sampling_temperature,
            file_batch_size=args.file_batch_size,
            duplicate_challenger_only=args.duplicate_challenger_only,
            pts=(DUPLICATE_PTS if args.duplicate_challenger_only else DEFAULT_PTS),
        )
    )
    print(f"policy RL done: {stats}")
    print(f"checkpoint: {args.out}")
    return 0


def _cmd_improve(args: argparse.Namespace) -> int:
    from .improve import ImproveConfig, improve_policy

    result = improve_policy(
        ImproveConfig(
            init=args.init,
            out_dir=args.out,
            control=args.control,
            rounds=args.rounds,
            rollout_games=args.rollout_games,
            updates=args.updates,
            batch_size=args.batch_size,
            duel_games=args.duel_games,
            device=args.device,
            seed=args.seed,
            lr=args.lr,
            rollout_temperature=args.rollout_temperature,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            anchor_kl_weight=args.anchor_kl_weight,
            entropy_weight=args.entropy_weight,
            promotion_z=args.promotion_z,
            promotion_margin=args.promotion_margin,
        )
    )
    print(
        f"league done: {result['promotions']} promotion(s), "
        f"champion={result['champion']}"
    )
    return 0


def _cmd_duel(args: argparse.Namespace) -> int:
    from .selfplay import duel

    result = duel(
        challenger_weights=args.challenger,
        champion_weights=args.champion,
        games=args.games,
        seed=args.seed,
        device=args.device,
        challenger_policy=args.challenger_policy,
        champion_policy=args.champion_policy,
        log_dir=args.log_dir,
    )
    standard_error = (
        "n/a" if result.standard_error is None else f"{result.standard_error:.3f}"
    )
    print(
        f"challenger rankings {result.rankings} "
        f"avg_rank={result.avg_rank:.4f} avg_pt={result.avg_pt:.2f} "
        f"se={standard_error} games={result.games}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jongbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--models", nargs=4, required=True)
    run.add_argument("--games", type=_positive_int, default=4)
    run.add_argument("--seed", type=_u64, default=10000)
    run.add_argument("--label", default="run")
    run.add_argument("--runs-root", default="runs")
    run.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    run.add_argument("--no-eval", action="store_true")
    run.add_argument("--concurrency", type=_positive_int, default=4)
    run.add_argument("--temperature", type=_temperature, default=0.6)
    run.add_argument(
        "--state-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include engine-derived shanten, waits, and furiten facts in model prompts",
    )
    run.add_argument(
        "--auto-pass-reactions",
        action="store_true",
        help="pass pure chi/pon/kan reactions without a model call (~15%% fewer calls; "
        "the seats never call on others' discards)",
    )
    run.set_defaults(func=_cmd_run)

    watch = subparsers.add_parser("watch")
    watch.add_argument("--models", nargs=4, required=True)
    watch.add_argument("--seed", type=_u64, default=10000)
    watch.add_argument("--label", default="watch")
    watch.add_argument("--runs-root", default="runs")
    watch.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    watch.add_argument("--no-eval", action="store_true")
    watch.add_argument("--delay", type=_nonnegative_float, default=0.4)
    watch.add_argument("--glyphs", action="store_true")
    watch.add_argument("--ui", choices=["term", "web"], default="term")
    watch.add_argument(
        "--state-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include engine-derived shanten, waits, and furiten facts in model prompts",
    )
    watch.set_defaults(func=_cmd_watch)

    replay_cmd = subparsers.add_parser("replay")
    replay_cmd.add_argument("run_dir")
    replay_cmd.add_argument("--game")
    replay_cmd.add_argument(
        "--out", help="write the bundle as JSON instead of serving it"
    )
    replay_cmd.add_argument("--host", default="127.0.0.1")
    replay_cmd.add_argument("--port", type=int, default=8642)
    replay_cmd.add_argument("--no-open", action="store_true")
    replay_cmd.set_defaults(func=_cmd_replay)

    review_cmd = subparsers.add_parser("review")
    review_cmd.add_argument("run_dir")
    review_cmd.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    review_cmd.add_argument("--force", action="store_true")
    review_cmd.set_defaults(func=_cmd_review)

    leaderboard_cmd = subparsers.add_parser("leaderboard")
    leaderboard_cmd.add_argument("batch_dir")
    leaderboard_cmd.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    leaderboard_cmd.add_argument("--review", action="store_true")
    leaderboard_cmd.set_defaults(func=_cmd_leaderboard)

    selfcheck = subparsers.add_parser("selfcheck")
    selfcheck.add_argument("--runs-root", default="runs")
    selfcheck.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    selfcheck.set_defaults(func=_cmd_selfcheck)

    reasoning_cmd = subparsers.add_parser("reasoning")
    reasoning_cmd.add_argument("run_dir")
    reasoning_cmd.add_argument("--worst", type=int, default=5)
    reasoning_cmd.add_argument("--chars", type=int, default=300)
    reasoning_cmd.set_defaults(func=_cmd_reasoning)

    positions_cmd = subparsers.add_parser("positions")
    positions_cmd.add_argument("--out", default="bank.jsonl")
    positions_cmd.add_argument("--games", type=int, default=1)
    positions_cmd.add_argument("--seed", type=int, default=20260101)
    positions_cmd.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    positions_cmd.add_argument("--temperature", type=_nonnegative_float, default=0.1)
    positions_cmd.add_argument(
        "--from-log",
        action="append",
        default=[],
        help="grade an existing mjai log instead of generating games",
    )
    positions_cmd.add_argument(
        "--source",
        choices=["random", "mortal"],
        default="mortal",
        help="engine used to generate boards; mortal gives positions a strong player "
        "would actually face",
    )
    positions_cmd.set_defaults(func=_cmd_positions)

    selfplay_cmd = subparsers.add_parser(
        "selfplay", help="generate Mortal self-play logs for training"
    )
    selfplay_cmd.add_argument("--weights", default=AUTO_MORTAL_WEIGHTS)
    selfplay_cmd.add_argument("--out", default="training/selfplay")
    selfplay_cmd.add_argument("--games", type=_positive_int, default=256)
    selfplay_cmd.add_argument("--seed", type=_u64, default=10000)
    selfplay_cmd.add_argument("--batch-games", type=_positive_int, default=32)
    selfplay_cmd.add_argument("--device", default="auto")
    selfplay_cmd.add_argument("--epsilon", type=float, default=0.0)
    selfplay_cmd.add_argument("--temp", type=float, default=0.2)
    selfplay_cmd.add_argument(
        "--policy",
        action="store_true",
        help="play with the checkpoint policy head instead of Q-argmax",
    )
    selfplay_cmd.set_defaults(func=_cmd_selfplay)

    train_cmd = subparsers.add_parser(
        "train", help="train a policy/value reviewer from mjai logs"
    )
    train_cmd.add_argument("--logs", default="training/selfplay")
    train_cmd.add_argument("--init", default=AUTO_MORTAL_WEIGHTS)
    train_cmd.add_argument("--out", default="weights/reviewer.pth")
    train_cmd.add_argument("--steps", type=_positive_int, default=4000)
    train_cmd.add_argument("--batch-size", type=_positive_int, default=256)
    train_cmd.add_argument("--device", default="auto")
    train_cmd.add_argument("--lr", type=float, default=3e-4)
    train_cmd.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="finetune Mortal's ResNet (off by default; easy to wreck Q)",
    )
    train_cmd.add_argument("--file-batch-size", type=_positive_int, default=4)
    train_cmd.add_argument("--teacher-temperature", type=float, default=0.1)
    train_cmd.add_argument("--validation-ratio", type=float, default=0.1)
    train_cmd.add_argument("--validation-batches", type=_positive_int, default=20)
    train_cmd.add_argument("--data-provenance")
    train_cmd.add_argument("--data-sha256")
    train_cmd.set_defaults(func=_cmd_train)

    policy_rl_cmd = subparsers.add_parser(
        "policy-rl", help="update a policy head from its stochastic self-play logs"
    )
    policy_rl_cmd.add_argument("--logs", required=True)
    policy_rl_cmd.add_argument("--init", required=True)
    policy_rl_cmd.add_argument("--out", required=True)
    policy_rl_cmd.add_argument("--anchor")
    policy_rl_cmd.add_argument("--steps", type=_positive_int, default=128)
    policy_rl_cmd.add_argument("--batch-size", type=_positive_int, default=512)
    policy_rl_cmd.add_argument("--device", default="auto")
    policy_rl_cmd.add_argument("--lr", type=float, default=1e-4)
    policy_rl_cmd.add_argument("--clip-ratio", type=float, default=0.2)
    policy_rl_cmd.add_argument("--target-kl", type=float, default=0.03)
    policy_rl_cmd.add_argument("--anchor-kl-weight", type=float, default=0.02)
    policy_rl_cmd.add_argument("--entropy-weight", type=float, default=0.001)
    policy_rl_cmd.add_argument("--sampling-temperature", type=float, default=1.0)
    policy_rl_cmd.add_argument("--file-batch-size", type=_positive_int, default=8)
    policy_rl_cmd.add_argument(
        "--duplicate-challenger-only",
        action="store_true",
        help="train only the challenger POV from OneVsThree a/b/c/d logs",
    )
    policy_rl_cmd.set_defaults(func=_cmd_policy_rl)

    improve_cmd = subparsers.add_parser(
        "improve",
        help="iterate stochastic self-play, policy updates, and duplicate gating",
    )
    improve_cmd.add_argument("--init", required=True)
    improve_cmd.add_argument("--out", required=True)
    improve_cmd.add_argument("--control", default=AUTO_MORTAL_WEIGHTS)
    improve_cmd.add_argument(
        "--no-control", action="store_const", dest="control", const=None
    )
    improve_cmd.add_argument("--rounds", type=_positive_int, default=4)
    improve_cmd.add_argument("--rollout-games", type=_positive_int, default=256)
    improve_cmd.add_argument("--updates", type=_positive_int, default=128)
    improve_cmd.add_argument("--batch-size", type=_positive_int, default=512)
    improve_cmd.add_argument("--duel-games", type=_positive_int, default=512)
    improve_cmd.add_argument("--device", default="auto")
    improve_cmd.add_argument("--seed", type=_u64, default=20270000)
    improve_cmd.add_argument("--lr", type=float, default=1e-4)
    improve_cmd.add_argument("--rollout-temperature", type=float, default=1.0)
    improve_cmd.add_argument("--clip-ratio", type=float, default=0.2)
    improve_cmd.add_argument("--target-kl", type=float, default=0.03)
    improve_cmd.add_argument("--anchor-kl-weight", type=float, default=0.02)
    improve_cmd.add_argument("--entropy-weight", type=float, default=0.001)
    improve_cmd.add_argument("--promotion-z", type=float, default=1.0)
    improve_cmd.add_argument("--promotion-margin", type=float, default=0.0)
    improve_cmd.set_defaults(func=_cmd_improve)

    duel_cmd = subparsers.add_parser(
        "duel", help="1-vs-3 duplicate match between two checkpoints"
    )
    duel_cmd.add_argument("--challenger", required=True)
    duel_cmd.add_argument("--champion", default=AUTO_MORTAL_WEIGHTS)
    duel_cmd.add_argument("--games", type=_positive_int, default=64)
    duel_cmd.add_argument("--seed", type=_u64, default=20000)
    duel_cmd.add_argument("--device", default="auto")
    duel_cmd.add_argument(
        "--challenger-policy",
        action="store_true",
        help="challenger plays with its policy head",
    )
    duel_cmd.add_argument(
        "--champion-policy",
        action="store_true",
        help="champion plays with its policy head",
    )
    duel_cmd.add_argument("--log-dir")
    duel_cmd.set_defaults(func=_cmd_duel)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _is_human_spec(spec: str) -> bool:
    return providers.parse_spec(spec).provider == "human"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _u64(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**64 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 2^64 - 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _temperature(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed > 2:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return parsed
