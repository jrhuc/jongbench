from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import arena, evaluate, providers, report
from .arena import GameSummary
from .artifacts import decision_filename
from .engines import RandomEngine, TerminalHumanIO, make_engine
from .spectator import Spectator, TerminalRenderer


def _new_run_dir(runs_root: str | Path, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-._")
    safe = safe or "run"
    root = Path(runs_root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{stamp}-{safe}"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = root / f"{stamp}-{safe}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _engine_names(specs: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    names = []
    for spec in specs:
        parsed = providers.parse_spec(spec)
        base = parsed.display_name
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return names


def _decision_sink(path: str | Path) -> Callable[[dict[str, Any]], None]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def sink(record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with lock:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    return sink


def _write_config(
    run_dir: str | Path,
    label: str,
    specs: Sequence[str],
    names: Sequence[str],
    games: int,
    seed_start: tuple[int, int],
    state_hints: bool,
) -> None:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    data = {
        "label": label,
        "created": datetime.now(timezone.utc).isoformat(),
        "models": list(specs),
        "names": list(names),
        "games": int(games),
        "seed_start": [int(seed_start[0]), int(seed_start[1])],
        "state_hints": bool(state_hints),
    }
    (run / "config.json").write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _evaluate_run(
    run_dir: str | Path,
    weights: str | Path,
    temperature: float = 0.1,
    progress: bool = True,
    summaries: Sequence[GameSummary]
    | Mapping[tuple[int, int], GameSummary]
    | None = None,
) -> None:
    run = Path(run_dir)
    log_paths = sorted((run / "logs").glob("*.json.gz"), key=_log_sort_key)
    if not log_paths:
        raise ValueError(f"no logs found in {run / 'logs'}")

    summary_by_seed = _summary_by_seed(summaries)
    mortal = evaluate.load_engine(str(weights))
    review_dir = run / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    for log_path in log_paths:
        events = evaluate.load_mjai_log(str(log_path))
        seed = _seed_from_events_or_path(events, log_path)
        game_summary = summary_by_seed.get(seed) or _reconstruct_summary(
            events, log_path
        )
        reviews = evaluate.review_game(events, mortal, temperature=temperature)
        players: dict[str, dict[str, Any]] = {}

        for player_id in range(4):
            player_review = reviews[player_id]
            players[str(player_id)] = {
                "name": game_summary.names[player_id],
                "review": player_review,
                "aggregates": evaluate.aggregates(player_review),
            }
            if progress:
                rating = float(player_review.get("rating") or 0.0) * 100.0
                print(
                    f"review {_seed_label(seed)} P{player_id} "
                    f"{game_summary.names[player_id]} rating {rating:.2f}"
                )

        payload = {
            "seed": [seed[0], seed[1]],
            "names": list(game_summary.names),
            "scores": list(game_summary.scores),
            "placements": dict(game_summary.placements),
            "players": players,
        }
        (review_dir / f"{_seed_label(seed)}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def _cmd_run(args: argparse.Namespace) -> int:
    specs = list(args.models)
    if any(_is_human_spec(spec) for spec in specs):
        raise ValueError("run does not support human seats")

    run_dir = _new_run_dir(args.runs_root, args.label)
    _prepare_run_dir(run_dir)
    names = _engine_names(specs)
    _write_config(
        run_dir,
        args.label,
        specs,
        names,
        int(args.games),
        (int(args.seed), 1),
        bool(args.state_hints),
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
    _print_summary(summary)
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

    run_dir = _new_run_dir(args.runs_root, args.label)
    _prepare_run_dir(run_dir)
    names = _engine_names(specs)
    _write_config(
        run_dir,
        args.label,
        specs,
        names,
        1,
        (int(args.seed), 1),
        bool(args.state_hints),
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
    _print_summary(summary)
    _print_seat_ratings(run_dir)
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
    if args.force or _reviews_missing(run_dir):
        _evaluate_run(run_dir, args.weights)
    summary = report.summarize(str(run_dir))
    report_path = report.write_report(str(run_dir), summary)
    _print_summary(summary)
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
            if _reviews_missing(run):
                print(f"reviewing {run.name}")
                _evaluate_run(run, args.weights, progress=False)

    board = report.leaderboard(str(batch_dir))
    _print_leaderboard(board)
    print(f"leaderboard: {batch_dir / 'leaderboard.json'}")
    return 0


def _cmd_reasoning(args: argparse.Namespace) -> int:
    from jongbench import reasoning as reasoning_module

    run_dir = Path(args.run_dir)
    review_paths = sorted((run_dir / "review").glob("*.json"), key=_log_sort_key)
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
    import tempfile

    from jongbench import evaluate as evaluate_module
    from jongbench import positions as positions_module

    engine = evaluate_module.load_engine(args.weights)

    logs: list[list[dict]] = []
    for path in args.from_log:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            logs.append([json.loads(line) for line in handle if line.strip()])

    if not logs:
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
                with gzip.open(path, "rt") as handle:
                    logs.append([json.loads(line) for line in handle if line.strip()])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    extracted: list[Any] = []
    with out.open("w") as handle:
        for events in logs:
            for position in positions_module.extract_positions(events, engine):
                handle.write(
                    json.dumps(position.to_dict(), separators=(",", ":")) + "\n"
                )
                extracted.append(position)

    source = "logs" if args.from_log else args.source
    print(
        f"wrote {len(extracted)} positions from {len(logs)} game(s) ({source}) to {out}"
    )
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
    run_dir = _new_run_dir(args.runs_root, "selfcheck")
    _prepare_run_dir(run_dir)
    specs = ["random", "random", "random", "random"]
    names = _engine_names(specs)
    _write_config(run_dir, "selfcheck", specs, names, 1, (777, 1), False)
    engines = [
        RandomEngine(name, seed=seed)
        for name, seed in zip(names, [1, 2, 3, 4], strict=True)
    ]
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
    _print_summary(summary)
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
    run.add_argument("--weights", default="weights/mortal.pth")
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
    watch.add_argument("--weights", default="weights/mortal.pth")
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
    review_cmd.add_argument("--weights", default="weights/mortal.pth")
    review_cmd.add_argument("--force", action="store_true")
    review_cmd.set_defaults(func=_cmd_review)

    leaderboard_cmd = subparsers.add_parser("leaderboard")
    leaderboard_cmd.add_argument("batch_dir")
    leaderboard_cmd.add_argument("--weights", default="weights/mortal.pth")
    leaderboard_cmd.add_argument("--review", action="store_true")
    leaderboard_cmd.set_defaults(func=_cmd_leaderboard)

    selfcheck = subparsers.add_parser("selfcheck")
    selfcheck.add_argument("--runs-root", default="runs")
    selfcheck.add_argument("--weights", default="weights/mortal.pth")
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
    positions_cmd.add_argument("--weights", default="weights/mortal.pth")
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
    selfplay_cmd.add_argument("--weights", default="weights/mortal.pth")
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
    train_cmd.add_argument("--init", default="weights/mortal.pth")
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
    improve_cmd.add_argument("--control", default="weights/mortal.pth")
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
    duel_cmd.add_argument("--champion", default="weights/mortal.pth")
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


def _prepare_run_dir(run_dir: Path) -> None:
    for name in ("logs", "decisions", "review"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _is_human_spec(spec: str) -> bool:
    return providers.parse_spec(spec).provider == "human"


def _summary_by_seed(
    summaries: Sequence[GameSummary] | Mapping[tuple[int, int], GameSummary] | None,
) -> dict[tuple[int, int], GameSummary]:
    if summaries is None:
        return {}
    if isinstance(summaries, Mapping):
        return {tuple(key): value for key, value in summaries.items()}
    return {tuple(summary.seed): summary for summary in summaries}


def _reconstruct_summary(events: list[dict[str, Any]], path: Path) -> GameSummary:
    seed = _seed_from_events_or_path(events, path)
    start_game = next(
        (event for event in events if event.get("type") == "start_game"), {}
    )
    names = [str(name) for name in start_game.get("names") or []]
    if len(names) != 4:
        names = [f"P{seat}" for seat in range(4)]

    last_scores = [25000, 25000, 25000, 25000]
    last_start_index = -1
    for index, event in enumerate(events):
        if event.get("type") != "start_kyoku":
            continue
        scores = event.get("scores")
        if isinstance(scores, list) and len(scores) == 4:
            last_scores = [int(score) for score in scores]
            last_start_index = index

    scores = list(last_scores)
    for event in events[last_start_index + 1 :]:
        event_type = event.get("type")
        if event_type == "reach_accepted":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                scores[actor] -= 1000
            continue
        if event_type not in {"hora", "ryukyoku"}:
            continue
        deltas = event.get("deltas")
        if isinstance(deltas, list) and len(deltas) == 4:
            scores = [
                score + int(delta) for score, delta in zip(scores, deltas, strict=True)
            ]

    if any(event.get("type") == "end_game" for event in events):
        outstanding_kyotaku = 100000 - sum(scores)
        if outstanding_kyotaku > 0:
            leader = min(range(4), key=lambda seat: (-scores[seat], seat))
            scores[leader] += outstanding_kyotaku

    return GameSummary(
        seed=seed, names=names, scores=scores, placements=_placements(names, scores)
    )


def _placements(names: Sequence[str], scores: Sequence[int]) -> dict[str, int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    return {str(names[seat]): rank + 1 for rank, seat in enumerate(order)}


def _reviews_missing(run_dir: Path) -> bool:
    logs = sorted((run_dir / "logs").glob("*.json.gz"), key=_log_sort_key)
    if not logs:
        raise ValueError(f"no logs found in {run_dir / 'logs'}")
    for log_path in logs:
        seed = _seed_from_path(log_path)
        if not (run_dir / "review" / f"{_seed_label(seed)}.json").exists():
            return True
    return False


def _seed_from_events_or_path(
    events: list[dict[str, Any]],
    path: Path,
) -> tuple[int, int]:
    for event in events:
        if event.get("type") != "start_game":
            continue
        seed = event.get("seed")
        if isinstance(seed, list | tuple) and len(seed) >= 2:
            return int(seed[0]), int(seed[1])
    return _seed_from_path(path)


def _seed_from_path(path: Path) -> tuple[int, int]:
    name = path.name
    if name.endswith(".json.gz"):
        stem = name[:-8]
    elif name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"cannot parse seed from {path.name}")


def _seed_label(seed: tuple[int, int]) -> str:
    return f"{int(seed[0])}_{int(seed[1])}"


def _log_sort_key(path: Path) -> tuple[int, int, str]:
    try:
        seed = _seed_from_path(path)
    except ValueError:
        return 0, 0, path.name
    return seed[0], seed[1], path.name


def _select_log(run_dir: Path, game: str | None) -> Path:
    logs = sorted((run_dir / "logs").glob("*.json.gz"), key=_log_sort_key)
    if not logs:
        raise ValueError(f"no logs found in {run_dir / 'logs'}")
    if game is None:
        return logs[0]
    target = game[:-8] if game.endswith(".json.gz") else game
    target = target[:-5] if target.endswith(".json") else target
    for path in logs:
        if _seed_label(_seed_from_path(path)) == target:
            return path
    raise ValueError(f"game not found: {game}")


def build_replay_bundle(run_dir: Path, game: str | None = None) -> dict[str, Any]:
    """One game as the web replay viewer wants it: every mjai event paired with the
    table snapshot after it, plus standings and the Mortal review when present."""
    from .spectator import TableState

    log_path = _select_log(run_dir, game)
    events = evaluate.load_mjai_log(str(log_path))
    seed = _seed_from_events_or_path(events, log_path)
    review_path = run_dir / "review" / f"{_seed_label(seed)}.json"
    review_data = _read_json(review_path) if review_path.exists() else None

    table = TableState()
    frames = []
    for seq, event in enumerate(events, start=1):
        table.apply(event)
        frames.append({"seq": seq, "event": event, "snapshot": table.snapshot()})

    if isinstance(review_data, dict):
        names = [str(name) for name in review_data.get("names") or []]
        scores = [int(score) for score in review_data.get("scores") or []]
        placements = dict(review_data.get("placements") or {})
    else:
        summary = _reconstruct_summary(events, log_path)
        names, scores, placements = summary.names, summary.scores, summary.placements

    bundle: dict[str, Any] = {
        "game": _seed_label(seed),
        "seed": [seed[0], seed[1]],
        "names": names,
        "scores": scores,
        "placements": placements,
        "frames": frames,
    }
    if isinstance(review_data, dict):
        bundle["review"] = review_data
    return bundle


def _print_summary(summary: dict[str, Any]) -> None:
    engines = list(summary.get("leaderboard") or [])
    headers = [
        "engine",
        "games",
        "place",
        "score",
        "rating",
        "match",
        "fallbacks",
        "cost",
    ]
    rows = []
    for engine in engines:
        fallback_rate = engine.get("fallback_rate")
        if fallback_rate is None:
            fallbacks = "n/a"
        else:
            fallbacks = f"{float(fallback_rate) * 100:.1f}% ({int(engine.get('fallback_count') or 0)})"
        rows.append(
            [
                str(engine.get("name") or ""),
                str(int(engine.get("games") or 0)),
                f"{float(engine.get('avg_placement') or 0.0):.2f}",
                f"{float(engine.get('avg_score') or 0.0):.0f}",
                f"{float(engine.get('mean_rating') or 0.0) * 100:.2f}",
                f"{float(engine.get('match_rate') or 0.0) * 100:.1f}%",
                fallbacks,
                _cost(engine.get("cost")),
            ]
        )

    _print_table(headers, rows)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _print_leaderboard(board: dict[str, Any]) -> None:
    print(
        f"{board['episode_count']} episode(s), {board['reviewed_count']} reviewed"
        f" -> {board['batch_dir']}"
    )
    headers = [
        "#",
        "spec",
        "eps",
        "place",
        "1/2/3/4",
        "score",
        "rating",
        "match",
        "fallbacks",
        "cost",
    ]
    rows = []
    for engine in board.get("leaderboard") or []:
        fallback_rate = engine.get("fallback_rate")
        fallbacks = (
            "n/a"
            if fallback_rate is None
            else f"{float(fallback_rate) * 100:.1f}% ({int(engine.get('fallback_count') or 0)})"
        )
        reviewed = int(engine.get("total_reviewed") or 0)
        rows.append(
            [
                str(engine.get("rank") or ""),
                str(engine.get("spec") or ""),
                str(int(engine.get("episodes") or 0)),
                f"{float(engine.get('avg_placement') or 0.0):.2f}",
                "/".join(str(count) for count in engine.get("placement_counts") or []),
                f"{float(engine.get('avg_score') or 0.0):.0f}",
                f"{float(engine.get('mean_rating') or 0.0) * 100:.2f}"
                if reviewed
                else "n/a",
                f"{float(engine.get('match_rate') or 0.0) * 100:.1f}%"
                if reviewed
                else "n/a",
                fallbacks,
                _cost(engine.get("cost")),
            ]
        )
    _print_table(headers, rows)


def _cost(value: Any) -> str:
    # Only metering providers report a cost; a local or unmetered seat has none.
    return "n/a" if value is None else f"${float(value):.4f}"


def _print_seat_ratings(run_dir: Path) -> None:
    paths = sorted((run_dir / "review").glob("*.json"), key=_log_sort_key)
    if not paths:
        return
    data = _read_json(paths[0])
    players = data.get("players") or {}
    print("seat ratings")
    for seat in range(4):
        player = players.get(str(seat)) or players.get(seat) or {}
        review_data = player.get("review") or {}
        rating = float(review_data.get("rating") or 0.0) * 100.0
        print(f"P{seat} {str(player.get('name') or f'P{seat}'):<24} {rating:6.2f}")


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
