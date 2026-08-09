from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import arena, bridge, providers


def test_newest_user_text_reads_plain_and_block_content() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": '{"choice": 0}'},
        {"role": "user", "content": providers.cacheable("newest")},
    ]
    assert bridge.newest_user_text(messages) == "newest"

    messages[-1] = {"role": "user", "content": "plain newest"}
    assert bridge.newest_user_text(messages) == "plain newest"


def test_callback_provider_forwards_only_the_newest_turn() -> None:
    seen: list[str] = []
    provider = bridge.CallbackProvider(lambda text: seen.append(text) or '{"choice": 1}')

    result = provider.complete(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": '{"choice": 0}'},
            {"role": "user", "content": providers.cacheable("new")},
        ]
    )

    # The consumer owns the transcript, so resending history would duplicate it.
    assert seen == ["new"]
    assert isinstance(result, providers.Completion)
    assert result.text == '{"choice": 1}'


def test_a_bridged_seat_plays_a_real_hanchan() -> None:
    asked = {i: 0 for i in range(4)}

    def ask_from(index: int):
        def ask(prompt: str) -> str:
            asked[index] += 1
            return '{"choice": 0}'

        return ask

    seats = [bridge.make_bridged_engine(f"seat{i}", ask_from(i)) for i in range(4)]
    assert all(seat.concurrency == 1 for seat in seats)

    summaries = arena.run_games(seats, 1, seed_start=(99, 1), log_dir=None)

    assert len(summaries) == 1
    assert sum(summaries[0].scores) == 100000
    assert sorted(summaries[0].placements.values()) == [1, 2, 3, 4]
    assert all(count > 50 for count in asked.values()), asked


def test_a_bridged_seat_needs_no_api_key() -> None:
    """Construction must not touch a provider client: the callback is the transport."""
    engine = bridge.make_bridged_engine("seat", lambda prompt: '{"choice": 0}')
    assert isinstance(engine.provider, bridge.CallbackProvider)
    assert engine.conversational is True
