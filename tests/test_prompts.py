import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jongbench
import libriichi

from jongbench.prompts import (
    _melds_by_player,
    _scores_and_kyotaku,
    build_user_prompt,
    extract_choice,
)


def assert_raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_prompt_contract():
    st = libriichi.state.PlayerState(0)
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_marker": "3p",
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                [
                    "1m",
                    "2m",
                    "3m",
                    "4p",
                    "5p",
                    "6p",
                    "7s",
                    "8s",
                    "9s",
                    "E",
                    "E",
                    "5s",
                    "6s",
                ],
                [
                    "4m",
                    "5m",
                    "6m",
                    "7m",
                    "8m",
                    "9m",
                    "1p",
                    "2p",
                    "4p",
                    "5m",
                    "6m",
                    "7m",
                    "8m",
                ],
                [
                    "1s",
                    "2s",
                    "3s",
                    "4s",
                    "5s",
                    "6s",
                    "7p",
                    "8p",
                    "9p",
                    "S",
                    "S",
                    "W",
                    "W",
                ],
                [
                    "N",
                    "N",
                    "P",
                    "P",
                    "F",
                    "F",
                    "C",
                    "C",
                    "1m",
                    "2m",
                    "3m",
                    "7p",
                    "8p",
                ],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
    ]
    for ev in events:
        st.update(json.dumps(ev))

    assert st.last_cans.can_discard is True

    menu = [
        {"label": "Discard 1m", "event": {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}, "kind": "discard"},
        {"label": "Discard 0p", "event": {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True}, "kind": "discard"},
        {"label": "Discard E", "event": {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False}, "kind": "discard"},
    ]

    raw_prompt = build_user_prompt(0, st, events, menu, state_hints=False)
    assert "Engine-derived state hints" not in raw_prompt
    assert "[after:" not in raw_prompt

    p = build_user_prompt(0, st, events, menu, state_hints=True)
    assert "5p(red)" in p
    assert "0p" not in p
    assert "Your concealed hand (14 tiles; 0 completed melds)" in p
    assert "Engine-derived state hints" in p
    assert "[after:" in p
    assert "Dora" in p
    assert "{\"choice\"" in p or "choice" in p
    assert "5p(red)" in build_user_prompt(
        0,
        st,
        events,
        [
            {
                "label": "chi 3p with 4p 0p",
                "event": {"type": "none"},
                "kind": "chi",
            }
        ],
    )
    shanten, waits, furiten = st.reaction_summary(json.dumps(menu[0]["event"]))
    assert isinstance(shanten, int)
    assert len(waits) == 34
    assert isinstance(furiten, bool)
    assert extract_choice('I think...\n{"choice": 2}', 3) == 2
    assert extract_choice('{"choice": 0}\n', 3) == 0
    assert_raises(lambda: extract_choice("garbage", 3), ValueError)
    assert_raises(lambda: extract_choice('{"choice": 9}', 3), ValueError)
    assert_raises(
        lambda: extract_choice('{"choice": 1}\n{"choice": 2}', 3),
        ValueError,
    )

    scores, kyotaku = _scores_and_kyotaku(
        events[0],
        [*events, {"type": "reach_accepted", "actor": 2}],
    )
    assert scores == [25000, 25000, 24000, 25000]
    assert kyotaku == 1

    melds = _melds_by_player(
        [
            {
                "type": "pon",
                "actor": 1,
                "target": 0,
                "pai": "5m",
                "consumed": ["5mr", "5m"],
            },
            {
                "type": "kakan",
                "actor": 1,
                "pai": "5m",
                "consumed": ["5mr", "5m", "5m"],
            },
        ]
    )
    assert len(melds[1]) == 1
    assert melds[1][0].startswith("kakan ")
    print("OK")


if __name__ == "__main__":
    test_prompt_contract()
