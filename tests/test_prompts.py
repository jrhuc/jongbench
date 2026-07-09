import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jongbench
import libriichi

from jongbench.prompts import build_user_prompt, extract_choice


def assert_raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def main():
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

    p = build_user_prompt(0, st, events, menu)
    assert "0p" in p or "5pr" in p
    assert "Dora" in p
    assert "{\"choice\"" in p or "choice" in p
    assert extract_choice('I think...\n{"choice": 2}', 3) == 2
    assert extract_choice('{"choice": 0}\n', 3) == 0
    assert_raises(lambda: extract_choice("garbage", 3), ValueError)
    assert_raises(lambda: extract_choice('{"choice": 9}', 3), ValueError)
    print("OK")


if __name__ == "__main__":
    main()
