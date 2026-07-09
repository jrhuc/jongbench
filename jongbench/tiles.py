from __future__ import annotations

TILES: list[str] = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
]

_TILE_TO_LABEL = {tile: i for i, tile in enumerate(TILES)}
_AKA_TO_BASE = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}
_BASE_TO_AKA = {"5m": "5mr", "5p": "5pr", "5s": "5sr"}
_HONORS = ("E", "S", "W", "N", "P", "F", "C")
_GLYPHS = {
    "E": "\U0001f000",
    "S": "\U0001f001",
    "W": "\U0001f002",
    "N": "\U0001f003",
    "C": "\U0001f004",
    "F": "\U0001f005",
    "P": "\U0001f006",
}
_GLYPHS.update({f"{n}m": chr(0x1F006 + n) for n in range(1, 10)})
_GLYPHS.update({f"{n}s": chr(0x1F00F + n) for n in range(1, 10)})
_GLYPHS.update({f"{n}p": chr(0x1F018 + n) for n in range(1, 10)})


def tile_to_label(s: str) -> int:
    try:
        return _TILE_TO_LABEL[s]
    except KeyError as exc:
        raise ValueError(f"unknown tile: {s}") from exc


def label_to_tile(i: int) -> str:
    if i < 0:
        raise ValueError(f"unknown tile label: {i}")
    try:
        return TILES[i]
    except IndexError as exc:
        raise ValueError(f"unknown tile label: {i}") from exc


def deaka(s: str) -> str:
    return _AKA_TO_BASE.get(s, s)


def akaize(s: str) -> str:
    return _BASE_TO_AKA.get(s, s)


def is_aka(s: str) -> bool:
    return s in _AKA_TO_BASE


def fmt_tile(s: str, glyphs: bool = False) -> str:
    if glyphs:
        try:
            return _GLYPHS[deaka(s)]
        except KeyError as exc:
            raise ValueError(f"unknown tile: {s}") from exc
    if is_aka(s):
        return "0" + s[1]
    tile_to_label(s)
    return s


def sort_tiles(tiles: list[str]) -> list[str]:
    return sorted(tiles, key=lambda tile: (tile_to_label(deaka(tile)), is_aka(tile)))


def fmt_hand(tiles: list[str], glyphs: bool = False) -> str:
    if glyphs:
        groups: list[list[str]] = [[], [], [], []]
        for tile in sort_tiles(tiles):
            base = deaka(tile)
            if base in _HONORS:
                groups[3].append(fmt_tile(tile, glyphs=True))
            else:
                groups["mps".index(base[1])].append(fmt_tile(tile, glyphs=True))
        return " ".join("".join(group) for group in groups if group)

    suit_counts = {suit: {str(n): 0 for n in range(1, 10)} for suit in "mps"}
    aka_counts = {suit: 0 for suit in "mps"}
    honor_counts = {honor: 0 for honor in _HONORS}

    for tile in tiles:
        base = deaka(tile)
        tile_to_label(tile)
        if base in _HONORS:
            honor_counts[base] += 1
            continue
        num, suit = base
        if is_aka(tile):
            aka_counts[suit] += 1
        else:
            suit_counts[suit][num] += 1

    groups: list[str] = []
    for suit in "mps":
        nums = []
        for n in range(1, 10):
            digit = str(n)
            if n == 5:
                nums.append("0" * aka_counts[suit])
            nums.append(digit * suit_counts[suit][digit])
        digits = "".join(nums)
        if digits:
            groups.append(digits + suit)

    honors = "".join(honor * honor_counts[honor] for honor in _HONORS)
    if honors:
        groups.append(honors)
    return " ".join(groups)


def tiles_from_counts(tehai: list[int], akas: list[bool]) -> list[str]:
    if len(tehai) != 34:
        raise ValueError(f"expected 34 tile counts, got {len(tehai)}")
    if len(akas) != 3:
        raise ValueError(f"expected 3 aka flags, got {len(akas)}")

    tiles: list[str] = []
    for i, count in enumerate(tehai):
        if count < 0:
            raise ValueError(f"negative tile count at label {i}: {count}")
        tile = TILES[i]
        if i in (4, 13, 22):
            aka_idx = (i - 4) // 9
            if akas[aka_idx]:
                if count == 0:
                    raise ValueError(f"aka flag set with no {tile} in hand")
                tiles.extend([tile] * (count - 1))
                tiles.append(akaize(tile))
            else:
                tiles.extend([tile] * count)
        else:
            tiles.extend([tile] * count)
    return tiles
