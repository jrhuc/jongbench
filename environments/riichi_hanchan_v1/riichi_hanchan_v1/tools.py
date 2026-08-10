"""Board-query tools for a hanchan seat: programs over data instead of tokens over data.

In tools mode the seat's turns are minimal — delta and menu, no inline state hints — and
these tools answer the questions the hints used to pre-answer, when the model actually
asks them. The framework launches this module as its own process (`python -m
riichi_hanchan_v1.tools`), so it cannot reach the live Rust game state; every answer is
precomputed by the engine at the decision point (`prompts.decision_snapshot`) and synced
through the rollout's state channel. Notes are the one thing that outlives a kyoku: the
env carries them from one per-kyoku rollout to the next, giving the model a memory it
must manage itself.
"""

from __future__ import annotations

import verifiers.v1 as vf

MAX_NOTES = 40
MAX_NOTE_CHARS = 500


class SeatState(vf.State):
    board: str = ""
    discards: dict[str, str] = {}
    waits: str = ""
    simulate: dict[str, str] = {}
    notes: list[str] = []


def normalize_tile(raw: str) -> str:
    """Map the model's tile spelling onto the snapshot's (`5p(red)` for reds, `E` for
    honors), accepting the common variants: 0p, 5pr, lowercase honors."""
    tile = raw.strip().replace(" ", "")
    if len(tile) == 1:
        return tile.upper()
    red = False
    lowered = tile.lower()
    if lowered.endswith("(red)"):
        red, lowered = True, lowered[: -len("(red)")]
    if len(lowered) == 3 and lowered.endswith("r"):
        red, lowered = True, lowered[:2]
    if len(lowered) == 2 and lowered[0] == "0":
        red, lowered = True, "5" + lowered[1]
    return f"{lowered}(red)" if red else lowered


class SeatToolset(vf.Toolset[vf.ToolsetConfig, SeatState]):
    TOOL_PREFIX = None

    @vf.tool
    def board(self) -> str:
        """The full table as it stands at your current decision: round, scores, dora,
        every player's melds and discards, and your hand."""
        return self.state.board or "no decision is pending"

    @vf.tool
    def discards(self, player: int) -> str:
        """One player's discard row (0-3), oldest first, with riichi status. A discard
        marked ' was drawn and immediately discarded; * is the riichi declaration."""
        row = self.state.discards.get(f"P{player}")
        if row is None:
            return "player must be 0, 1, 2 or 3"
        return f"P{player}: {row}"

    @vf.tool
    def waits(self) -> str:
        """Your own shape right now: shanten, the tiles that complete you when tenpai,
        and your furiten status."""
        return self.state.waits or "no decision is pending"

    @vf.tool
    def simulate(self, tile: str) -> str:
        """The shape after discarding `tile` from your hand at this decision: resulting
        shanten, waits and furiten. Only tiles you may legally discard now can be
        simulated."""
        if not self.state.simulate:
            return "no discard is available at this decision"
        key = normalize_tile(tile)
        answer = self.state.simulate.get(key)
        if answer is None:
            legal = " ".join(sorted(self.state.simulate))
            return f"{tile} is not a legal discard now; legal discards: {legal}"
        return f"after discarding {key}: {answer}"

    @vf.tool
    def note(self, text: str) -> str:
        """Save a private note for yourself. Notes survive into later hands of this
        match — use them for opponent reads, standings plans, or anything worth
        remembering after the board resets."""
        trimmed = text.strip()[:MAX_NOTE_CHARS]
        if not trimmed:
            return "empty note ignored"
        self.state.notes.append(trimmed)
        del self.state.notes[:-MAX_NOTES]
        return f"saved ({len(self.state.notes)} notes)"

    @vf.tool
    def notes(self) -> str:
        """Everything you have saved with note(), oldest first."""
        if not self.state.notes:
            return "no notes saved"
        return "\n".join(f"{i}: {text}" for i, text in enumerate(self.state.notes))


if __name__ == "__main__":
    SeatToolset.run()
