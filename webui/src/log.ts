import type { MjaiEvent } from "./types";

const KYOKU_WINDS: Record<string, string> = { E: "East", S: "South", W: "West", N: "North" };
const CALL_NAMES: Record<string, string> = { chi: "chi", pon: "pon", daiminkan: "kan", ankan: "ankan", kakan: "kakan" };

export function formatEvent(event: MjaiEvent, names: string[]): string | null {
  const name = (seat: number | undefined) =>
    seat === undefined ? "?" : names[seat] ?? `P${seat}`;
  switch (event.type) {
    case "start_kyoku": {
      const wind = KYOKU_WINDS[String(event.bakaze)] ?? event.bakaze;
      const honba = Number(event.honba ?? 0);
      return `— ${wind} ${event.kyoku}${honba > 0 ? ` · ${honba} honba` : ""} —`;
    }
    case "tsumo":
      return `${name(event.actor)} drew ${event.pai ?? "a tile"}`;
    case "dahai":
      return `${name(event.actor)} discarded ${event.pai}${event.tsumogiri ? " (tsumogiri)" : ""}`;
    case "chi":
    case "pon":
    case "daiminkan":
      return `${name(event.actor)} called ${CALL_NAMES[event.type]} on ${name(event.target)}'s ${event.pai}`;
    case "ankan":
      return `${name(event.actor)} declared ankan ${(event.consumed ?? [])[0] ?? ""}`;
    case "kakan":
      return `${name(event.actor)} declared kakan ${event.pai}`;
    case "reach":
      return `${name(event.actor)} declared riichi`;
    case "reach_accepted":
      return `${name(event.actor)}'s riichi accepted`;
    case "dora":
      return `New dora indicator ${event.dora_marker ?? event.pai ?? ""}`;
    case "hora":
      return event.actor === event.target
        ? `${name(event.actor)} won by tsumo`
        : `${name(event.actor)} won from ${name(event.target)}`;
    case "ryukyoku":
      return "Hand ended in a draw";
    case "end_game":
      return "Game finished";
    default:
      return null;
  }
}
