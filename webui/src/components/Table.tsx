import { useState } from "preact/hooks";
import { MAX_VISIBLE_LOG_ENTRIES } from "../log";
import { sortTiles, TileView } from "../tiles";
import type { Discard, Meld, MjaiEvent, Pending, PendingOption, SeatState, SessionState, Snapshot, Tile } from "../types";
import "./table.css";

interface TableProps {
  snapshot: Snapshot;
  lastEvent: MjaiEvent | null;
  session: SessionState;
  pending: Pending | null;
  log: string[];
  onChoose(generation: number, choice: number): void;
  onAbort(): void;
  onResults?: () => void;
}

const WIND_CHARS: Record<string, string> = { E: "東", S: "南", W: "西", N: "北" };
const EMPHASIZED_ACTIONS = new Set(["riichi", "reach", "ron", "tsumo", "hora"]);
const DORA_NEXT: Record<string, string> = { E: "S", S: "W", W: "N", N: "E", P: "F", F: "C", C: "P" };

function optionsFor(pending: Pending): PendingOption[] {
  return pending.options ?? [];
}

/* The tile a dora indicator points at: next rank wrapping 9->1, winds and
   dragons cycling in their own rings. */
function doraOf(indicator: Tile): string {
  const match = /^([1-9])([mps])r?$/.exec(indicator);
  if (match) return `${match[1] === "9" ? 1 : Number(match[1]) + 1}${match[2]}`;
  return DORA_NEXT[indicator] ?? "";
}

function doraSetOf(indicators: Tile[]): Set<string> {
  return new Set(indicators.map(doraOf));
}

function isDora(tile: Tile, dora: Set<string>): boolean {
  return dora.has(tile.replace(/r$/, ""));
}

function MeldView({ meld, size, dora }: { meld: Meld; size: "s" | "m"; dora: Set<string> }) {
  const calledIndex = meld.from_seat === null ? -1 : meld.tiles.length - 1;
  const hidden = meld.kind === "ankan";
  return (
    <span class="seat-meld">
      {meld.tiles.map((tile, index) => {
        const shown = hidden && (index === 0 || index === meld.tiles.length - 1) ? "?" : tile;
        return (
          <TileView
            key={`${tile}-${index}`}
            tile={shown}
            size={size}
            rotated={index === calledIndex}
            dora={shown !== "?" && isDora(tile, dora)}
          />
        );
      })}
    </span>
  );
}

/* Rows of six, each an independent flex line so a rotated riichi tile only
   widens its own row — a grid column would shift the neighbouring rows too. */
function Pond({ discards, size, glowNewest, dora }: { discards: Discard[]; size: "s" | "m"; glowNewest: boolean; dora: Set<string> }) {
  const newest = discards.length - 1;
  const rows: Discard[][] = [];
  for (let start = 0; start < discards.length; start += 6) rows.push(discards.slice(start, start + 6));
  return (
    <div class="pond">
      {rows.map((row, rowIndex) => (
        <div class="pond-row" key={rowIndex}>
          {row.map((discard, columnIndex) => {
            const index = rowIndex * 6 + columnIndex;
            return (
              <span
                class={`pond-tile${glowNewest && index === newest ? " pond-newest" : ""}${discard.tsumogiri ? " pond-tsumogiri" : ""}`}
                key={`${discard.tile}-${index}`}
              >
                <TileView tile={discard.tile} size={size} rotated={discard.riichi} dimmed={discard.called} dora={isDora(discard.tile, dora)} />
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/* One edge of the autotable's electronic center panel: wind, player identity
   and an LED score readout, oriented along that player's side of the square. */
function CenterSeat({ seat, side, dealer, active }: { seat: SeatState; side: number; dealer: boolean; active: boolean }) {
  return (
    <div class={`cseat cseat-${side}${dealer ? " cseat-dealer" : ""}${active ? " cseat-active" : ""}`} title={seat.name}>
      <span class="cseat-wind">{WIND_CHARS[seat.wind] ?? seat.wind}</span>
      <span class="cseat-name">{seat.name}</span>
      {seat.riichi_declared && <span class="cseat-lamp" title="Riichi" />}
      <span class="cseat-score">{seat.score}</span>
    </div>
  );
}

function SeatLayer({
  seat,
  side,
  isPov,
  pending,
  lastEvent,
  dora,
  onChoose,
}: {
  seat: SeatState;
  side: number;
  isPov: boolean;
  pending: Pending | null;
  lastEvent: MjaiEvent | null;
  dora: Set<string>;
  onChoose: (generation: number, choice: number) => void;
}) {
  const discardOptions = pending ? optionsFor(pending).filter((option) => option.action === "discard" && option.tile) : [];
  const canDiscard = isPov && discardOptions.length > 0;
  const justDiscarded = lastEvent?.type === "dahai" && lastEvent.actor === seat.seat;

  return (
    <div class={`seat-layer seat-layer-${side}`}>
      {seat.riichi_declared && <span class="riichi-stick" title="Riichi" />}
      <Pond discards={seat.discards} size={isPov ? "m" : "s"} glowNewest={justDiscarded} dora={dora} />
      <div class="seat-melds">
        {seat.melds.map((meld, index) => (
          <MeldView key={index} meld={meld} size={isPov ? "m" : "s"} dora={dora} />
        ))}
      </div>
      <div class="seat-hand">
        {sortTiles(seat.hand).map((tile, index) => {
          const option = canDiscard ? discardOptions.find((item) => item.tile === tile) : undefined;
          return (
            <TileView
              key={`${tile}-${index}`}
              tile={tile}
              size={isPov ? "l" : "m"}
              dimmed={canDiscard && !option}
              dora={isDora(tile, dora)}
              onClick={option ? () => onChoose(pending!.generation, option.choice) : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}

function Center({ snapshot, sides, activeSeat }: {
  snapshot: Snapshot;
  sides: { side: number; seat: SeatState }[];
  activeSeat: number | null;
}) {
  const dora: Tile[] = [
    ...snapshot.dora_indicators,
    ...Array<Tile>(Math.max(0, 5 - snapshot.dora_indicators.length)).fill("?"),
  ];
  return (
    <section class="center">
      <span class="center-wall center-wall-h center-wall-top" />
      <span class="center-wall center-wall-h center-wall-bottom" />
      <span class="center-wall center-wall-v center-wall-left" />
      <span class="center-wall center-wall-v center-wall-right" />
      <div class="center-square">
        {sides.map(({ side, seat }) => (
          <CenterSeat
            key={seat.seat}
            seat={seat}
            side={side}
            dealer={seat.seat === snapshot.oya}
            active={activeSeat === seat.seat}
          />
        ))}
        <div class="center-info">
          <div class="center-round">
            {WIND_CHARS[snapshot.bakaze] ?? snapshot.bakaze} <b>{snapshot.kyoku}</b>
          </div>
          <div class="center-meta">
            {snapshot.honba > 0 && <span>{snapshot.honba} honba</span>}
            {snapshot.kyotaku > 0 && <span class="center-kyotaku">{snapshot.kyotaku} riichi</span>}
            <span><b>{snapshot.tiles_left}</b> left</span>
          </div>
          <div class="center-dora">
            {dora.map((tile, index) => (
              <TileView key={`${tile}-${index}`} tile={tile} size="s" />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function ActionBar({ pending, onChoose }: { pending: Pending; onChoose: (generation: number, choice: number) => void }) {
  const actions = optionsFor(pending).filter((option) => option.action !== "discard");
  if (actions.length === 0) return null;
  return (
    <div class="abar" role="group" aria-label="Available actions">
      {actions.map((option) => (
        <button
          class={`abar-action${EMPHASIZED_ACTIONS.has(option.action) ? " abar-emphasis" : ""}${option.action === "pass" ? " abar-pass" : ""}`}
          key={option.choice}
          onClick={() => onChoose(pending.generation, option.choice)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Table({ snapshot, lastEvent, session, pending, log, onChoose, onAbort, onResults }: TableProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const pov = session.human_seat ?? 0;
  const seatsByNumber = new Map(snapshot.seats.map((seat) => [seat.seat, seat]));
  const sides = [0, 1, 2, 3].flatMap((side) => {
    const seat = seatsByNumber.get((pov + side) % 4);
    return seat ? [{ side, seat }] : [];
  });
  const actor = typeof lastEvent?.actor === "number" ? lastEvent.actor : null;
  const humanTurn = pending !== null && pending.seat === pov;
  const activeSeat = humanTurn ? pov : session.status === "running" ? actor : null;
  const dora = doraSetOf(snapshot.dora_indicators);

  return (
    <div class="table-root">
      <div class="table-board">
        {sides.map(({ side, seat }) => (
          <SeatLayer
            key={seat.seat}
            seat={seat}
            side={side}
            isPov={side === 0 && session.human_seat !== null}
            pending={humanTurn ? pending : null}
            lastEvent={lastEvent}
            dora={dora}
            onChoose={onChoose}
          />
        ))}
        <Center snapshot={snapshot} sides={sides} activeSeat={activeSeat} />
        {humanTurn && <ActionBar pending={pending} onChoose={onChoose} />}
      </div>
      <div class="table-controls">
        <span class={`table-status table-status-${session.status}`}>
          <i />
          {session.status === "evaluating" ? "reviewing" : session.status}
        </span>
        {onResults && <button class="table-results" onClick={onResults}>Results</button>}
        {session.status !== "done" && session.status !== "error" && session.status !== "aborted" && (
          <button
            class="danger table-leave"
            onClick={() => {
              if (confirm("Abort this game?")) onAbort();
            }}
          >
            Abort
          </button>
        )}
      </div>
      <aside class={`drawer${drawerOpen ? " drawer-open" : ""}`}>
        <button class="drawer-toggle" onClick={() => setDrawerOpen((open) => !open)} aria-expanded={drawerOpen}>
          {drawerOpen ? "Close log" : "Log"}
        </button>
        <div class="drawer-panel">
          {pending?.state_text && <pre class="drawer-state">{pending.state_text}</pre>}
          <div class="drawer-ticker">
            {log.slice(-MAX_VISIBLE_LOG_ENTRIES).reverse().map((entry, index) => (
              <div key={`${log.length - index}-${entry}`}>{entry}</div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
