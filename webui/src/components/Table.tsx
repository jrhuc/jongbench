import { useState } from "preact/hooks";
import { MAX_VISIBLE_LOG_ENTRIES } from "../log";
import { ProviderIcon, providerOfName } from "../providers";
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
  replay?: boolean;
}

const WIND_CHARS: Record<string, string> = { E: "東", S: "南", W: "西", N: "北" };
const EMPHASIZED_ACTIONS = new Set(["riichi", "reach", "ron", "tsumo", "hora"]);

function optionsFor(pending: Pending): PendingOption[] {
  return pending.options ?? [];
}

function MeldView({ meld, size }: { meld: Meld; size: "s" | "m" }) {
  const calledIndex = meld.from_seat === null ? -1 : meld.tiles.length - 1;
  const hidden = meld.kind === "ankan";
  return (
    <span class="seat-meld">
      {meld.tiles.map((tile, index) => (
        <TileView
          key={`${tile}-${index}`}
          tile={hidden && (index === 0 || index === meld.tiles.length - 1) ? "?" : tile}
          size={size}
          rotated={index === calledIndex}
        />
      ))}
    </span>
  );
}

function Pond({ discards, size, glowNewest }: { discards: Discard[]; size: "s" | "m"; glowNewest: boolean }) {
  const newest = discards.length - 1;
  return (
    <div class="pond">
      {discards.map((discard, index) => (
        <span
          class={`pond-tile${glowNewest && index === newest ? " pond-newest" : ""}${discard.tsumogiri ? " pond-tsumogiri" : ""}`}
          key={`${discard.tile}-${index}`}
        >
          <TileView tile={discard.tile} size={size} rotated={discard.riichi} dimmed={discard.called} />
        </span>
      ))}
    </div>
  );
}

function SeatPlate({ seat, corner, dealer, active }: { seat: SeatState; corner: number; dealer: boolean; active: boolean }) {
  const provider = providerOfName(seat.name);
  return (
    <div class={`seat-plate seat-plate-${corner}${dealer ? " seat-dealer" : ""}${active ? " seat-active" : ""}`}>
      <span class="seat-wind">{WIND_CHARS[seat.wind] ?? seat.wind}</span>
      <span class="seat-provider">{provider && <ProviderIcon provider={provider} size={14} />}</span>
      <span class="seat-name" title={seat.name}>{seat.name}</span>
      <span class="seat-score">{seat.score.toLocaleString()}</span>
      {seat.riichi_declared && <span class="seat-riichi" title="Riichi" />}
      {active && <span class="seat-thinking">…</span>}
    </div>
  );
}

function SeatLayer({
  seat,
  side,
  isPov,
  pending,
  lastEvent,
  onChoose,
}: {
  seat: SeatState;
  side: number;
  isPov: boolean;
  pending: Pending | null;
  lastEvent: MjaiEvent | null;
  onChoose: (generation: number, choice: number) => void;
}) {
  const discardOptions = pending ? optionsFor(pending).filter((option) => option.action === "discard" && option.tile) : [];
  const canDiscard = isPov && discardOptions.length > 0;
  const justDiscarded = lastEvent?.type === "dahai" && lastEvent.actor === seat.seat;

  return (
    <div class={`seat-layer seat-layer-${side}`}>
      <Pond discards={seat.discards} size={isPov ? "m" : "s"} glowNewest={justDiscarded} />
      <div class="seat-melds">
        {seat.melds.map((meld, index) => (
          <MeldView key={index} meld={meld} size={isPov ? "m" : "s"} />
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
              onClick={option ? () => onChoose(pending!.generation, option.choice) : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}

function Center({ snapshot, turnSide }: { snapshot: Snapshot; turnSide: number | null }) {
  const dora: Tile[] = [
    ...snapshot.dora_indicators,
    ...Array<Tile>(Math.max(0, 5 - snapshot.dora_indicators.length)).fill("?"),
  ];
  return (
    <section class={`center${turnSide === null ? "" : ` center-turn-${turnSide}`}`}>
      <span class="center-wall center-wall-h center-wall-top" />
      <span class="center-wall center-wall-h center-wall-bottom" />
      <span class="center-wall center-wall-v center-wall-left" />
      <span class="center-wall center-wall-v center-wall-right" />
      <div class="center-square">
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

export function Table({ snapshot, lastEvent, session, pending, log, onChoose, onAbort, onResults, replay }: TableProps) {
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
  const turnSide = activeSeat === null ? null : (activeSeat - pov + 4) % 4;

  return (
    <div class="table-root">
      <div class="table-board">
        {sides.map(({ side, seat }) => (
          <SeatLayer
            key={seat.seat}
            seat={seat}
            side={side}
            isPov={side === 0}
            pending={humanTurn ? pending : null}
            lastEvent={lastEvent}
            onChoose={onChoose}
          />
        ))}
        {sides.map(({ side, seat }) => (
          <SeatPlate
            key={seat.seat}
            seat={seat}
            corner={side}
            dealer={seat.seat === snapshot.oya}
            active={activeSeat === seat.seat}
          />
        ))}
        <Center snapshot={snapshot} turnSide={turnSide} />
        {humanTurn && <ActionBar pending={pending} onChoose={onChoose} />}
      </div>
      <div class="table-controls">
        <span class={`table-status table-status-${replay ? "replay" : session.status}`}>
          <i />
          {replay ? "replay" : session.status}
        </span>
        {onResults && <button class="table-results" onClick={onResults}>Results</button>}
        <button
          class="danger table-leave"
          onClick={() => {
            const active = !replay && session.status !== "done" && session.status !== "error";
            if (!active || confirm("Leave this game?")) onAbort();
          }}
        >
          {replay ? "Exit replay" : session.status === "done" || session.status === "error" ? "Exit" : "Leave game"}
        </button>
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
