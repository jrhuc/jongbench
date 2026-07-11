import { useMemo, useState } from "preact/hooks";
import { ProviderIcon, providerOfName } from "../providers";
import { TileView } from "../tiles";
import type { MjaiEvent, PlayerReview, ReviewEntry, Tile } from "../types";
import "./decisionlog.css";

export function formatKyoku(kyoku: number): string {
  const winds = ["東", "南", "西", "北"];
  const wind = winds[Math.floor(kyoku / 4)];
  return wind ? `${wind}${(kyoku % 4) + 1}` : `局${kyoku + 1}`;
}

export function actionKind(event: MjaiEvent): string {
  if (event.type === "dahai") return "discard";
  if (event.type === "reach") return "riichi";
  if (event.type === "hora") return "win";
  if (event.type === "none") return "pass";
  if (event.type === "chi" || event.type === "pon") return event.type;
  if (event.type.includes("kan") || event.type === "kan") return "kan";
  return event.type;
}

export function Action({ event }: { event: MjaiEvent }) {
  const kind = actionKind(event);
  if (kind === "discard") return <span class="action-tiles"><TileView tile={(event.pai ?? "?") as Tile} size="s" /></span>;
  if (kind === "chi" || kind === "pon" || kind === "kan") {
    return <span class="action-call"><span class="action-word">{kind}</span>{(event.consumed ?? []).map((tile, index) => <TileView key={`${tile}-${index}`} tile={tile as Tile} size="s" />)}</span>;
  }
  return <span class="action-word">{kind}</span>;
}

interface DecisionGroup {
  key: string;
  kyoku: number;
  honba: number;
  entries: ReviewEntry[];
}

function probLoss(entry: ReviewEntry): number {
  if (entry.actual_index < 0 || entry.actual_index >= entry.details.length) return 0;
  return Math.max(0, (entry.details[0]?.prob ?? 0) - entry.details[entry.actual_index].prob);
}

function lossTone(loss: number): string {
  if (loss < 0.02) return "muted";
  if (loss < 0.1) return "gold";
  return "crimson";
}

function DecisionEntry({ entry, entryKey, expanded, onToggle }: { entry: ReviewEntry; entryKey: string; expanded: boolean; onToggle(): void }) {
  const loss = probLoss(entry);
  const bestProb = entry.details[0]?.prob ?? 0;

  return (
    <div class={`dlog-entry ${entry.is_equal ? "dlog-entry-match" : ""}`}>
      <button class="dlog-entry-row" type="button" onClick={onToggle} aria-expanded={expanded}>
        <span class="dlog-turn">{entry.junme}巡</span>
        <span class="dlog-tiles-left">{entry.tiles_left} left</span>
        <span class="dlog-played"><Action event={entry.actual} /></span>
        {!entry.is_equal && <span class="dlog-expected">Mortal: <Action event={entry.expected} /></span>}
        {!entry.is_equal && <span class={`dlog-loss dlog-loss-${lossTone(loss)}`}>−{(loss * 100).toFixed(1)}%</span>}
        <span class="dlog-entry-chevron" aria-hidden="true">{expanded ? "⌃" : "⌄"}</span>
      </button>
      {expanded && (
        <div class="dlog-entry-detail">
          <div class="dlog-meta">
            <span class="dlog-context">on <TileView tile={entry.tile} size="s" /></span>
            <span>shanten {entry.shanten}</span>
            {entry.at_furiten && <span class="dlog-furiten">furiten</span>}
          </div>
          <div class="dlog-candidates">
            {entry.details.slice(0, 8).map((candidate, index) => {
              const actual = index === entry.actual_index;
              const best = index === 0;
              const tag = actual && best ? "played · best" : actual ? "played" : best ? "best" : null;
              const width = bestProb > 0 ? Math.min(100, (candidate.prob / bestProb) * 100) : 0;
              return (
                <div class={`dlog-candidate ${actual ? "dlog-candidate-actual" : ""}`} key={`${entryKey}-${index}`}>
                  <span class="dlog-rank">{index + 1}</span>
                  <span class="dlog-candidate-action"><Action event={candidate.event} /></span>
                  <span class="dlog-prob-track"><span class="dlog-prob-fill" style={{ width: `${width}%` }} /></span>
                  <span class="dlog-prob">{(candidate.prob * 100).toFixed(1)}%</span>
                  {tag && <span class="dlog-tag">{tag}</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export function DecisionLog({ player, onBack }: { player: PlayerReview; onBack(): void }) {
  const groups = useMemo<DecisionGroup[]>(() => {
    const grouped = new Map<string, DecisionGroup>();
    for (const entry of player.review.entries) {
      const key = `${entry.kyoku}-${entry.honba}`;
      const group = grouped.get(key);
      if (group) group.entries.push(entry);
      else grouped.set(key, { key, kyoku: entry.kyoku, honba: entry.honba, entries: [entry] });
    }
    return [...grouped.values()];
  }, [player.review.entries]);
  const [openGroups, setOpenGroups] = useState<Set<string>>(() => new Set(groups[0] ? [groups[0].key] : []));
  const [openEntries, setOpenEntries] = useState<Set<string>>(() => new Set());
  const provider = providerOfName(player.name);

  return (
    <section class="dlog">
      <button class="dlog-back" type="button" onClick={onBack}>← Back to summary</button>
      <header class="dlog-header">
        <h2 class="dlog-player">{provider && <ProviderIcon provider={provider} size={20} />}{player.name}</h2>
        <div class="dlog-stats">
          <span><strong>{(player.review.rating * 100).toFixed(1)}</strong> rating</span>
          <span><strong>{(player.aggregates.match_rate * 100).toFixed(1)}%</strong> match rate</span>
          <span><strong>{player.review.total_reviewed}</strong> decisions reviewed</span>
        </div>
      </header>
      <div class="dlog-groups">
        {groups.map((group) => {
          const open = openGroups.has(group.key);
          const mistakes = group.entries.filter((entry) => probLoss(entry) > 0.05).length;
          return (
            <section class="dlog-group" key={group.key}>
              <button class="dlog-group-toggle" type="button" onClick={() => setOpenGroups((current) => {
                const next = new Set(current);
                if (next.has(group.key)) next.delete(group.key); else next.add(group.key);
                return next;
              })} aria-expanded={open}>
                <span>{formatKyoku(group.kyoku)}{group.honba > 0 && ` · ${group.honba} honba`}</span>
                <span class="dlog-group-summary">{mistakes} {mistakes === 1 ? "mistake" : "mistakes"} <span aria-hidden="true">{open ? "⌃" : "⌄"}</span></span>
              </button>
              {open && <div class="dlog-entries">
                {group.entries.map((entry, index) => {
                  const entryKey = `${group.key}-${index}`;
                  return <DecisionEntry key={entryKey} entry={entry} entryKey={entryKey} expanded={openEntries.has(entryKey)} onToggle={() => setOpenEntries((current) => {
                    const next = new Set(current);
                    if (next.has(entryKey)) next.delete(entryKey); else next.add(entryKey);
                    return next;
                  })} />;
                })}
              </div>}
            </section>
          );
        })}
      </div>
    </section>
  );
}
