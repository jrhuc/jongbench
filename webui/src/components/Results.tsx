import { useMemo, useState } from "preact/hooks";
import { ProviderIcon, providerOfName } from "../providers";
import type { MjaiEvent, PlayerReview, Review, SessionState } from "../types";
import { Action, DecisionLog, formatKyoku } from "./DecisionLog";
import "./results.css";

interface ReviewPlayer {
  seat: string;
  player: PlayerReview;
  placement: number;
}

interface Disagreement {
  name: string;
  placement: number;
  kyoku: number;
  junme: number;
  loss: number;
  actual: MjaiEvent;
  expected: MjaiEvent;
}

function placementFor(name: string, session: SessionState, review: Review): number {
  return session.final?.placements[name] ?? review.placements[name] ?? 99;
}

function formatScoreDelta(score: number): string {
  const delta = score - 25000;
  return `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toLocaleString()}`;
}

function ordinal(placement: number): string {
  if (placement === 1) return "1st";
  if (placement === 2) return "2nd";
  if (placement === 3) return "3rd";
  return "4th";
}

function PlayerName({ name }: { name: string }) {
  const provider = providerOfName(name);
  return <span class="result-player-name">{provider && <ProviderIcon provider={provider} />}{name}</span>;
}

export function Results({ session, review, onClose, onNewGame }: { session: SessionState; review: Review | null; onClose(): void; onNewGame(): void }) {
  const [showDisagreements, setShowDisagreements] = useState(false);
  const [selectedSeat, setSelectedSeat] = useState<string | null>(null);
  const standings = useMemo(() => {
    if (session.final === null) return [];
    return session.final.names.map((name, index) => ({
      name,
      score: session.final!.scores[index],
      placement: session.final!.placements[name] ?? 99,
    })).sort((a, b) => a.placement - b.placement);
  }, [session.final]);
  const players = useMemo<ReviewPlayer[]>(() => {
    if (review === null) return [];
    return Object.entries(review.players).map(([seat, player]) => ({
      seat,
      player,
      placement: placementFor(player.name, session, review),
    })).sort((a, b) => a.placement - b.placement);
  }, [review, session]);
  const disagreements = useMemo<Disagreement[]>(() => players.flatMap(({ player, placement }) =>
    player.aggregates.worst.map((entry) => ({ name: player.name, placement, ...entry })),
  ).sort((a, b) => b.loss - a.loss).slice(0, 8), [players]);
  const maxRating = Math.max(1, ...players.map(({ player }) => player.review.rating * 100));
  const selectedPlayer = players.find(({ seat }) => seat === selectedSeat)?.player ?? null;

  return (
    <div
      class="results-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="results-title"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section class="results-panel">
        <button class="results-close" type="button" aria-label="Close results" onClick={onClose}>×</button>
        <header class="results-header">
          <p class="results-kicker">Hanchan complete</p>
          <h1 id="results-title">Game over</h1>
          {session.status === "evaluating" && <p class="mortal-reviewing"><span />Mortal is reviewing the game…</p>}
        </header>

        {selectedPlayer ? <DecisionLog key={selectedSeat} player={selectedPlayer} onBack={() => setSelectedSeat(null)} /> : <>
        {standings.length > 0 && (
          <section class="results-section">
            <h2>Standings</h2>
            <div class="standings">
              {standings.map(({ name, score, placement }) => (
                <div class="standing-row" key={name}>
                  <span class={`placement placement-${placement}`}>{ordinal(placement)}</span>
                  <PlayerName name={name} />
                  <span class="final-score">{score.toLocaleString()}</span>
                  <span class={`score-delta ${score >= 25000 ? "positive" : "negative"}`}>{formatScoreDelta(score)}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {review !== null && (
          <section class="results-section review-section">
            <h2>Mortal review</h2>
            <div class="review-stat-grid">
              {players.map(({ seat, player }) => (
                <button class="review-stat" type="button" key={seat} onClick={() => setSelectedSeat(seat)}>
                  <PlayerName name={player.name} />
                  <strong>{(player.review.rating * 100).toFixed(1)}</strong>
                  <span class="review-stat-label">Mortal rating</span>
                  <dl>
                    <div><dt>Match rate</dt><dd>{(player.aggregates.match_rate * 100).toFixed(1)}%</dd></div>
                    <div><dt>Mean prob-loss</dt><dd>{player.aggregates.mean_prob_loss.toFixed(3)}</dd></div>
                  </dl>
                  <span class="review-stat-affordance">View decisions <span aria-hidden="true">›</span></span>
                </button>
              ))}
            </div>
            <div class="rating-bars" aria-label="Mortal ratings">
              {players.map(({ seat, player }) => {
                const rating = player.review.rating * 100;
                return (
                  <div class="rating-row" key={seat}>
                    <PlayerName name={player.name} />
                    <div class="rating-track"><div class="rating-fill" style={{ width: `${(rating / maxRating) * 100}%` }} /></div>
                    <span class="rating-value">{rating.toFixed(1)}</span>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {review !== null && disagreements.length > 0 && (
          <section class="results-section disagreements">
            <button class="disagreements-toggle" type="button" onClick={() => setShowDisagreements((value) => !value)} aria-expanded={showDisagreements}>
              <span>Sharpest disagreements</span><span>{showDisagreements ? "−" : "+"}</span>
            </button>
            {showDisagreements && (
              <div class="disagreement-list">
                {disagreements.map((item, index) => (
                  <div class="disagreement-row" key={`${item.name}-${item.kyoku}-${item.junme}-${index}`}>
                    <PlayerName name={item.name} />
                    <span class="turn-label">{formatKyoku(item.kyoku)} · {item.junme}巡</span>
                    <span class="decision"><span>played <Action event={item.actual} /></span><span class="preferred">Mortal preferred <Action event={item.expected} /></span></span>
                    <span class="loss-value">{(item.loss * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        <footer class="results-footer">
          <button class="primary" type="button" onClick={onNewGame}>New game</button>
          <button type="button" onClick={onClose}>Close</button>
        </footer>
        </>}
      </section>
    </div>
  );
}
