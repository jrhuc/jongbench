import { useEffect, useMemo, useState } from "preact/hooks";
import * as api from "../api";
import { PROVIDERS, ProviderIcon, providerOf, providerOfName } from "../providers";
import type { ProviderInfo } from "../providers";
import type { SessionListItem } from "../types";
import "./setup.css";

interface SeatDraft {
  providerId: string;
  model: string;
}

const WINDS = [
  ["東", "East"],
  ["南", "South"],
  ["西", "West"],
  ["北", "North"],
] as const;

const DEFAULT_SEATS: SeatDraft[] = [
  { providerId: "anthropic", model: "claude-sonnet-5" },
  { providerId: "openai", model: "gpt-5.2" },
  { providerId: "google", model: "gemini-3-pro" },
  { providerId: "xai", model: "grok-4" },
];

function createdTime(created: number): string {
  const milliseconds = created < 10_000_000_000 ? created * 1000 : created;
  return new Date(milliseconds).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function Setup({ onStarted }: { onStarted(runId: string): void }) {
  const [seats, setSeats] = useState<SeatDraft[]>(DEFAULT_SEATS);
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasDemo, setHasDemo] = useState(false);
  const [stateHints, setStateHints] = useState(true);

  const active = sessions.filter((item) => item.status === "starting" || item.status === "running" || item.status === "evaluating");
  const finished = sessions.filter((item) => item.status === "done" && item.final !== null);

  const selectedProviders = useMemo(
    () => Array.from(new Set(seats.map((seat) => seat.providerId))).map((id) => providerOf(id)),
    [seats],
  );
  const missingKeyProviders = selectedProviders.filter(
    (provider) => provider.keyName !== null && !keys[provider.keyName]?.trim(),
  );
  const missingKeysMessage = missingKeyProviders.length === 0
    ? null
    : `Enter an API key for ${missingKeyProviders.map((provider) => provider.label).join(", ")}.`;
  const humanSeat = seats.findIndex((seat) => seat.providerId === "human");

  useEffect(() => {
    api.demoAvailable().then(setHasDemo);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      api.listSessions().then((items) => {
        if (!cancelled) setSessions(items);
      }).catch(() => {});
    };
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const updateSeat = (index: number, providerId: string) => {
    const provider = providerOf(providerId);
    setSeats((current) =>
      current.map((seat, seatIndex) =>
        seatIndex === index ? { providerId, model: provider.placeholder } : seat,
      ),
    );
  };

  const updateModel = (index: number, model: string) => {
    setSeats((current) => current.map((seat, seatIndex) => (seatIndex === index ? { ...seat, model } : seat)));
  };

  const start = (event: Event) => {
    event.preventDefault();
    setError(null);
    if (missingKeysMessage) {
      setError(missingKeysMessage);
      return;
    }
    setSubmitting(true);
    const requestKeys = Object.fromEntries(
      selectedProviders
        .filter((provider): provider is ProviderInfo & { keyName: string } => provider.keyName !== null)
        .map((provider) => [provider.keyName, keys[provider.keyName]!.trim()]),
    );
    const models = seats.map((seat) => {
      const provider = providerOf(seat.providerId);
      return provider.keyName === null ? provider.id : `${provider.id}:${seat.model.trim() || provider.placeholder}`;
    });
    api.startGame({
      models,
      keys: requestKeys,
      seed: null,
      human_seat: humanSeat === -1 ? null : humanSeat,
      label: null,
      state_hints: stateHints,
    }).then(({ run_id }) => onStarted(run_id)).catch((exc: Error) => {
      setError(exc.message);
      setSubmitting(false);
    });
  };

  return (
    <main class="setup-main">
      <section class="setup-hero">
        <h1 class="setup-kicker">Riichi mahjong model benchmark</h1>
        {hasDemo && <a class="setup-demo" href="#replay">watch a recorded game →</a>}
      </section>

      <form class="setup-form" onSubmit={start}>
        <section class="setup-section dragon-hatsu">
          <div class="section-heading">
            <h2><span class="dragon-glyph">發</span>Choose the table</h2>
            <p>Seat four players and let the tiles decide.</p>
          </div>
          <div class="seat-grid">
            {seats.map((seat, index) => {
              const provider = providerOf(seat.providerId);
              const [wind, windName] = WINDS[index];
              return (
                <article class="seat-card" key={wind}>
                  <div class="seat-title">
                    <span class="seat-wind">{wind}</span>
                    <span>{windName} seat</span>
                  </div>
                  <label class="provider-field">
                    <span>Player</span>
                    <span class="provider-select-wrap">
                      <ProviderIcon provider={provider} />
                      <select value={seat.providerId} onInput={(event) => updateSeat(index, event.currentTarget.value)}>
                        {PROVIDERS.map((option) => (
                          <option
                            value={option.id}
                            disabled={option.id === "human" && humanSeat !== -1 && humanSeat !== index}
                          >
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </span>
                  </label>
                  <label class={`model-field${provider.keyName === null ? " model-field-hidden" : ""}`}>
                    <span>Model id</span>
                    <input
                      value={seat.model}
                      placeholder={provider.placeholder}
                      disabled={provider.keyName === null}
                      onInput={(event) => updateModel(index, event.currentTarget.value)}
                    />
                  </label>
                </article>
              );
            })}
          </div>
        </section>

        {selectedProviders.some((provider) => provider.keyName !== null) && (
          <section class="setup-section keys-panel dragon-haku">
            <div class="section-heading">
              <h2><span class="dragon-glyph">白</span>Keys</h2>
              <p>Enter a key for every selected model provider. Keys are held in memory for this game only and never logged.</p>
            </div>
            <div class="keys-grid">
              {selectedProviders.filter((provider): provider is ProviderInfo & { keyName: string } => provider.keyName !== null).map((provider) => (
                <label class="key-field" key={provider.id}>
                  <span><ProviderIcon provider={provider} /> {provider.label} API key</span>
                  <input
                    type="password"
                    value={keys[provider.keyName] ?? ""}
                    placeholder="Required"
                    required
                    aria-invalid={!keys[provider.keyName]?.trim()}
                    onInput={(event) => setKeys((current) => ({ ...current, [provider.keyName]: event.currentTarget.value }))}
                  />
                </label>
              ))}
            </div>
          </section>
        )}

        <section class="setup-section prompt-panel dragon-chun">
          <div class="section-heading">
            <h2><span class="dragon-glyph">中</span>Prompt assistance</h2>
            <p>Choose whether models receive rule-derived structure alongside the visible table.</p>
          </div>
          <label class="prompt-toggle">
            <input
              type="checkbox"
              checked={stateHints}
              onInput={(event) => setStateHints(event.currentTarget.checked)}
            />
            <span>
              <b>Show state hints</b>
              <small>Includes shanten, tenpai waits, furiten, and each discard's resulting structure. No EV, safety, or recommended move is supplied.</small>
            </span>
          </label>
        </section>

        <div class="deal-row">
          <button class="primary deal-button" type="submit" disabled={submitting || missingKeysMessage !== null}>
            {submitting ? "Checking keys…" : "Deal"}
          </button>
          {missingKeysMessage && <p class="deal-error" role="status">{missingKeysMessage}</p>}
        </div>
      </form>

      {active.length > 0 && (
        <section class="tables-section">
          <div class="section-heading">
            <h2>Tables in play</h2>
          </div>
          <div class="session-list">
            {active.map((session) => (
              <div class="session-row" key={session.run_id}>
                <span class="status-dot session-active" title={session.status} />
                <span class="session-names">
                  {session.names.map((name, index) => (
                    <SessionPlayer key={`${name}-${index}`} name={name} />
                  ))}
                </span>
                <time>{createdTime(session.created)}</time>
                <a href={`#run=${session.run_id}`}>Watch</a>
              </div>
            ))}
          </div>
        </section>
      )}

      {finished.length > 0 && (
        <section class="tables-section">
          <div class="section-heading">
            <h2>Completed games</h2>
          </div>
          <div class="session-list">
            {finished.map((session) => (
              <div class="session-row" key={session.run_id}>
                <span class="status-dot session-finished" title="done" />
                <span class="session-names">
                  {standingsOf(session).map(({ name, score, placement }) => (
                    <span class="session-standing" key={`${name}-${placement}`}>
                      <b>{placement}.</b> <SessionPlayer name={name} /> <span class="session-score">{score.toLocaleString()}</span>
                    </span>
                  ))}
                </span>
                <time>{createdTime(session.created)}</time>
                <a href={`#run=${session.run_id}`}>View</a>
              </div>
            ))}
          </div>
        </section>
      )}
      {error && (
        <div class="toast toast-error" role="alert">
          <span class="toast-text">{error}</span>
          <button type="button" class="toast-dismiss" aria-label="Dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}
    </main>
  );
}

function SessionPlayer({ name }: { name: string }) {
  const provider = providerOfName(name);
  return <span class="session-player">{provider && <ProviderIcon provider={provider} size={14} />}{name}</span>;
}

function standingsOf(session: SessionListItem) {
  const final = session.final!;
  return final.names
    .map((name, index) => ({ name, score: final.scores[index], placement: final.placements[name] ?? 9 }))
    .sort((a, b) => a.placement - b.placement);
}
