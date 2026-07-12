import { useEffect, useMemo, useState } from "preact/hooks";
import * as api from "../api";
import type { ModelEntry } from "../api";
import { Dropdown } from "./Dropdown";
import { ModelCombobox } from "./ModelCombobox";
import { PROVIDERS, ProviderIcon, providerOf, providerOfName } from "../providers";
import type { ProviderInfo } from "../providers";
import type { SessionListItem } from "../types";
import "./setup.css";

interface SeatDraft {
  providerId: string;
  model: string;
  reasoning: string;
  models: ModelEntry[] | null;
}

const WINDS = [
  ["東", "East"],
  ["南", "South"],
  ["西", "West"],
  ["北", "North"],
] as const;

const DEFAULT_SEATS: SeatDraft[] = [
  { providerId: "anthropic", model: "claude-sonnet-5", reasoning: "default", models: null },
  { providerId: "openai", model: "gpt-5.2", reasoning: "default", models: null },
  { providerId: "google", model: "gemini-3-pro", reasoning: "default", models: null },
  { providerId: "xai", model: "grok-4.5", reasoning: "default", models: null },
];

const REASONING_LABELS: Record<string, string> = {
  off: "Off",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "X-High",
  max: "Max",
};

function createdTime(created: number): string {
  const milliseconds = created < 10_000_000_000 ? created * 1000 : created;
  return new Date(milliseconds).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function Setup({ onStarted }: { onStarted(runId: string): void }) {
  const [seats, setSeats] = useState<SeatDraft[]>(DEFAULT_SEATS);
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [localBaseUrl, setLocalBaseUrl] = useState("http://127.0.0.1:11434/v1");
  const [allowLocal, setAllowLocal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasDemo, setHasDemo] = useState(false);
  const [stateHints, setStateHints] = useState(true);

  const selectedProviders = useMemo(
    () => Array.from(new Set(seats.map((seat) => seat.providerId))).map((id) => providerOf(id)),
    [seats],
  );
  const missingKeyProviders = selectedProviders.filter(
    (provider) => provider.keyName !== null && !provider.optionalKey && !keys[provider.keyName]?.trim(),
  );
  const missingKeysMessage = missingKeyProviders.length === 0
    ? null
    : `Enter an API key for ${missingKeyProviders.map((provider) => provider.label).join(", ")}.`;
  const humanSeat = seats.findIndex((seat) => seat.providerId === "human");

  useEffect(() => {
    api.demoAvailable().then(setHasDemo);
    api.localEndpointsAvailable().then(setAllowLocal).catch(() => {});
  }, []);

  const updateSeat = (index: number, providerId: string) => {
    const provider = providerOf(providerId);
    setSeats((current) =>
      current.map((seat, seatIndex) =>
        seatIndex === index
          ? { providerId, model: provider.placeholder, reasoning: "default", models: null }
          : seat,
      ),
    );
  };

  const updateModel = (index: number, model: string) => {
    setSeats((current) => current.map((seat, seatIndex) => {
      if (seatIndex !== index) return seat;
      const reasoningOptions = seat.models?.find((entry) => entry.id === model.trim())?.reasoning ?? [];
      return {
        ...seat,
        model,
        reasoning: reasoningOptions.includes(seat.reasoning) ? seat.reasoning : "default",
      };
    }));
  };

  const updateModels = (index: number, models: ModelEntry[] | null) => {
    setSeats((current) => current.map((seat, seatIndex) => {
      if (seatIndex !== index) return seat;
      if (models === null) return seat.models === null ? seat : { ...seat, models };
      const reasoningOptions = models.find((entry) => entry.id === seat.model.trim())?.reasoning ?? [];
      return {
        ...seat,
        models,
        reasoning: reasoningOptions.includes(seat.reasoning) ? seat.reasoning : "default",
      };
    }));
  };

  const updateReasoning = (index: number, reasoning: string) => {
    setSeats((current) => current.map((seat, seatIndex) =>
      seatIndex === index ? { ...seat, reasoning } : seat,
    ));
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
        .map((provider) => [provider.keyName, (keys[provider.keyName] ?? "").trim()]),
    );
    const models = seats.map((seat) => {
      const provider = providerOf(seat.providerId);
      if (provider.id === "local") {
        return `compat:${localBaseUrl.trim()}:${seat.model.trim() || provider.placeholder}`;
      }
      return provider.keyName === null ? provider.id : `${provider.id}:${seat.model.trim() || provider.placeholder}`;
    });
    api.startGame({
      models,
      keys: requestKeys,
      seed: null,
      human_seat: humanSeat === -1 ? null : humanSeat,
      label: null,
      state_hints: stateHints,
      reasoning: seats.map((seat) => {
        const levels = seat.models?.find((entry) => entry.id === seat.model.trim())?.reasoning ?? [];
        return levels.includes(seat.reasoning) ? seat.reasoning : null;
      }),
    }).then(({ run_id }) => onStarted(run_id)).catch((exc: Error) => {
      setError(exc.message);
      setSubmitting(false);
    });
  };

  return (
    <main class="setup-main">
      <section class="setup-hero">
        <h1 class="setup-kicker">Riichi mahjong × LLM benchmark</h1>
        {hasDemo && <a class="setup-demo" href="#replay">watch a recorded game →</a>}
      </section>

      <form class="setup-form" onSubmit={start}>
        <section class="setup-section seats-section" aria-label="Seats">
          <div class="seat-grid">
            {seats.map((seat, index) => {
              const provider = providerOf(seat.providerId);
              const reasoningOptions = seat.models?.find((entry) => entry.id === seat.model.trim())?.reasoning ?? [];
              const [wind, windName] = WINDS[index];
              return (
                <article class={`seat-tile${index === 0 ? " seat-tile-east" : ""}`} key={wind}>
                  <div class="seat-title">
                    <span class="seat-wind">{wind}</span>
                    <span class="seat-label">{windName} seat</span>
                  </div>
                  <div class="provider-field">
                    <span>Player</span>
                    <Dropdown
                      label={`${windName} seat player`}
                      value={seat.providerId}
                      onChange={(providerId) => updateSeat(index, providerId)}
                      options={PROVIDERS.filter((option) => option.id !== "local" || allowLocal).map((option) => ({
                        value: option.id,
                        label: option.label,
                        icon: <ProviderIcon provider={option} />,
                        disabled: option.id === "human" && humanSeat !== -1 && humanSeat !== index,
                      }))}
                    />
                  </div>
                  <div class={`model-field${provider.keyName === null ? " model-field-hidden" : ""}`}>
                    <span>Model id</span>
                    <div class="model-control-row">
                      <ModelCombobox
                        key={seat.providerId}
                        provider={provider}
                        apiKey={provider.keyName === null ? "" : keys[provider.keyName] ?? ""}
                        baseUrl={provider.id === "local" ? localBaseUrl : undefined}
                        value={seat.model}
                        label={`${windName} seat model`}
                        onChange={(model) => updateModel(index, model)}
                        onModelsChange={(models) => updateModels(index, models)}
                      />
                      {reasoningOptions.length > 0 && (
                        <div class="reasoning-control">
                          <Dropdown
                            label={`${windName} seat reasoning`}
                            value={seat.reasoning}
                            onChange={(reasoning) => updateReasoning(index, reasoning)}
                            options={[
                              { value: "default", label: "Default" },
                              ...reasoningOptions.map((level) => ({
                                value: level,
                                label: REASONING_LABELS[level] ?? level,
                              })),
                            ]}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {selectedProviders.some((provider) => provider.keyName !== null) && (
          <section class="setup-card keys-panel">
            <div class="section-heading">
              <span class="tile-chip chip-haku" aria-hidden="true"></span>
              <div>
                <h2>Keys</h2>
                <p>Configure access for each selected provider. Keys are held in memory for this game only and never logged.</p>
              </div>
            </div>
            <div class="keys-grid">
              {selectedProviders.some((provider) => provider.id === "local") && (
                <label class="key-field">
                  <span>Local OpenAI-compatible URL</span>
                  <input
                    type="url"
                    value={localBaseUrl}
                    required
                    onInput={(event) => setLocalBaseUrl(event.currentTarget.value)}
                  />
                </label>
              )}
              {selectedProviders.filter((provider): provider is ProviderInfo & { keyName: string } => provider.keyName !== null).map((provider) => (
                <label class="key-field" key={provider.id}>
                  <span><ProviderIcon provider={provider} /> {provider.label} API key</span>
                  <input
                    type="password"
                    value={keys[provider.keyName] ?? ""}
                    placeholder={provider.optionalKey ? "Optional" : "Required"}
                    required={!provider.optionalKey}
                    aria-invalid={!provider.optionalKey && !keys[provider.keyName]?.trim()}
                    onInput={(event) => setKeys((current) => ({ ...current, [provider.keyName]: event.currentTarget.value }))}
                  />
                </label>
              ))}
            </div>
          </section>
        )}

        <section class="setup-card prompt-panel">
          <div class="section-heading">
            <span class="tile-chip chip-chun" aria-hidden="true">中</span>
            <div>
              <h2>Prompt assistance</h2>
              <p>Choose whether models receive rule-derived structure alongside the visible table.</p>
            </div>
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
          <button class="deal-button" type="submit" disabled={submitting || missingKeysMessage !== null}>
            <span class="deal-dot" aria-hidden="true" />
            {submitting ? "Checking keys…" : "Deal"}
          </button>
          {missingKeysMessage && <p class="deal-error" role="status">{missingKeysMessage}</p>}
        </div>
      </form>

      <SessionTables />

      {error && (
        <div class="toast toast-error" role="alert">
          <span class="toast-text">{error}</span>
          <button type="button" class="toast-dismiss" aria-label="Dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}
    </main>
  );
}

/** Owns its own fetch so form keystrokes never re-render the tables list. */
function SessionTables() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);

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

  const active = sessions.filter((item) => item.status === "starting" || item.status === "running" || item.status === "evaluating");
  const finished = sessions.filter((item) => item.status === "done" && item.final !== null);

  if (active.length === 0 && finished.length === 0) return null;

  return (
    <>
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
    </>
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
