import { useEffect, useMemo, useState } from "preact/hooks";
import * as api from "../api";
import { formatEvent, MAX_VISIBLE_LOG_ENTRIES } from "../log";
import type { DemoBundle, SessionState } from "../types";
import { Dropdown } from "./Dropdown";
import { Table } from "./Table";
import { Results } from "./Results";
import "./replay.css";

const SPEEDS = [0.5, 1, 2, 4];

export function Replay() {
  const [bundle, setBundle] = useState<DemoBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    api.fetchDemo().then(setBundle).catch((exc: Error) => setError(exc.message));
  }, []);

  const frames = bundle?.frames ?? [];
  const atEnd = index >= frames.length - 1;
  const formattedEvents = useMemo(
    () => bundle?.frames.map((item) => formatEvent(item.event, bundle.names)) ?? [],
    [bundle],
  );
  const log = useMemo(
    () => formattedEvents
      .slice(Math.max(0, index + 1 - MAX_VISIBLE_LOG_ENTRIES), index + 1)
      .filter((line): line is string => line !== null),
    [formattedEvents, index],
  );

  useEffect(() => {
    if (!playing || frames.length === 0) return;
    if (atEnd) {
      setPlaying(false);
      setShowResults(true);
      return;
    }
    const timer = setTimeout(() => setIndex((current) => current + 1), 550 / speed);
    return () => clearTimeout(timer);
  }, [playing, index, atEnd, speed, frames.length]);

  const session = useMemo<SessionState | null>(() => {
    if (!bundle) return null;
    return {
      status: "done",
      error: null,
      names: bundle.names,
      human_seat: null,
      latest_seq: frames.length,
      final: { names: bundle.names, scores: bundle.scores, placements: bundle.placements },
    };
  }, [bundle, frames.length]);

  if (error) {
    return (
      <div class="notice notice-error">
        <h2>No replay available</h2>
        <p>{error}</p>
        <button onClick={() => (location.hash = "")}>Back to setup</button>
      </div>
    );
  }
  if (!bundle || !session || frames.length === 0) {
    return (
      <div class="notice">
        <div class="spinner" />
        <p>Loading replay...</p>
      </div>
    );
  }

  const frame = frames[Math.min(index, frames.length - 1)];
  return (
    <main class="run-main replay-main">
      <Table
        snapshot={frame.snapshot}
        lastEvent={frame.event}
        session={session}
        pending={null}
        log={log}
        onChoose={() => {}}
        onAbort={() => (location.hash = "")}
        onResults={() => setShowResults(true)}
        replay
      />
      <div class="replay-bar">
        <button onClick={() => { setPlaying(false); setIndex((i) => Math.max(0, i - 1)); }} title="Step back">⏮</button>
        <button class="primary replay-play" onClick={() => (atEnd ? (setIndex(0), setPlaying(true)) : setPlaying(!playing))}>
          {playing ? "Pause" : atEnd ? "Restart" : "Play"}
        </button>
        <button onClick={() => { setPlaying(false); setIndex((i) => Math.min(frames.length - 1, i + 1)); }} title="Step forward">⏭</button>
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={index}
          onInput={(event) => { setPlaying(false); setIndex(Number(event.currentTarget.value)); }}
        />
        <span class="replay-pos">{index + 1}/{frames.length}</span>
        <Dropdown
          label="Speed"
          up
          value={String(speed)}
          onChange={(value) => setSpeed(Number(value))}
          options={SPEEDS.map((value) => ({ value: String(value), label: `${value}×` }))}
        />
        <button onClick={() => setShowResults(true)}>Results</button>
      </div>
      {showResults && (
        <Results
          session={session}
          review={bundle.review ?? null}
          onClose={() => setShowResults(false)}
          onNewGame={() => (location.hash = "")}
        />
      )}
    </main>
  );
}
