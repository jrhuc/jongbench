import { useEffect, useMemo, useState } from "preact/hooks";
import { formatEvent, MAX_VISIBLE_LOG_ENTRIES } from "../log";
import type { ReplayBundle, SessionState } from "../types";
import { Shell } from "./Shell";
import { Table } from "./Table";
import { Results } from "./Results";
import "./replay.css";

const SPEEDS = [0.5, 1, 2, 4];

export function Replay({ initial }: { initial: ReplayBundle | null }) {
  const [bundle, setBundle] = useState(initial);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showResults, setShowResults] = useState(false);

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

  if (!bundle || !session || frames.length === 0) {
    return (
      <Shell>
        <Picker
          onLoad={(loaded) => {
            setBundle(loaded);
            setIndex(0);
            setPlaying(true);
          }}
        />
      </Shell>
    );
  }

  const frame = frames[Math.min(index, frames.length - 1)];
  return (
    <Shell names={bundle.names}>
      <main class="run-main replay-main">
        <Table
          snapshot={frame.snapshot}
          lastEvent={frame.event}
          session={session}
          pending={null}
          log={log}
          onChoose={() => {}}
          onAbort={() => {}}
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
          <button
            title="Playback speed"
            onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])}
          >
            {speed}×
          </button>
          <button onClick={() => setShowResults(true)}>Results</button>
        </div>
        {showResults && (
          <Results
            session={session}
            review={bundle.review ?? null}
            onClose={() => setShowResults(false)}
          />
        )}
      </main>
    </Shell>
  );
}

function Picker({ onLoad }: { onLoad: (bundle: ReplayBundle) => void }) {
  const [error, setError] = useState<string | null>(null);
  return (
    <div class="notice replay-picker">
      <h2>Load a replay</h2>
      <p>
        Pick a replay bundle — <code>jongbench replay &lt;run_dir&gt; --out replay.json</code> writes one
        from any finished run.
      </p>
      <input
        type="file"
        accept=".json,application/json"
        onChange={(event) => {
          const file = event.currentTarget.files?.[0];
          if (!file) return;
          file
            .text()
            .then((text) => {
              const parsed = JSON.parse(text) as ReplayBundle;
              if (!Array.isArray(parsed.frames) || parsed.frames.length === 0) {
                throw new Error("no frames in this file");
              }
              onLoad(parsed);
            })
            .catch((exc: Error) => setError(`not a replay bundle: ${exc.message}`));
        }}
      />
      {error && <p class="replay-error">{error}</p>}
    </div>
  );
}
