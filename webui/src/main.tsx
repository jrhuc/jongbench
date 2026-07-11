import { render } from "preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import "./theme.css";
import * as api from "./api";
import { formatEvent, MAX_VISIBLE_LOG_ENTRIES } from "./log";
import type { Frame, MjaiEvent, Pending, Review, SessionState, Snapshot } from "./types";
import { Replay } from "./components/Replay";
import { Setup } from "./components/Setup";
import { Table } from "./components/Table";
import { Results } from "./components/Results";

type Route = { view: "setup" } | { view: "run"; runId: string } | { view: "replay" };

function parseRoute(): Route {
  if (location.hash === "#replay") return { view: "replay" };
  const match = /^#run=([A-Za-z0-9]+)$/.exec(location.hash);
  return match ? { view: "run", runId: match[1] } : { view: "setup" };
}

function App() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const onStarted = useCallback((runId: string) => {
    location.hash = `#run=${runId}`;
  }, []);

  useEffect(() => {
    const onHash = () => setRoute(parseRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <>
      <header class="app-header">
        <a class="brand" href="#" onClick={() => (location.hash = "")}>
          <span class="brand-mark">雀</span> jongbench
        </a>
        {route.view === "run" && (
          <span class="run-id" title="run id">
            {route.runId}
          </span>
        )}
      </header>
      {route.view === "setup" ? (
        <Setup onStarted={onStarted} />
      ) : route.view === "replay" ? (
        <Replay />
      ) : (
        <Run runId={route.runId} key={route.runId} />
      )}
    </>
  );
}

function Run({ runId }: { runId: string }) {
  const [session, setSession] = useState<SessionState | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [lastEvent, setLastEvent] = useState<MjaiEvent | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const lastSeq = useRef(0);

  useEffect(() => {
    let stream: api.EventStream | null = null;
    let cancelled = false;

    const onStatus = (state: SessionState) => {
      if (cancelled) return;
      setSession(state);
    };

    api
      .fetchState(runId)
      .then((state) => {
        if (cancelled) return;
        onStatus(state);
        stream = api.streamEvents(
          runId,
          0,
          (frame: Frame) => {
            if (cancelled || frame.seq <= lastSeq.current) return;
            lastSeq.current = frame.seq;
            setSnapshot(frame.snapshot);
            setLastEvent(frame.event);
            const line = formatEvent(frame.event, frame.snapshot.names);
            if (line) setLog((current) => [...current, line].slice(-MAX_VISIBLE_LOG_ENTRIES));
          },
          onStatus,
        );
      })
      .catch((exc: Error) => setError(exc.message));

    return () => {
      cancelled = true;
      stream?.close();
    };
  }, [runId]);

  const humanSeat = session?.human_seat ?? null;
  const status = session?.status ?? "starting";
  useEffect(() => {
    if (humanSeat === null || (status !== "running" && status !== "starting")) {
      setPending(null);
      return;
    }
    let cancelled = false;
    const tick = () =>
      api
        .fetchPending(runId)
        .then((value) => {
          if (!cancelled) setPending(value);
        })
        .catch(() => {});
    tick();
    const timer = setInterval(tick, 700);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId, humanSeat, status]);

  useEffect(() => {
    if (status !== "done") return;
    let cancelled = false;
    api
      .fetchReview(runId)
      .then((value) => {
        if (cancelled) return;
        setReview(value);
        setShowResults(true);
      })
      .catch(() => {
        // Absent on --no-eval runs — still open standings once the game is done.
        if (!cancelled) setShowResults(true);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, status]);

  const onChoose = useCallback(
    (generation: number, choice: number) => {
      setPending(null);
      api.choose(runId, generation, choice).catch((exc: Error) => setError(exc.message));
    },
    [runId],
  );

  const finished = status === "done" || status === "error";
  const onAbort = useCallback(() => {
    if (finished) {
      location.hash = "";
      return;
    }
    api.abortGame(runId).then(() => (location.hash = "")).catch((exc: Error) => setError(exc.message));
  }, [runId, finished]);

  if (error) {
    return (
      <div class="notice notice-error">
        <h2>Something went wrong</h2>
        <p>{error}</p>
        <button onClick={() => (location.hash = "")}>Back to setup</button>
      </div>
    );
  }
  if (session === null || snapshot === null) {
    return (
      <div class="notice">
        <div class="spinner" />
        <p>{session?.status === "error" ? session.error : "Seating players..."}</p>
        {session?.status === "error" && <button onClick={() => (location.hash = "")}>Back to setup</button>}
      </div>
    );
  }

  return (
    <main class="run-main">
      <Table
        snapshot={snapshot}
        lastEvent={lastEvent}
        session={session}
        pending={pending}
        log={log}
        onChoose={onChoose}
        onAbort={onAbort}
        onResults={status === "done" || (status === "error" && session.final !== null) ? () => setShowResults(true) : undefined}
      />
      {showResults && (
        <Results
          session={session}
          review={review}
          onClose={() => setShowResults(false)}
          onNewGame={() => (location.hash = "")}
        />
      )}
    </main>
  );
}

render(<App />, document.getElementById("app")!);
