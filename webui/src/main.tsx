import { render } from "preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import "./theme.css";
import * as api from "./api";
import { formatEvent, MAX_VISIBLE_LOG_ENTRIES } from "./log";
import type { Frame, MjaiEvent, Pending, Review, SessionState, Snapshot } from "./types";
import { Replay } from "./components/Replay";
import { Shell } from "./components/Shell";
import { Table } from "./components/Table";
import { Results } from "./components/Results";

function App() {
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
      .fetchState()
      .then((state) => {
        if (cancelled) return;
        onStatus(state);
        stream = api.streamEvents(
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
  }, []);

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
        .fetchPending()
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
  }, [humanSeat, status]);

  useEffect(() => {
    if (status !== "done") return;
    let cancelled = false;
    api
      .fetchReview()
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
  }, [status]);

  const onChoose = useCallback((generation: number, choice: number) => {
    setPending(null);
    api.choose(generation, choice).catch((exc: Error) => setError(exc.message));
  }, []);

  const onAbort = useCallback(() => {
    api.abortGame().catch((exc: Error) => setError(exc.message));
  }, []);

  if (error) {
    return (
      <Shell>
        <div class="notice notice-error">
          <h2>Something went wrong</h2>
          <p>{error}</p>
        </div>
      </Shell>
    );
  }
  if (session === null || snapshot === null) {
    return (
      <Shell>
        <div class="notice">
          <div class="spinner" />
          <p>{session?.status === "error" ? session.error : "Waiting for the arena..."}</p>
        </div>
      </Shell>
    );
  }

  return (
    <Shell names={session.names}>
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
          />
        )}
      </main>
    </Shell>
  );
}

// A served or co-hosted replay bundle takes over the page (that server has no live
// session to watch); #replay opens the viewer's file picker even without one, which
// is what a static deployment of this page links to.
const root = document.getElementById("app")!;
api.findReplay().then((bundle) => {
  if (bundle || location.hash === "#replay") render(<Replay initial={bundle} />, root);
  else render(<App />, root);
});
