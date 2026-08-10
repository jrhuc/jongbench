import { useMemo, useState } from "preact/hooks";
import { TileView } from "../tiles";
import type { ReplayBundle, SessionState } from "../types";
import { Shell } from "./Shell";
import { Table } from "./Table";
import { Replay } from "./Replay";
import "./landing.css";

const REPO_URL = "https://github.com/jrhuc/jongbench";
const KOKUSHI = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"];

export function Landing({ bundle }: { bundle: ReplayBundle }) {
  const [watching, setWatching] = useState(false);

  // The busiest board of the game makes the best backdrop.
  const frame = useMemo(() => {
    let best = bundle.frames[0];
    let bestCount = -1;
    for (const candidate of bundle.frames) {
      const count = candidate.snapshot.seats.reduce((n, seat) => n + seat.discards.length, 0);
      if (count > bestCount) {
        best = candidate;
        bestCount = count;
      }
    }
    return best;
  }, [bundle]);

  const session = useMemo<SessionState>(
    () => ({
      status: "done",
      error: null,
      names: bundle.names,
      human_seat: null,
      latest_seq: bundle.frames.length,
      final: { names: bundle.names, scores: bundle.scores, placements: bundle.placements },
    }),
    [bundle],
  );

  if (watching) return <Replay initial={bundle} />;

  return (
    <Shell names={bundle.names}>
      <main class="run-main landing-main">
        <div class="landing-board" aria-hidden="true">
          <Table
            snapshot={frame.snapshot}
            lastEvent={null}
            session={session}
            pending={null}
            log={[]}
            onChoose={() => {}}
            onAbort={() => {}}
            replay
          />
        </div>
        <div class="landing-card">
          <h1>jongbench</h1>
          <p>
            Riichi mahjong as an LLM eval. Four language models sit a full Tenhou-rules
            hanchan — every discard, call and fold their own read of the board — and each
            decision is graded against Mortal, a superhuman deep-RL player.
          </p>
          <div class="landing-tiles">
            {KOKUSHI.map((tile) => (
              <TileView key={tile} tile={tile} size="s" />
            ))}
          </div>
          <p>
            Behind this card is a real benchmark game. Watch it play out tile by tile,
            with the standings and Mortal's review at the end.
          </p>
          <div class="landing-actions">
            <button class="primary" onClick={() => setWatching(true)}>
              ▶ Watch the example replay
            </button>
            <a class="landing-repo" href={REPO_URL}>
              GitHub ↗
            </a>
          </div>
        </div>
      </main>
    </Shell>
  );
}
