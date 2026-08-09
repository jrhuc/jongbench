import type { Frame, Pending, Review, SessionState } from "./types";

async function json<T>(promise: Promise<Response>): Promise<T> {
  let response: Response;
  try {
    response = await promise;
  } catch {
    throw new Error("could not reach the jongbench server — is it still running?");
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.error === "string") message = body.error;
    } catch {}
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export function fetchState(): Promise<SessionState> {
  return json<SessionState>(fetch("/api/state", { cache: "no-store" }));
}

export function fetchPending(): Promise<Pending | null> {
  return json<Pending | null>(fetch("/api/pending", { cache: "no-store" }));
}

export function choose(generation: number, choice: number): Promise<{ ok: boolean }> {
  const response = fetch("/api/choose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation, choice }),
  });
  return json<{ ok: boolean }>(response);
}

export function abortGame(): Promise<{ ok: boolean }> {
  return json<{ ok: boolean }>(fetch("/api/abort", { method: "POST" }));
}

export function fetchReview(): Promise<Review> {
  return json<Review>(fetch("/api/review", { cache: "no-store" }));
}

export interface EventStream {
  close(): void;
}

export function streamEvents(
  since: number,
  onFrame: (frame: Frame) => void,
  onStatus: (state: SessionState) => void,
): EventStream {
  const source = new EventSource(`/api/events?since=${since}`);
  source.addEventListener("frame", (event) => {
    onFrame(JSON.parse((event as MessageEvent).data) as Frame);
  });
  source.addEventListener("status", (event) => {
    const state = JSON.parse((event as MessageEvent).data) as SessionState;
    onStatus(state);
    if (state.status === "done" || state.status === "error" || state.status === "aborted") source.close();
  });
  return { close: () => source.close() };
}
