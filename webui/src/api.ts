import type { DemoBundle, Frame, Pending, Review, SessionListItem, SessionState } from "./types";

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

export interface StartRequest {
  models: string[];
  keys: Record<string, string>;
  seed: number | null;
  human_seat: number | null;
  label: string | null;
}

export interface StartResponse {
  run_id: string;
  names: string[];
  human_seat: number | null;
}

export function startGame(request: StartRequest): Promise<StartResponse> {
  const response = fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return json<StartResponse>(response);
}

export function listSessions(): Promise<SessionListItem[]> {
  return json<SessionListItem[]>(fetch("/api/sessions", { cache: "no-store" }));
}

export function fetchState(runId: string): Promise<SessionState> {
  return json<SessionState>(fetch(`/api/state/${encodeURIComponent(runId)}`, { cache: "no-store" }));
}

export function fetchPending(runId: string): Promise<Pending | null> {
  return json<Pending | null>(fetch(`/api/pending/${encodeURIComponent(runId)}`, { cache: "no-store" }));
}

export function choose(runId: string, generation: number, choice: number): Promise<{ ok: boolean }> {
  const response = fetch(`/api/choose/${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation, choice }),
  });
  return json<{ ok: boolean }>(response);
}

export function abortGame(runId: string): Promise<{ ok: boolean }> {
  return json<{ ok: boolean }>(fetch(`/api/abort/${encodeURIComponent(runId)}`, { method: "POST" }));
}

export function fetchReview(runId: string): Promise<Review> {
  return json<Review>(fetch(`/api/review/${encodeURIComponent(runId)}`, { cache: "no-store" }));
}

export function fetchDemo(): Promise<DemoBundle> {
  return json<DemoBundle>(fetch("/api/demo", { cache: "no-store" }));
}

export function demoAvailable(): Promise<boolean> {
  return fetch("/api/demo", { method: "HEAD" }).then((r) => r.ok).catch(() => false);
}

export interface EventStream {
  close(): void;
}

export function streamEvents(
  runId: string,
  since: number,
  onFrame: (frame: Frame) => void,
  onStatus: (state: SessionState) => void,
): EventStream {
  const source = new EventSource(`/api/events/${encodeURIComponent(runId)}?since=${since}`);
  source.addEventListener("frame", (event) => {
    onFrame(JSON.parse((event as MessageEvent).data) as Frame);
  });
  source.addEventListener("status", (event) => {
    onStatus(JSON.parse((event as MessageEvent).data) as SessionState);
    source.close();
  });
  return { close: () => source.close() };
}
