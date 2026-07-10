// Mirrors the payloads produced by jongbench/webui.py and jongbench/spectator.py.

export type Tile = string; // "1m".."9m" "1p".. "1s".., honors "E S W N P F C", red "5mr 5pr 5sr", "?" = hidden

export interface Discard {
  tile: Tile;
  tsumogiri: boolean;
  riichi: boolean;
  called: boolean;
}

export interface Meld {
  kind: "chi" | "pon" | "daiminkan" | "ankan" | "kakan";
  tiles: Tile[];
  from_seat: number | null;
}

export interface SeatState {
  seat: number;
  name: string;
  wind: string;
  score: number;
  hand: Tile[];
  melds: Meld[];
  discards: Discard[];
  riichi_declared: boolean;
  riichi_accepted: boolean;
}

export interface Snapshot {
  names: string[];
  scores: number[];
  seats: SeatState[];
  bakaze: string;
  kyoku: number;
  honba: number;
  kyotaku: number;
  oya: number;
  dora_indicators: Tile[];
  tiles_left: number;
  ticker: string[];
  kyoku_index: number;
  done: boolean;
  final_names: string[] | null;
  final_scores: number[] | null;
}

export interface MjaiEvent {
  type: string;
  actor?: number;
  target?: number;
  pai?: Tile;
  consumed?: Tile[];
  tsumogiri?: boolean;
  deltas?: number[];
  ura_markers?: Tile[];
  points?: number;
  fu?: number;
  han?: number;
  yakuman?: number;
  yaku?: [string, number][];
  [key: string]: unknown;
}

export interface Frame {
  seq: number;
  event: MjaiEvent;
  snapshot: Snapshot;
}

export type SessionStatus = "starting" | "running" | "evaluating" | "done" | "error" | "aborted";

export interface SessionState {
  status: SessionStatus;
  error: string | null;
  names: string[];
  human_seat: number | null;
  latest_seq: number;
  final: { names: string[]; scores: number[]; placements: Record<string, number> } | null;
}

export interface SessionListItem {
  run_id: string;
  status: SessionStatus;
  names: string[];
  created: number;
  human_seat: number | null;
  final: { names: string[]; scores: number[]; placements: Record<string, number> } | null;
}

export interface PendingOption {
  choice: number;
  action: string; // "discard" | "riichi"... | "chi" | "pon" | "kan" | "ron" | "tsumo" | "pass"
  label: string;
  tile?: Tile;
}

export interface Pending {
  generation: number;
  seat: number;
  state_text: string;
  menu: string[];
  options?: PendingOption[];
}

export interface DemoBundle {
  game: string;
  seed: [number, number];
  names: string[];
  scores: number[];
  placements: Record<string, number>;
  frames: Frame[];
  review?: Review;
}

// --- review (Mortal evaluation) ---

export interface ReviewCandidate {
  event: MjaiEvent;
  q_value: number;
  prob: number;
}

export interface ReviewEntry {
  kyoku: number;
  honba: number;
  junme: number;
  tiles_left: number;
  last_actor: number;
  tile: Tile;
  actual: MjaiEvent;
  expected: MjaiEvent;
  is_equal: boolean;
  actual_index: number;
  shanten: number;
  at_furiten: boolean;
  details: ReviewCandidate[];
}

export interface PlayerReview {
  name: string;
  review: {
    rating: number;
    total_reviewed: number;
    total_matches: number;
    temperature: number;
    entries: ReviewEntry[];
  };
  aggregates: {
    match_rate: number;
    mean_prob_loss: number;
    by_kind: Record<string, { count: number; matches: number; mean_loss: number }>;
    worst: { kyoku: number; junme: number; loss: number; actual: MjaiEvent; expected: MjaiEvent }[];
  };
}

export interface Review {
  seed: [number, number];
  names: string[];
  scores: number[];
  placements: Record<string, number>;
  players: Record<string, PlayerReview>;
  run_dir?: string;
  report_path?: string;
}
