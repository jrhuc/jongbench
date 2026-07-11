import type { Tile } from "./types";

// Sort order matches jongbench/tiles.py.
const SUIT_ORDER: Record<string, number> = { m: 0, p: 1, s: 2 };
const HONOR_ORDER: Record<string, number> = { E: 0, S: 1, W: 2, N: 3, P: 4, F: 5, C: 6 };

function tileSortKey(tile: Tile): number {
  if (tile in HONOR_ORDER) return 300 + HONOR_ORDER[tile] * 10;
  const rank = Number(tile[0]);
  const suit = SUIT_ORDER[tile[1]];
  if (Number.isNaN(rank) || suit === undefined) return 999;
  return suit * 100 + rank * 10 + (tile.endsWith("r") ? 1 : 0);
}

export function sortTiles(tiles: Tile[]): Tile[] {
  return [...tiles].sort((a, b) => tileSortKey(a) - tileSortKey(b));
}

function isRed(tile: Tile): boolean {
  return tile.endsWith("r");
}

interface TileProps {
  tile: Tile;
  size?: "s" | "m" | "l";
  rotated?: boolean;
  dimmed?: boolean;
  onClick?: () => void;
}

export function TileView({ tile, size = "m", rotated, dimmed, onClick }: TileProps) {
  const back = tile === "?";
  const classes = ["tile", `tile-${size}`];
  if (back) classes.push("tile-back");
  if (rotated) classes.push("tile-rot");
  if (dimmed) classes.push("tile-dim");
  if (isRed(tile)) classes.push("tile-aka");
  if (onClick) classes.push("tile-click");
  const contents = (
    <>
      {!back && (
        <svg viewBox="0 0 320 446" aria-label={tile}>
          <use href={`#pai-${tile.toLowerCase()}`} />
        </svg>
      )}
    </>
  );
  if (onClick) {
    return (
      <button
        type="button"
        class={classes.join(" ")}
        onClick={onClick}
        title={tile}
        aria-label={`Discard ${tile}`}
      >
        {contents}
      </button>
    );
  }
  return (
    <span class={classes.join(" ")} title={back ? undefined : tile}>
      {contents}
    </span>
  );
}
