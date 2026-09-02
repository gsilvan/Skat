import type { CSSProperties } from "react";
import type { CardData } from "./types";

const suitRow: Record<string, number> = { "♣": 0, "♠": 1, "♥": 2, "♦": 3 };
const rankColumn: Record<string, number> = {
  A: 0,
  K: 1,
  Q: 2,
  J: 3,
  "10": 4,
  "9": 5,
  "8": 6,
  "7": 7,
};

export function spritePosition(card: Pick<CardData, "suit" | "rank">) {
  return {
    x: `${(rankColumn[card.rank] / 7) * 100}%`,
    y: `${(suitRow[card.suit] / 3) * 100}%`,
  };
}

type Props = {
  card?: CardData;
  hidden?: boolean;
  playable?: boolean;
  selected?: boolean;
  compact?: boolean;
  onClick?: () => void;
};

export default function Card({
  card,
  hidden = false,
  playable = false,
  selected = false,
  compact = false,
  onClick,
}: Props) {
  if (hidden || !card) {
    return <div className={`card card--back ${compact ? "card--compact" : ""}`} />;
  }
  const position = spritePosition(card);
  const style = {
    "--sprite-x": position.x,
    "--sprite-y": position.y,
  } as CSSProperties;
  const red = card.suit === "♥" || card.suit === "♦";
  return (
    <button
      type="button"
      className={`card card--face ${playable ? "card--playable" : ""} ${selected ? "card--selected" : ""} ${compact ? "card--compact" : ""}`}
      style={style}
      disabled={!onClick}
      onClick={onClick}
      aria-label={`${card.rank}${card.suit}`}
    >
      <span className={`card__fallback ${red ? "card__fallback--red" : ""}`}>
        <strong>{card.rank}</strong>
        <span>{card.suit}</span>
      </span>
    </button>
  );
}
