export type CardData = {
  id: number;
  suit: string;
  rank: string;
  value: number;
};

export type SeatConfig = {
  kind: string;
  label: string;
  checkpoint?: string;
  architecture?: string;
  bidding: string;
  biddingThreshold: number;
};

export type AppConfig = {
  agents: string[];
  biddingStyles: string[];
  checkpoints: Record<string, string[]>;
  spriteAvailable: boolean;
};

export type PendingAction = {
  seat: number;
  kind: "bid" | "pickup" | "press" | "declare" | "play";
  mode?: "say" | "hear";
  offer?: number;
  current?: number;
  count?: number;
  options: Array<number | string | boolean>;
};

export type PlayerState = {
  seat: number;
  label: string;
  kind: string;
  hand: CardData[];
  handCount: number;
  points: number;
  soloist: boolean;
  biddingProbability: number | null;
  biddingGame: string | null;
  biddingScores: Record<string, number> | null;
};

export type TrickCard = { seat: number; card: CardData };

export type GameState = {
  id: string;
  revision: number;
  status: string;
  error: string | null;
  paused: boolean;
  speedMs: number;
  reveal: boolean;
  pending: PendingAction | null;
  activeAction: { seat: number; kind: string; current?: number; offer?: number } | null;
  lastAction: { seat: number; kind: string; value: unknown } | null;
  result: { soloistWon: boolean | null; message: string } | null;
  config: Record<string, unknown>;
  phase?: string;
  dealer?: number;
  frontHand?: number;
  nextPlayer?: number | null;
  soloist?: number | null;
  highestBid?: number;
  handGame?: boolean | null;
  game?: string | null;
  gameValue?: number;
  overbid?: boolean;
  pointsSoloist?: number;
  pointsDefenders?: number;
  players?: PlayerState[];
  currentTrick?: TrickCard[];
  history?: Array<{ winner: number; value: number; cards: TrickCard[] }>;
  skat?: CardData[];
};
