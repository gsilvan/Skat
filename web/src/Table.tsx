import Card from "./Card";
import type { GameState, PlayerState } from "./types";

type Props = {
  game: GameState;
  pressSelection: number[];
  onCard: (id: number) => void;
  onAction: (value: unknown) => void;
  onControl: (command: string, value?: unknown) => void;
  onExit: () => void;
};

function HandView({
  player,
  game,
  pressSelection,
  onCard,
}: {
  player: PlayerState;
  game: GameState;
  pressSelection: number[];
  onCard: (id: number) => void;
}) {
  const pending = game.pending?.seat === player.seat ? game.pending : null;
  const legal = new Set(
    pending && (pending.kind === "play" || pending.kind === "press")
      ? (pending.options as number[])
      : [],
  );
  return (
    <div className={`player player--${player.seat} ${game.nextPlayer === player.seat ? "player--active" : ""}`}>
      <div className="player__meta">
        <span className="player__seat">Seat {player.seat}</span>
        <strong>{player.label}</strong>
        {player.soloist && <span className="role-pill">Solist</span>}
        <span>{player.points} Augen</span>
      </div>
      <div className="hand">
        {player.hand.length
          ? player.hand.map((card) => (
              <Card
                card={card}
                key={card.id}
                playable={legal.has(card.id)}
                selected={pressSelection.includes(card.id)}
                onClick={legal.has(card.id) ? () => onCard(card.id) : undefined}
              />
            ))
          : Array.from({ length: player.handCount }, (_, index) => (
              <Card hidden compact key={index} />
            ))}
      </div>
    </div>
  );
}

function ActionPanel({ game, onAction, pressSelection }: Pick<Props, "game" | "onAction" | "pressSelection">) {
  const pending = game.pending;
  if (!pending) return null;
  if (pending.kind === "play") {
    return <div className="action-prompt">Wähle eine hervorgehobene Karte.</div>;
  }
  if (pending.kind === "press") {
    return (
      <div className="action-prompt">
        <span>Wähle zwei Karten zum Drücken.</span>
        <button disabled={pressSelection.length !== 2} onClick={() => onAction(pressSelection)}>
          Karten drücken
        </button>
      </div>
    );
  }
  if (pending.kind === "pickup") {
    return (
      <div className="action-prompt">
        <span>Skat aufnehmen?</span>
        <button onClick={() => onAction(true)}>Aufnehmen</button>
        <button className="button-muted" onClick={() => onAction(false)}>Hand spielen</button>
      </div>
    );
  }
  if (pending.kind === "declare") {
    const names: Record<string, string> = {
      diamonds: "Karo",
      hearts: "Herz",
      spades: "Pik",
      clubs: "Kreuz",
      grand: "Grand",
      null: "Null",
    };
    return (
      <div className="action-prompt action-prompt--wrap">
        <span>Spiel ansagen:</span>
        {(pending.options as string[]).map((gameName) => (
          <button key={gameName} onClick={() => onAction(gameName)}>{names[gameName]}</button>
        ))}
      </div>
    );
  }
  const offer = pending.offer ?? (pending.options.find((value) => value !== 0) as number | undefined);
  return (
    <div className="action-prompt">
      <span>{pending.mode === "hear" ? `${offer} halten?` : `Nächstes Gebot: ${offer ?? "–"}`}</span>
      {offer !== undefined && <button onClick={() => onAction(offer)}>{pending.mode === "hear" ? "Ja" : offer}</button>}
      <button className="button-muted" onClick={() => onAction(0)}>Passen</button>
    </div>
  );
}

export default function Table(props: Props) {
  const { game } = props;
  const players = game.players ?? [];
  const actionText = (() => {
    if (game.activeAction) {
      const actor = players[game.activeAction.seat]?.label ?? `Seat ${game.activeAction.seat}`;
      if (game.activeAction.kind === "bid" && game.activeAction.offer) {
        return `${actor} überlegt, ob ${game.activeAction.offer} gehalten wird …`;
      }
      return `${actor} überlegt: ${game.activeAction.kind} …`;
    }
    if (game.lastAction) {
      const actor = players[game.lastAction.seat]?.label ?? `Seat ${game.lastAction.seat}`;
      if (game.lastAction.kind === "bid") {
        return Number(game.lastAction.value) === 0
          ? `${actor} passt.`
          : `${actor}: ${game.lastAction.value}`;
      }
      if (game.lastAction.kind === "pickup") {
        return `${actor} ${game.lastAction.value ? "nimmt den Skat auf" : "spielt Hand"}.`;
      }
      if (game.lastAction.kind === "declare") return `${actor} sagt ${game.lastAction.value} an.`;
    }
    return "SKAT";
  })();
  return (
    <main className="game-shell">
      <header className="game-header">
        <div>
          <p className="eyebrow">SKAT TABLE</p>
          <strong>{game.game ?? game.phase ?? "Runde wird vorbereitet"}</strong>
        </div>
        <div className="score-strip">
          <span>Reizwert <strong>{game.highestBid ?? 0}</strong></span>
          <span>Solist <strong>{game.pointsSoloist ?? 0}</strong></span>
          <span>Gegner <strong>{game.pointsDefenders ?? 0}</strong></span>
          <span>Stich <strong>{(game.history ?? []).length + 1}/10</strong></span>
        </div>
        <button className="icon-button" onClick={props.onExit}>×</button>
      </header>

      <section className="table-layout">
        <div className="felt-grain" />
        {players.map((player) => (
          <HandView
            key={player.seat}
            player={player}
            game={game}
            pressSelection={props.pressSelection}
            onCard={props.onCard}
          />
        ))}

        <div className="trick-area">
          {(game.currentTrick ?? []).map((turn) => (
            <div className={`trick-card trick-card--${turn.seat}`} key={`${turn.seat}-${turn.card.id}`}>
              <Card card={turn.card} />
            </div>
          ))}
          {(game.currentTrick ?? []).length === 0 && (
            <div className="table-mark">{actionText}</div>
          )}
        </div>

        <aside className="game-panel">
          <div className="control-row">
            <button onClick={() => props.onControl(game.paused ? "resume" : "pause")}>
              {game.paused ? "▶ Weiter" : "Ⅱ Pause"}
            </button>
            <button onClick={() => props.onControl("step")}>Ein Zug</button>
          </div>
          <label>
            Tempo
            <input
              type="range"
              min="0"
              max="1500"
              step="50"
              value={game.speedMs}
              onChange={(event) => props.onControl("speed", Number(event.target.value))}
            />
          </label>
          <label className="check-label">
            <input type="checkbox" checked={game.reveal} onChange={(event) => props.onControl("reveal", event.target.checked)} />
            Alle Karten zeigen
          </label>
          <div className="history-list">
            <h3>Stiche</h3>
            {(game.history ?? []).map((trick, index) => (
              <div className="history-row" key={index}>
                <span>{index + 1}</span>
                <div>{trick.cards.map((turn) => <Card card={turn.card} compact key={`${turn.seat}-${turn.card.id}`} />)}</div>
                <strong>S{trick.winner} · {trick.value}</strong>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <ActionPanel game={game} onAction={props.onAction} pressSelection={props.pressSelection} />

      {(game.result || game.error) && (
        <div className="result-card">
          <p className="eyebrow">RUNDE BEENDET</p>
          <h2>{game.error ?? game.result?.message}</h2>
          <p>{game.game} · {game.pointsSoloist ?? 0}:{game.pointsDefenders ?? 0} Augen</p>
          <button className="primary-button" onClick={props.onExit}>Zur Konfiguration</button>
        </div>
      )}
    </main>
  );
}
