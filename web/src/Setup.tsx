import type { AppConfig, SeatConfig } from "./types";

type Props = {
  config: AppConfig;
  seats: SeatConfig[];
  setSeats: (seats: SeatConfig[]) => void;
  seed: number;
  setSeed: (seed: number) => void;
  device: string;
  setDevice: (device: string) => void;
  biddingModel: string;
  setBiddingModel: (path: string) => void;
  simulations: number;
  setSimulations: (value: number) => void;
  determinizations: number;
  setDeterminizations: (value: number) => void;
  reveal: boolean;
  setReveal: (value: boolean) => void;
  speedMs: number;
  setSpeedMs: (value: number) => void;
  busy: boolean;
  onStart: () => void;
};

const labels: Record<string, string> = {
  human: "Mensch",
  random: "Random",
  mcts: "MCTS",
  "belief-mcts": "Belief-MCTS",
  policy: "Policy-Netz",
  dqn: "DQN",
};

function checkpointGroup(kind: string): string | null {
  if (kind === "policy") return "policy";
  if (kind === "belief-mcts") return "belief";
  if (kind === "dqn") return "dqn";
  return null;
}

export default function Setup(props: Props) {
  const updateSeat = (index: number, patch: Partial<SeatConfig>) => {
    const next = props.seats.map((seat, seatIndex) =>
      seatIndex === index ? { ...seat, ...patch } : seat,
    );
    props.setSeats(next);
  };

  return (
    <main className="setup-shell">
      <section className="setup-hero">
        <p className="eyebrow">LOCAL SKAT LAB</p>
        <h1>Setz dich an den Tisch.</h1>
        <p>
          Spiele selbst oder beobachte deine Agents Zug für Zug. Alle Spiele laufen
          lokal auf deiner Maschine.
        </p>
      </section>

      <section className="setup-panel">
        <div className="seat-configs">
          {props.seats.map((seat, index) => {
            const group = checkpointGroup(seat.kind);
            const checkpoints = group ? props.config.checkpoints[group] ?? [] : [];
            return (
              <article className="seat-config" key={index}>
                <div className="seat-number">0{index + 1}</div>
                <label>
                  Spieler
                  <select
                    value={seat.kind}
                    onChange={(event) => {
                      const kind = event.target.value;
                      const checkpointKind = checkpointGroup(kind);
                      const checkpoint = checkpointKind
                        ? props.config.checkpoints[checkpointKind]?.[0] ?? ""
                        : "";
                      updateSeat(index, {
                        kind,
                        label: labels[kind] ?? kind,
                        checkpoint,
                      });
                    }}
                  >
                    {props.config.agents.map((agent) => (
                      <option value={agent} key={agent}>
                        {labels[agent] ?? agent}
                      </option>
                    ))}
                  </select>
                </label>

                {group && (
                  <label>
                    Checkpoint
                    <input
                      list={`checkpoints-${index}`}
                      value={seat.checkpoint ?? ""}
                      onChange={(event) =>
                        updateSeat(index, { checkpoint: event.target.value })
                      }
                      placeholder={`${group}.pt`}
                    />
                    <datalist id={`checkpoints-${index}`}>
                      {checkpoints.map((path) => (
                        <option value={path} key={path} />
                      ))}
                    </datalist>
                  </label>
                )}

                {seat.kind === "dqn" && (
                  <label>
                    Architektur
                    <select
                      value={seat.architecture ?? "mlp"}
                      onChange={(event) =>
                        updateSeat(index, { architecture: event.target.value })
                      }
                    >
                      <option value="mlp">MLP</option>
                      <option value="transformer">Transformer</option>
                    </select>
                  </label>
                )}

                {seat.kind !== "human" && (
                  <label>
                    Reizen
                    <select
                      value={seat.bidding}
                      onChange={(event) =>
                        updateSeat(index, { bidding: event.target.value })
                      }
                    >
                      {props.config.biddingStyles.map((style) => (
                        <option value={style} key={style}>
                          {style}
                        </option>
                      ))}
                    </select>
                  </label>
                )}

                {seat.bidding === "neural" && seat.kind !== "human" && (
                  <label>
                    Schwelle
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      value={seat.biddingThreshold}
                      onChange={(event) =>
                        updateSeat(index, {
                          biddingThreshold: Number(event.target.value),
                        })
                      }
                    />
                  </label>
                )}
              </article>
            );
          })}
        </div>

        <div className="advanced-config">
          <label>
            Seed
            <input type="number" value={props.seed} onChange={(e) => props.setSeed(Number(e.target.value))} />
          </label>
          <label>
            Device
            <input value={props.device} onChange={(e) => props.setDevice(e.target.value)} />
          </label>
          <label>
            Bidding-Modell
            <input
              list="bidding-checkpoints"
              value={props.biddingModel}
              onChange={(e) => props.setBiddingModel(e.target.value)}
              placeholder="bidding.pt"
            />
            <datalist id="bidding-checkpoints">
              {(props.config.checkpoints.bidding ?? []).map((path) => (
                <option value={path} key={path} />
              ))}
            </datalist>
          </label>
          <label>
            MCTS Simulationen
            <input type="number" min="1" value={props.simulations} onChange={(e) => props.setSimulations(Number(e.target.value))} />
          </label>
          <label>
            Determinisierungen
            <input type="number" min="1" value={props.determinizations} onChange={(e) => props.setDeterminizations(Number(e.target.value))} />
          </label>
          <label>
            Zugtempo
            <select value={props.speedMs} onChange={(e) => props.setSpeedMs(Number(e.target.value))}>
              <option value={0}>sofort</option>
              <option value={200}>schnell</option>
              <option value={450}>normal</option>
              <option value={900}>langsam</option>
            </select>
          </label>
          <label className="check-label">
            <input type="checkbox" checked={props.reveal} onChange={(e) => props.setReveal(e.target.checked)} />
            Agentenkarten aufdecken
          </label>
        </div>

        <button className="primary-button" type="button" onClick={props.onStart} disabled={props.busy}>
          {props.busy ? "Tisch wird vorbereitet …" : "Runde starten"}
        </button>
      </section>
    </main>
  );
}
