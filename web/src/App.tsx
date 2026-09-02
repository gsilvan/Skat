import { useEffect, useRef, useState } from "react";
import { control, createSession, loadConfig, stopSession, submitAction } from "./api";
import Setup from "./Setup";
import Table from "./Table";
import type { AppConfig, GameState, SeatConfig } from "./types";

const defaultSeats: SeatConfig[] = [
  { kind: "human", label: "Du", bidding: "conservative", biddingThreshold: 0.65 },
  { kind: "random", label: "Random", bidding: "conservative", biddingThreshold: 0.65 },
  { kind: "random", label: "Random", bidding: "conservative", biddingThreshold: 0.65 },
];

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [seats, setSeats] = useState<SeatConfig[]>(defaultSeats);
  const [seed, setSeed] = useState(42);
  const [device, setDevice] = useState("cpu");
  const [biddingModel, setBiddingModel] = useState("");
  const [simulations, setSimulations] = useState(600);
  const [determinizations, setDeterminizations] = useState(20);
  const [speedMs, setSpeedMs] = useState(450);
  const [reveal, setReveal] = useState(false);
  const [game, setGame] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pressSelection, setPressSelection] = useState<number[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    loadConfig()
      .then((loaded) => {
        setConfig(loaded);
        setBiddingModel(loaded.checkpoints.bidding?.[0] ?? "");
      })
      .catch((reason) => setError(String(reason)));
    const sprite = new Image();
    sprite.onload = () => document.documentElement.classList.add("has-card-sprite");
    sprite.src = "/cards/skatstube-g1@2x.png";
  }, []);

  useEffect(() => {
    if (!game?.id) return;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/api/sessions/${game.id}/events`);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      const next = JSON.parse(event.data) as GameState;
      setGame(next);
      if (next.pending?.kind !== "press") setPressSelection([]);
    };
    socket.onerror = () => setError("Die Live-Verbindung zum Tisch wurde unterbrochen.");
    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [game?.id]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const configuredSeats = seats.map((seat) => ({
        ...seat,
        ...(seat.bidding === "neural" ? { biddingModel } : {}),
      }));
      setGame(
        await createSession({
          seats: configuredSeats,
          seed,
          device,
          simulations,
          determinizations,
          speedMs,
          reveal,
        }),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const action = async (value: unknown) => {
    if (!game) return;
    try {
      await submitAction(game.id, value);
      if (game.pending?.kind === "press") setPressSelection([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const cardAction = (id: number) => {
    if (game?.pending?.kind === "press") {
      setPressSelection((current) =>
        current.includes(id)
          ? current.filter((value) => value !== id)
          : current.length < 2
            ? [...current, id]
            : current,
      );
    } else {
      void action(id);
    }
  };

  const sendControl = async (command: string, value?: unknown) => {
    if (!game) return;
    try {
      setGame(await control(game.id, command, value));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const exit = () => {
    socketRef.current?.close();
    if (game && !["finished", "passed", "error", "stopped"].includes(game.status)) {
      void stopSession(game.id);
    }
    setGame(null);
    setPressSelection([]);
  };

  if (!config) {
    return <div className="loading-screen">{error ?? "Kartentisch wird geladen …"}</div>;
  }

  return (
    <>
      {error && <div className="error-toast" onClick={() => setError(null)}>{error}</div>}
      {game ? (
        <Table
          game={game}
          pressSelection={pressSelection}
          onCard={cardAction}
          onAction={(value) => void action(value)}
          onControl={(command, value) => void sendControl(command, value)}
          onExit={exit}
        />
      ) : (
        <Setup
          config={config}
          seats={seats}
          setSeats={setSeats}
          seed={seed}
          setSeed={setSeed}
          device={device}
          setDevice={setDevice}
          biddingModel={biddingModel}
          setBiddingModel={setBiddingModel}
          simulations={simulations}
          setSimulations={setSimulations}
          determinizations={determinizations}
          setDeterminizations={setDeterminizations}
          reveal={reveal}
          setReveal={setReveal}
          speedMs={speedMs}
          setSpeedMs={setSpeedMs}
          busy={busy}
          onStart={() => void start()}
        />
      )}
    </>
  );
}
