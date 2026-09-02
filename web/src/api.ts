import type { AppConfig, GameState, SeatConfig } from "./types";

async function json<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error ?? `HTTP ${response.status}`);
  }
  return body as T;
}

export async function loadConfig(): Promise<AppConfig> {
  return json(await fetch("/api/config"));
}

export async function createSession(options: {
  seats: SeatConfig[];
  seed: number;
  device: string;
  simulations: number;
  determinizations: number;
  speedMs: number;
  reveal: boolean;
}): Promise<GameState> {
  return json(
    await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    }),
  );
}

export async function submitAction(session: string, value: unknown): Promise<void> {
  await json(
    await fetch(`/api/sessions/${session}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    }),
  );
}

export async function control(
  session: string,
  command: string,
  value?: unknown,
): Promise<GameState> {
  return json(
    await fetch(`/api/sessions/${session}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, value }),
    }),
  );
}

export async function stopSession(session: string): Promise<void> {
  const response = await fetch(`/api/sessions/${session}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
}
