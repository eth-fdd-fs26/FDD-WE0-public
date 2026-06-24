// WebSocket + REST helpers. In dev (vite on :5173) the backend is on :8000;
// in the built app FastAPI serves both from the same origin.
const backendHost = import.meta.env.DEV ? `${location.hostname}:8000` : location.host;
const httpBase = import.meta.env.DEV ? `http://${location.hostname}:8000` : "";

export function runDay(onEvent, onClose, pace = 0.7) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${backendHost}/ws/run-day?pace=${pace}`);
  ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  ws.onclose = () => onClose && onClose();
  ws.onerror = () => onClose && onClose("error");
  return ws;
}

export async function fetchState() {
  const res = await fetch(`${httpBase}/api/state`);
  return res.json();
}
