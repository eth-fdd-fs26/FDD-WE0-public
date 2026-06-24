import React, { useEffect, useReducer, useRef, useState } from "react";
import { runDay, fetchState } from "./api/ws.js";
import ReactorCore from "./components/ReactorCore.jsx";
import DayClock from "./components/DayClock.jsx";
import PumpPanel from "./components/PumpPanel.jsx";
import BasinPanel from "./components/BasinPanel.jsx";
import WastePanel from "./components/WastePanel.jsx";
import MissionLog from "./components/MissionLog.jsx";
import MeltdownScreen from "./components/MeltdownScreen.jsx";

// seconds-per-beat the backend paces with (results then linger ~1.7×)
const SPEEDS = { Slow: 1.1, Normal: 0.6, Fast: 0.3 };

const emptyParts = () => ({
  1: { source: "reference", steps: {}, exploded: false, line: "" },
  2: { source: "reference", steps: {}, exploded: false, line: "" },
  3: { source: "reference", steps: {}, exploded: false, line: "" },
});

const initial = {
  status: "idle", activePart: 0, events: [],
  parts: emptyParts(), failure: null, title: "",
};

function reducer(state, action) {
  switch (action.type) {
    case "sources": {
      const parts = emptyParts();
      action.parts.forEach((p) => { parts[p.part].source = p.source; });
      return { ...initial, parts };
    }
    case "start":
      return { ...initial, status: "running",
        parts: Object.fromEntries(Object.entries(state.parts).map(
          ([k, v]) => [k, { ...v, steps: {}, exploded: false, line: "" }])) };
    case "event": {
      const ev = action.ev;
      const next = { ...state, events: [...state.events, ev] };
      const parts = { ...state.parts };

      if (ev.step_id?.endsWith(".start") && ev.payload?.part) {
        const p = ev.payload.part;
        parts[p] = { ...parts[p], source: ev.payload.source || parts[p].source };
        next.activePart = p;
      }
      if (ev.part >= 1 && ev.part <= 3) {
        const p = ev.part;
        parts[p] = {
          ...parts[p],
          steps: { ...parts[p].steps, [ev.step_id]: ev },
          line: ev.message || parts[p].line,
          exploded: parts[p].exploded || ev.status === "exploded",
        };
        next.activePart = p;
      }
      if (ev.status === "running" && ev.title) next.title = ev.title;
      if (ev.status === "exploded")
        next.failure = { part: ev.part, title: ev.title, message: ev.message };
      if (ev.step_id === "day.victory") next.status = "victory";
      if (ev.step_id === "day.meltdown") next.status = "meltdown";
      if (ev.step_id === "stream.error") {
        next.status = "meltdown";
        next.failure = { part: ev.part, title: ev.title, message: ev.message };
      }
      next.parts = parts;
      return next;
    }
    default:
      return state;
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial);
  const [speed, setSpeed] = useState("Slow");
  const [reportOpen, setReportOpen] = useState(true);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchState().then((d) => dispatch({ type: "sources", parts: d.parts })).catch(() => {});
    return () => wsRef.current && wsRef.current.close();
  }, []);

  const start = () => {
    if (wsRef.current) wsRef.current.close();
    setReportOpen(true);
    dispatch({ type: "start" });
    wsRef.current = runDay((ev) => dispatch({ type: "event", ev }), null, SPEEDS[speed]);
  };

  const dayOver = state.status === "victory" || state.status === "meltdown";

  const panelPhase = (part) => {
    const p = state.parts[part];
    if (p.exploded) return "dead";
    if (state.status === "victory") return "alive";
    if (part < state.activePart) return "alive";
    if (part === state.activePart && state.status === "running") return "busy";
    return "idle";
  };
  const clockPhase = (part) => ({ dead: "dead", alive: "ok", busy: "active", idle: "" }[panelPhase(part)]);

  const partProps = (part) => ({ ...state.parts[part], phase: panelPhase(part) });
  const running = state.status === "running";
  const lastLine = state.events.length ? state.events[state.events.length - 1].message : "";

  return (
    <div className="app">
      <div className="topbar">
        <div>
          <h1>☢️ Nuclear Central Manager</h1>
          <div className="sub">Keep the plant alive for one full day — pump, basin & waste must all hold.</div>
        </div>
        <div className="spacer" />
        <DayClock phaseOf={clockPhase} />
        <span className={`statuschip ${state.status}`}>
          {state.status === "idle" && "● standby"}
          {state.status === "running" && "● shift in progress"}
          {state.status === "victory" && "● day survived"}
          {state.status === "meltdown" && "● meltdown"}
        </span>
        <div className="speed" title="Playback speed">
          {Object.keys(SPEEDS).map((s) => (
            <button key={s} className={speed === s ? "on" : ""} disabled={running}
              onClick={() => setSpeed(s)}>{s}</button>
          ))}
        </div>
        {dayOver && !reportOpen && (
          <button className="btn ghost" onClick={() => setReportOpen(true)}>📋 Report</button>
        )}
        <button className="btn" onClick={start} disabled={running}>
          {running ? "Running…" : state.status === "idle" ? "▶ Start shift" : "↻ Restart"}
        </button>
      </div>

      <div className="grid">
        <PumpPanel part={partProps(1)} />
        <ReactorCore status={state.status} activePart={state.activePart} line={lastLine} title={state.title} />
        <BasinPanel part={partProps(2)} />
      </div>

      <WastePanel part={partProps(3)} />

      <MissionLog events={state.events} />

      <MeltdownScreen
        status={state.status}
        failure={state.failure}
        events={state.events}
        onRestart={start}
        open={reportOpen}
        onClose={() => setReportOpen(false)}
      />
    </div>
  );
}
