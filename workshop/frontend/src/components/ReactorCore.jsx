import React from "react";

// The central "display core / engine". Colour + animation reflect plant health.
export default function ReactorCore({ status, activePart, line, title }) {
  let coreA = "#55a868", coreB = "#1c2a1f", label = "Core stable";
  if (status === "idle") { coreA = "#3a4a63"; coreB = "#161d2b"; label = "Cold start"; }
  if (status === "running") { coreA = "#6f7bf0"; coreB = "#1a2238"; label = title || "Running diagnostics…"; }
  if (status === "meltdown") { coreA = "#c44e52"; coreB = "#2a1416"; label = "CORE BREACH"; }
  if (status === "victory") { coreA = "#55a868"; coreB = "#16261b"; label = "Shift complete ☀️"; }

  const phaseNames = { 1: "Coolant pump", 2: "Basin temperature", 3: "Waste machines" };

  return (
    <div className="panel core-wrap">
      <h2>☢️ Reactor Core</h2>
      <div className={`reactor ${status === "running" ? "busy" : ""} ${status === "meltdown" ? "dead" : ""}`}>
        <div className="ring" />
        <div className="pit" style={{ "--coreA": coreA, "--coreB": coreB }} />
      </div>
      <div className="core-label">{label}</div>
      <div className="core-sub">
        {status === "running" && activePart ? `▶ ${phaseNames[activePart]}` : ""}
        {status === "idle" ? "Press Start shift to bring the plant online" : ""}
      </div>
      <div className="core-sub" style={{ marginTop: 6 }}>{line}</div>
    </div>
  );
}
