import React from "react";

const PHASES = [
  { part: 1, icon: "🛠️", name: "Pump" },
  { part: 2, icon: "🌡️", name: "Basin" },
  { part: 3, icon: "♻️", name: "Waste" },
];

// A little shift timeline: each subsystem lights up as the day plays out.
export default function DayClock({ phaseOf }) {
  return (
    <div className="clock">
      <span className="sub">06:00</span>
      {PHASES.map((p) => (
        <div key={p.part} className={`phase ${phaseOf(p.part)}`}>
          <span className="dot" />{p.icon} {p.name}
        </div>
      ))}
      <span className="sub">→ dawn</span>
    </div>
  );
}
