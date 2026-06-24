import React, { useEffect, useRef } from "react";

// A control-room console that narrates every beat of the day as it streams.
const ICON = { info: "🟦", running: "⏳", ok: "✅", exploded: "💥" };
const PART_COLOR = { 0: "#8794ad", 1: "#6f7bf0", 2: "#dd8452", 3: "#55a868" };

export default function MissionLog({ events }) {
  const bodyRef = useRef(null);
  const shown = events.filter((e) => e.step_id !== "stream.end" && e.step_id !== "stream.error");

  useEffect(() => {
    // Scroll only the log container, not the page — scrollIntoView would pull the
    // whole window down on every event and make the screen jump.
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length]);

  return (
    <div className="panel full log">
      <h2>🖥️ Mission Log <span className="badge">{shown.length} events</span></h2>
      <div className="logbody" ref={bodyRef}>
        {shown.length === 0 && (
          <div className="waitnote">Press ▶ Start shift to begin the day — each step is narrated here as it runs.</div>
        )}
        {shown.map((e, i) => (
          <div key={i} className={`logline ${e.status}`} style={{ borderLeftColor: PART_COLOR[e.part] ?? "#8794ad" }}>
            <span className="ic">{ICON[e.status] ?? "•"}</span>
            <span className="lt">{e.title}</span>
            <span className="lm">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
