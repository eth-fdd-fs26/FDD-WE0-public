import React from "react";

// Overlay shown when the day ends — meltdown (with the failure dashboard) or victory.
// Closable (✕ / click-outside) so you can inspect the panels + log behind it.
export default function MeltdownScreen({ status, failure, events, onRestart, open, onClose }) {
  if ((status !== "meltdown" && status !== "victory") || !open) return null;
  const checks = events.filter((e) => e.status === "ok" || e.status === "exploded");

  return (
    <div className={`overlay ${status}`} onClick={onClose}>
      <div className="card" onClick={(e) => e.stopPropagation()}>
        <button className="overlay-close" onClick={onClose} title="Close — inspect the panels & log">✕</button>
        {status === "meltdown" ? (
          <>
            <h2>☢️ MELTDOWN</h2>
            <div className="why">
              <b>{failure?.title || "A subsystem failed."}</b><br />
              {failure?.message}
            </div>
          </>
        ) : (
          <>
            <h2>🏁 You survived the day</h2>
            <div className="why">Every subsystem held through the night. The plant lives to glow another day. ☀️</div>
          </>
        )}

        <div className="recap">
          {checks.map((e, i) => (
            <div key={i} className={`r ${e.status}`}>
              <span className="ic">{e.status === "ok" ? "✅" : "💥"}</span>
              <span>{e.title}</span>
              <span className="t" style={{ marginLeft: "auto" }}>{e.message}</span>
            </div>
          ))}
        </div>

        <div className="overlay-actions">
          <button className="btn" onClick={onRestart}>
            {status === "meltdown" ? "↻ Restart the shift" : "↻ Run another day"}
          </button>
          <button className="btn ghost" onClick={onClose}>🔍 Close &amp; inspect</button>
        </div>
      </div>
    </div>
  );
}
