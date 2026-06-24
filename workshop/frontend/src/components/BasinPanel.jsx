import React from "react";

export default function BasinPanel({ part }) {
  const s = part.steps || {};
  // a step carries a payload once it's ok OR exploded (the "running" beat is empty)
  const payloadOf = (id) =>
    s[id] && s[id].payload && Object.keys(s[id].payload).length ? s[id].payload : null;
  const arch = payloadOf("basin.arch");
  const wiring = payloadOf("basin.wiring");
  const checks = [...(arch?.checks || []), ...(wiring?.checks || [])];

  return (
    <div className={`panel ${part.phase}`}>
      <h2>🌡️ Basin Temperature
        <span className={`badge ${part.source === "student" ? "student" : ""}`}>{part.source}</span>
      </h2>
      <div className="status-line">{part.line || "Standing by — finish the backup predictor's architecture so it can watch the basin."}</div>

      {checks.length === 0 && <div className="waitnote">Inspecting the architecture…</div>}

      {arch && arch.trainable_params != null && (
        <div className="metrics">
          <div className="metric">
            <div className="k">Fusion</div>
            <div className="v flow">{arch.signal_features}+{arch.thermal_features}→{arch.fusion_dim}→1</div>
          </div>
          <div className="metric good">
            <div className="k">Trainable</div>
            <div className="v">{arch.trainable_params.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="k">Frozen</div>
            <div className="v">{(arch.frozen_params ?? 0).toLocaleString()}</div>
          </div>
        </div>
      )}

      {checks.length > 0 && (
        <div className="checklist">
          {checks.map((c, i) => (
            <div key={i} className={`checkrow ${c.ok ? "ok" : "bad"}`}>
              <span className="ci">{c.ok ? "✓" : "✗"}</span>
              <span className="cn">{c.name}</span>
              <span className="cd">{c.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
