import React from "react";
import { LineChart, Gauge } from "./charts.jsx";

// Dot + label tracking which pipeline phase is active / done / errored
function PipelineStep({ icon, label, status }) {
  const dot  = { idle: "var(--line)", running: "var(--blue)", done: "var(--green)", error: "var(--red)" };
  const text = { idle: "var(--muted)", running: "var(--blue)", done: "var(--green)", error: "var(--red)" };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <svg width="9" height="9" viewBox="0 0 9 9" style={{ flexShrink: 0 }}>
        <circle cx="4.5" cy="4.5" r="4.5" fill={dot[status] ?? dot.idle} />
      </svg>
      <span style={{ fontSize: 11, color: text[status] ?? text.idle }}>{icon} {label}</span>
    </div>
  );
}

// Horizontal bar: green portion = fraction of rows that survived cleaning
function RowBar({ before, after }) {
  if (!before) return null;
  const w = Math.round(220 * Math.min(1, after / before));
  return (
    <svg viewBox="0 0 220 10" width="100%" height="10">
      <rect x="0" y="2" width="220" height="6" rx="3" fill="var(--line)" />
      <rect x="0" y="2" width={w}   height="6" rx="3" fill="var(--green)" />
    </svg>
  );
}

// Compact preview of a handful of raw-log rows; nulls rendered red to show the mess
function LogPreview({ rows }) {
  if (!rows?.length) return null;
  const keys = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ fontSize: 9, borderCollapse: "collapse", width: "100%", color: "var(--muted)" }}>
        <thead>
          <tr>
            {keys.map(k => (
              <th key={k} style={{
                padding: "2px 5px", textAlign: "left",
                borderBottom: "1px solid var(--line)", whiteSpace: "nowrap",
              }}>
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 3).map((r, i) => (
            <tr key={i}>
              {keys.map(k => (
                <td key={k} style={{
                  padding: "2px 5px", whiteSpace: "nowrap",
                  color: r[k] == null ? "var(--red)" : "var(--ink)",
                }}>
                  {r[k] == null ? "—" : typeof r[k] === "number" ? r[k].toFixed(1) : r[k]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, letterSpacing: "0.6px",
      textTransform: "uppercase", color: "var(--muted)",
    }}>
      {children}
    </div>
  );
}

export default function PumpPanel({ part }) {
  const s = part.steps || {};
  // pump.load is emitted as "running" (no separate ok beat) — payload is available immediately
  const load  = s["pump.load"]?.payload;
  const clean = s["pump.clean"]?.status  === "ok" ? s["pump.clean"].payload  : null;
  const pred  = s["pump.predict"]?.status === "ok" ? s["pump.predict"].payload : null;

  // Derive per-step status for the pipeline tracker
  const loadSt = load
    ? "done"
    : s["pump.load"] ? "running" : "idle";
  const cleanSt = !s["pump.clean"] ? "idle"
    : s["pump.clean"].status === "ok"       ? "done"
    : s["pump.clean"].status === "exploded" ? "error"
    : "running";
  const predSt = !s["pump.predict"] ? "idle"
    : s["pump.predict"].status === "ok"       ? "done"
    : s["pump.predict"].status === "exploded" ? "error"
    : "running";

  const failRate = load?.rows > 0
    ? `${(load.failures / load.rows * 100).toFixed(1)}%`
    : null;

  return (
    <div className={`panel ${part.phase}`}>
      <h2>🛠️ Coolant Pump
        <span className={`badge ${part.source === "student" ? "student" : ""}`}>{part.source}</span>
      </h2>
      <div className="status-line">
        {part.line || "Standing by — clean the messy logs, then predict pump failure."}
      </div>

      {/* ── Pipeline progress tracker ── */}
      <div style={{
        display: "flex", flexDirection: "column", gap: 5,
        padding: "8px 10px", background: "var(--panel-2)",
        borderRadius: 10, border: "1px solid var(--line)",
      }}>
        <PipelineStep icon="🔍" label="Inspect raw logs"          status={loadSt}  />
        <PipelineStep icon="🧹" label="Reformat & impute"         status={cleanSt} />
        <PipelineStep icon="📉" label="Discard outliers (RANSAC)" status={cleanSt} />
        <PipelineStep icon="🤖" label="Predict failure"           status={predSt}  />
      </div>

      {/* ── Step 1: raw log inspection ── */}
      {load ? (
        <>
          <SectionLabel>🔍 Raw Log</SectionLabel>
          <div className="metrics">
            <div className="metric">
              <div className="k">Rows</div>
              <div className="v">{load.rows.toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="k">Failure rate</div>
              <div className="v" style={{ fontSize: 16 }}>{failRate}</div>
            </div>
            <div className="metric bad">
              <div className="k">Missing cells</div>
              <div className="v">{load.nan_cells.toLocaleString()}</div>
            </div>
            <div className="metric bad">
              <div className="k">Dupes</div>
              <div className="v">{load.dup_rows}</div>
            </div>
          </div>
          {load.preview && <LogPreview rows={load.preview} />}
        </>
      ) : (
        <div className="waitnote">Awaiting the night shift…</div>
      )}

      {/* ── Steps 2 & 3: cleaning + RANSAC ── */}
      {clean && (
        <>
          <SectionLabel>🧹 After Cleaning &amp; RANSAC</SectionLabel>
          <div className="metrics">
            <div className="metric good">
              <div className="k">Rows kept</div>
              <div className="v">{clean.rows_after.toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="k">Gaps filled</div>
              <div className="v">{clean.nan_before}</div>
            </div>
            <div className="metric">
              <div className="k">Dropped</div>
              <div className="v">{clean.dupes_removed}</div>
            </div>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>
            {clean.rows_before.toLocaleString()} → {clean.rows_after.toLocaleString()} rows
            &nbsp;(dedup + RANSAC sensor screen)
          </div>
          <RowBar before={clean.rows_before} after={clean.rows_after} />
        </>
      )}

      {/* ── Step 4: prediction results ── */}
      {pred && (
        <>
          <div className="metrics">
            <div className={`metric ${pred.recall >= pred.recall_min ? "good" : "bad"}`}>
              <div className="k">Recall</div>
              <div className="v">{Math.round(pred.recall * 100)}%</div>
              <Gauge value={pred.recall} threshold={pred.recall_min} good={pred.recall >= pred.recall_min} />
            </div>
            <div className={`metric ${pred.precision >= pred.precision_min ? "good" : "bad"}`}>
              <div className="k">Precision</div>
              <div className="v">{Math.round(pred.precision * 100)}%</div>
              <Gauge value={pred.precision} threshold={pred.precision_min} good={pred.precision >= pred.precision_min} />
            </div>
          </div>
          <div className="cm">
            <div className="cell tp"><div className="n">{pred.confusion.tp}</div><div className="l">caught ✔</div></div>
            <div className="cell fn"><div className="n">{pred.confusion.fn}</div><div className="l">missed ✘</div></div>
            <div className="cell"><div className="n">{pred.confusion.fp}</div><div className="l">false alarm</div></div>
            <div className="cell"><div className="n">{pred.confusion.tn.toLocaleString()}</div><div className="l">safe ✔</div></div>
          </div>
          <div className="legend">
            <span><i style={{ background: "#c44e52" }} />actual failures</span>
            <span><i style={{ background: "#6f7bf0" }} />flagged risk</span>
          </div>
          <LineChart
            yMin={0} yMax={1}
            yTickCount={3}
            yFormat={(v) => `${Math.round(v * 100)}%`}
            xLabels={[
              { i: 0,  label: "0h"  },
              { i: 6,  label: "6h"  },
              { i: 12, label: "12h" },
              { i: 18, label: "18h" },
              { i: 23, label: "24h" },
            ]}
            series={[
              { data: pred.real_curve, color: "#c44e52" },
              { data: pred.risk_curve, color: "#6f7bf0" },
            ]}
          />
          <div style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.5 }}>
            Shift split into 24 hourly buckets. Each point is the fraction of rows
            flagged (blue) vs actually failing (red) in that hour. Blue tracking red
            spikes = failures caught; blue flat while red spikes = missed detections.
          </div>
        </>
      )}
    </div>
  );
}
