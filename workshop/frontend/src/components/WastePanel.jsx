import React from "react";
import { SearchChart, Scatter, MACHINE_COLORS } from "./charts.jsx";

export default function WastePanel({ part }) {
  const s = part.steps || {};
  // only read a step's data once it reports "ok" (the "running" beat is empty)
  const done = (id) => (s[id]?.status === "ok" ? s[id].payload : null);
  const repair = done("waste.repair");
  const search = done("waste.search");
  const cfg = done("waste.configure");
  const route = done("waste.route");

  return (
    <div className={`panel full ${part.phase}`}>
      <h2>♻️ Waste Machines
        <span className={`badge ${part.source === "student" ? "student" : ""}`}>{part.source}</span>
      </h2>
      <div className="status-line">{part.line || "Standing by — repair tomorrow's log, then cluster the waste and configure the machines."}</div>

      {repair && (
        <div className="metrics">
          <div className="metric"><div className="k">Gaps filled</div><div className="v">{repair.missing_quantity + repair.missing_critical}</div></div>
          <div className={`metric ${repair.r2 >= repair.r2_min ? "good" : "bad"}`}>
            <div className="k">Quantity R²</div><div className="v">{repair.r2}</div></div>
          <div className={`metric ${repair.accuracy >= repair.acc_min ? "good" : "bad"}`}>
            <div className="k">Critical acc</div><div className="v">{Math.round(repair.accuracy * 100)}%</div></div>
        </div>
      )}

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 4 }}>
        <div>
          <div className="legend" style={{ marginBottom: 4 }}>
            <span><i style={{ background: "#6f7bf0" }} />silhouette (peak = best k)</span>
            <span><i style={{ background: "#8794ad" }} />inertia (elbow)</span>
          </div>
          {search
            ? <SearchChart ks={search.ks} silhouettes={search.silhouettes} inertias={search.inertias} bestK={search.best_k} />
            : <div className="waitnote">Searching for the right number of machines…</div>}
        </div>
        <div>
          <div className="legend" style={{ marginBottom: 4 }}>
            <span>● existing batches</span><span>★ tonight's arrivals</span>
          </div>
          {route
            ? <Scatter train={route.train_points} incoming={route.incoming_points} k={route.k} />
            : <div className="waitnote">Routing map appears once arrivals are dispatched…</div>}
        </div>
      </div>

      {cfg && (
        <div className="machines">
          {cfg.machines.map((m) => (
            <div key={m.id} className="mcard" style={{ borderLeftColor: MACHINE_COLORS[m.id % MACHINE_COLORS.length] }}>
              <div className="mh">🛠️ Machine {m.id}
                <span className={`tag ${m.danger ? "crit" : "ok"}`}>{m.danger ? "⚠️ CRITICAL" : "routine"}</span>
              </div>
              <div className="ml">{m.load_kg.toLocaleString()} kg · {m.count} batches{route ? ` · +${route.counts[m.id] || 0} new` : ""}</div>
              <div className="row"><span>🛡️ Shielding</span><b>{m.shielding}</b></div>
              <div className="row"><span>🌡️ Heat</span><b>{m.temperature}</b></div>
              <div className="row"><span>⚙️ Half-life</span><b>{m.half_life}</b></div>
              <div className="row"><span>🧪 Class</span><b>{m.chemical}</b></div>
              {m.critical_share != null && (
                <div className="row"><span>☢️ Critical</span><b>{m.critical_share}%</b></div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
