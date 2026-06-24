// Tiny dependency-free SVG charts shared by the subsystem panels.
import React from "react";

const C = { blue: "#6f7bf0", orange: "#dd8452", green: "#55a868", red: "#c44e52", muted: "#8794ad" };
export const MACHINE_COLORS = ["#6f7bf0", "#dd8452", "#55a868", "#c44e52", "#b07bd6", "#4bb0c4", "#d6a44b"];

function scale(vals, lo, hi, a, b) {
  const min = lo ?? Math.min(...vals), max = hi ?? Math.max(...vals);
  const span = max - min || 1;
  return (v) => a + ((v - min) / span) * (b - a);
}

// Multi-series line chart. series = [{ data:[...], color, label, dashed }]
// Optional axis props (omit for the original no-axis look):
//   xLabels  — [{ i, label }]  tick positions + strings along the x-axis
//   yTickCount — number of evenly-spaced y-axis ticks (needs yMin/yMax)
//   yFormat  — (value) => string  formatter for y-tick labels
export function LineChart({ series, width = 320, height = 130, yMin, yMax, hLine,
                            xLabels, yTickCount = 0, yFormat }) {
  const hasX = !!(xLabels?.length);
  const hasY = yTickCount > 0 && !!yFormat;
  const pad  = { l: hasY ? 30 : 6, r: 6, t: 8, b: hasX ? 20 : 8 };

  const n    = Math.max(...series.map((s) => s.data.length));
  const allY = series.flatMap((s) => s.data);
  const x    = scale([0, n - 1], 0, n - 1, pad.l, width - pad.r);
  const y    = scale(allY, yMin, yMax, height - pad.b, pad.t);

  const yLo    = yMin ?? Math.min(...allY);
  const yHi    = yMax ?? Math.max(...allY);
  const yTicks = hasY
    ? Array.from({ length: yTickCount }, (_, i) => yLo + (yHi - yLo) * i / (yTickCount - 1))
    : [];

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      {/* y-axis grid lines + labels */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={pad.l} x2={width - pad.r} y1={y(v)} y2={y(v)}
            stroke={C.muted} strokeWidth="0.5" opacity="0.25" />
          <text x={pad.l - 4} y={y(v) + 3.5} textAnchor="end" fontSize="8" fill={C.muted}>
            {yFormat(v)}
          </text>
        </g>
      ))}

      {hLine != null && (
        <line x1={pad.l} x2={width - pad.r} y1={y(hLine)} y2={y(hLine)}
          stroke={C.red} strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
      )}

      {series.map((s, i) => {
        const d = s.data.map((v, j) => `${j === 0 ? "M" : "L"}${x(j).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
        return <path key={i} d={d} fill="none" stroke={s.color} strokeWidth="2"
          strokeDasharray={s.dashed ? "5 3" : "0"} opacity="0.95" />;
      })}

      {/* x-axis tick labels */}
      {hasX && xLabels.map(({ i, label }) => (
        <text key={i} x={x(i)} y={height - 4} textAnchor="middle" fontSize="8" fill={C.muted}>
          {label}
        </text>
      ))}
    </svg>
  );
}

// Silhouette/inertia search chart with the chosen k highlighted.
export function SearchChart({ ks, silhouettes, inertias, bestK, width = 360, height = 150 }) {
  const pad = { l: 8, r: 8, t: 12, b: 22 };
  const x = scale(ks, ks[0], ks[ks.length - 1], pad.l, width - pad.r);
  const ys = scale(silhouettes, Math.min(...silhouettes), Math.max(...silhouettes), height - pad.b, pad.t);
  const yi = scale(inertias, Math.min(...inertias), Math.max(...inertias), height - pad.b, pad.t);
  const silD = ks.map((k, j) => `${j === 0 ? "M" : "L"}${x(k)},${ys(silhouettes[j])}`).join(" ");
  const inD = ks.map((k, j) => `${j === 0 ? "M" : "L"}${x(k)},${yi(inertias[j])}`).join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      <line x1={x(bestK)} x2={x(bestK)} y1={pad.t} y2={height - pad.b} stroke={C.green} strokeWidth="2" opacity="0.6" />
      <path d={inD} fill="none" stroke={C.muted} strokeWidth="1.6" strokeDasharray="4 3" />
      <path d={silD} fill="none" stroke={C.blue} strokeWidth="2.4" />
      {ks.map((k) => (
        <g key={k}>
          <circle cx={x(k)} cy={ys(silhouettes[ks.indexOf(k)])} r={k === bestK ? 5 : 3}
            fill={k === bestK ? C.green : C.blue} />
          <text x={x(k)} y={height - 7} fontSize="9" fill={C.muted} textAnchor="middle">{k}</text>
        </g>
      ))}
    </svg>
  );
}

// PCA scatter: existing batches (dots by machine) + incoming (stars by machine).
export function Scatter({ train, incoming, k, width = 360, height = 200 }) {
  const pad = 12;
  const xs = [...train, ...incoming].map((p) => p.x);
  const ys = [...train, ...incoming].map((p) => p.y);
  const x = scale(xs, Math.min(...xs), Math.max(...xs), pad, width - pad);
  const y = scale(ys, Math.min(...ys), Math.max(...ys), height - pad, pad);
  const col = (m) => MACHINE_COLORS[m % MACHINE_COLORS.length];
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      {train.map((p, i) => (
        <circle key={i} cx={x(p.x)} cy={y(p.y)} r="2.4" fill={col(p.m)} opacity="0.5" />
      ))}
      {incoming.map((p, i) => (
        <g key={`i${i}`} transform={`translate(${x(p.x)},${y(p.y)})`}>
          <path d="M0,-6 L1.8,-1.8 L6,-1.8 L2.6,1.4 L3.8,6 L0,3 L-3.8,6 L-2.6,1.4 L-6,-1.8 L-1.8,-1.8 Z"
            fill={col(p.m)} stroke="#0c1018" strokeWidth="0.6" />
        </g>
      ))}
    </svg>
  );
}

// Horizontal gauge bar 0..1 with a pass threshold marker.
export function Gauge({ value, threshold, good = true, width = 150 }) {
  const w = Math.max(0, Math.min(1, value)) * width;
  return (
    <svg viewBox={`0 0 ${width} 10`} width="100%" height="10">
      <rect x="0" y="2" width={width} height="6" rx="3" fill="#2a3446" />
      <rect x="0" y="2" width={w} height="6" rx="3" fill={good ? C.green : C.red} />
      {threshold != null && (
        <line x1={threshold * width} x2={threshold * width} y1="0" y2="10" stroke="#e7ecf5" strokeWidth="1.4" />
      )}
    </svg>
  );
}
