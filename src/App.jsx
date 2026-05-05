import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";
const API_URL = "http://localhost:8000";

/* ─── Severity palette ─────────────────────────────────────────── */
const SEV = {
  CRITICAL: { pill: "#501313", pillText: "#F09595", row: "#1a0808", border: "#A32D2D", glyph: "◈", glyphColor: "#E24B4A" },
  HIGH:     { pill: "#412402", pillText: "#FAC775", row: "#180e00", border: "#854F0B", glyph: "▲", glyphColor: "#EF9F27" },
  MEDIUM:   { pill: "#04342C", pillText: "#5DCAA5", row: "#021a14", border: "#0F6E56", glyph: "◆", glyphColor: "#1D9E75" },
  LOW:      { pill: "#042C53", pillText: "#85B7EB", row: "#02111f", border: "#185FA5", glyph: "●", glyphColor: "#378ADD" },
  NORMAL:   { pill: "#2C2C2A", pillText: "#B4B2A9", row: "#111",    border: "#444441", glyph: "○", glyphColor: "#5F5E5A" },
};

/* ─── Sparkline ────────────────────────────────────────────────── */
function useSparkCanvas(data, key, color, maxPts = 80) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    const vals = data.slice(-maxPts).map(d => d[key] ?? 0);
    const max = Math.max(...vals, 0.001);
    const pts = vals.map((v, i) => [(i / Math.max(vals.length - 1, 1)) * w, h - (v / max) * h * 0.85 - 2]);
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = color + "18"; ctx.fill();
  }, [data, key, color, maxPts]);
  return ref;
}

function Spark({ data, valueKey, color, h = 44 }) {
  const ref = useSparkCanvas(data, valueKey, color);
  return <canvas ref={ref} width={400} height={h} style={{ width: "100%", height: h, display: "block" }} />;
}

/* ─── Donut chart ──────────────────────────────────────────────── */
const DONUT_COLORS = ["#A32D2D","#854F0B","#185FA5","#0F6E56","#533AB7","#993556","#3B6D11","#5F5E5A"];

function DonutChart({ counts }) {
  const ref = useRef(null);
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !entries.length) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 160, 160);
    let angle = -Math.PI / 2;
    entries.forEach(([, v], i) => {
      const slice = (v / total) * Math.PI * 2;
      ctx.beginPath(); ctx.moveTo(80, 80); ctx.arc(80, 80, 68, angle, angle + slice);
      ctx.closePath(); ctx.fillStyle = DONUT_COLORS[i % DONUT_COLORS.length]; ctx.fill();
      ctx.strokeStyle = "#0a0a0a"; ctx.lineWidth = 2; ctx.stroke();
      angle += slice;
    });
    ctx.beginPath(); ctx.arc(80, 80, 42, 0, Math.PI * 2);
    ctx.fillStyle = "#0a0a0a"; ctx.fill();
    ctx.fillStyle = "#ccc"; ctx.font = "bold 20px 'IBM Plex Mono'"; ctx.textAlign = "center";
    ctx.fillText(total, 80, 86);
    ctx.fillStyle = "#555"; ctx.font = "10px sans-serif"; ctx.fillText("total", 80, 100);
  }, [counts]);
  return (
    <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
      <canvas ref={ref} width={160} height={160} style={{ flexShrink: 0 }} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 7 }}>
        {entries.length ? entries.map(([type, count], i) => (
          <div key={type} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: DONUT_COLORS[i % DONUT_COLORS.length], flexShrink: 0 }} />
            <span style={{ fontSize: 11, color: "#aaa", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{type}</span>
            <span style={{ fontSize: 12, color: "#ddd", fontVariantNumeric: "tabular-nums" }}>{count}</span>
            <span style={{ fontSize: 10, color: "#555", width: 28, textAlign: "right" }}>{total ? Math.round(count / total * 100) : 0}%</span>
          </div>
        )) : <span style={{ fontSize: 12, color: "#444", fontStyle: "italic" }}>Aucune attaque détectée</span>}
      </div>
    </div>
  );
}

/* ─── RMSE timeline ────────────────────────────────────────────── */
function RmseTimeline({ data, threshold }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    const pts = data.slice(-100);
    const maxV = Math.max(...pts.map(d => d.rmse), threshold || 0.001, 0.001) * 1.1;
    ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 0.5;
    [0.25, 0.5, 0.75].forEach(f => { const y = h - f * h * 0.88; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); });
    if (threshold) {
      const ty = h - (threshold / maxV) * h * 0.88;
      ctx.fillStyle = "#854f0b11"; ctx.fillRect(0, 0, w, ty);
      ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(w, ty);
      ctx.strokeStyle = "#854F0B"; ctx.lineWidth = 1; ctx.setLineDash([5, 5]); ctx.stroke(); ctx.setLineDash([]);
    }
    pts.forEach((d, i) => {
      if (d.rmse > threshold && threshold) {
        const x = (i / (pts.length - 1)) * w;
        ctx.fillStyle = "#A32D2D22"; ctx.fillRect(x - 2, 0, 4, h);
      }
    });
    ctx.beginPath();
    pts.forEach((d, i) => {
      const x = (i / Math.max(pts.length - 1, 1)) * w;
      const y = h - (d.rmse / maxV) * h * 0.88;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#378ADD"; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = "#378ADD14"; ctx.fill();
  }, [data, threshold]);
  return <canvas ref={ref} width={800} height={130} style={{ width: "100%", height: 130, display: "block" }} />;
}

/* ─── Training bar ─────────────────────────────────────────────── */
function TrainingBar({ progress, trained }) {
  const pct = Math.min(Math.round((progress || 0) * 100), 100);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11 }}>
        <span style={{ color: "#666" }}>{trained ? "Surveillance active — modèle entraîné" : `Phase d'apprentissage… ${pct}%`}</span>
        <span style={{ color: trained ? "#1D9E75" : "#EF9F27", fontWeight: 600 }}>{trained ? "ACTIF" : `${pct}%`}</span>
      </div>
      <div style={{ background: "#1a1a1a", borderRadius: 2, height: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: trained ? "#1D9E75" : "#EF9F27", transition: "width 0.5s ease", boxShadow: trained ? "0 0 6px #1D9E7588" : "0 0 6px #EF9F2788" }} />
      </div>
    </div>
  );
}

/* ─── Stat card ────────────────────────────────────────────────── */
function Card({ label, value, sub, accent = "#85B7EB", icon }) {
  return (
    <div style={{ background: "#0d0d0d", border: "0.5px solid #1f1f1f", borderRadius: 10, padding: "14px 16px", flex: "1 1 0", minWidth: 0, position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 8, right: 12, fontSize: 18, opacity: 0.15 }}>{icon}</div>
      <div style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: accent, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#444", marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

/* ─── Alert row ────────────────────────────────────────────────── */
function AlertRow({ alert, idx }) {
  const cfg = SEV[alert.severity] || SEV.NORMAL;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px 62px 96px 1fr 100px 72px 62px", gap: 10, padding: "7px 14px", borderBottom: "0.5px solid #151515", fontSize: 11, alignItems: "center", fontFamily: "'IBM Plex Mono', monospace", background: idx % 2 === 0 ? "#0a0a0a" : "#0c0c0c", borderLeft: `2px solid ${cfg.border}` }}>
      <span style={{ color: cfg.glyphColor, fontSize: 9 }}>{cfg.glyph}</span>
      <span style={{ color: "#444", fontSize: 10 }}>{alert.ts_human}</span>
      <span style={{ background: cfg.pill, color: cfg.pillText, borderRadius: 4, padding: "1px 7px", fontSize: 9, fontWeight: 700, textAlign: "center", fontFamily: "sans-serif", letterSpacing: "0.08em" }}>{alert.severity}</span>
      <span style={{ color: "#bbb", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        <span style={{ color: "#666" }}>{alert.src_ip}</span>
        <span style={{ color: "#333", margin: "0 4px" }}>→</span>
        <span style={{ color: "#777" }}>{alert.dst_ip}</span>
        {alert.dport ? <span style={{ color: "#444" }}>:{alert.dport}</span> : null}
      </span>
      <span style={{ color: alert.is_attack ? "#F09595" : "#5DCAA5", fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{alert.attack_type || (alert.is_attack ? "Unknown" : "Anomalie")}</span>
      <span style={{ color: "#555", textAlign: "center" }}>{alert.proto}</span>
      <span style={{ color: "#888", textAlign: "right" }}>{(alert.confidence * 100).toFixed(0)}%</span>
    </div>
  );
}

/* ─── Ticker ───────────────────────────────────────────────────── */
function Ticker({ alerts }) {
  const latest = alerts[0];
  if (!latest) return null;
  const cfg = SEV[latest.severity] || SEV.NORMAL;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "5px 14px", background: "#0e0e0e", borderBottom: "0.5px solid #181818", fontSize: 11, fontFamily: "monospace", overflow: "hidden" }}>
      <span style={{ color: cfg.glyphColor, fontSize: 8, animation: "pulse 1.2s ease-in-out infinite" }}>◉</span>
      <span style={{ color: "#444" }}>DERNIER :</span>
      <span style={{ color: cfg.pillText }}>{latest.severity}</span>
      <span style={{ color: "#555" }}>•</span>
      <span style={{ color: "#888" }}>{latest.src_ip} → {latest.dst_ip}</span>
      <span style={{ color: "#555" }}>•</span>
      <span style={{ color: "#aaa" }}>{latest.attack_type || "Anomalie"}</span>
      <span style={{ color: "#555", marginLeft: "auto" }}>{latest.ts_human}</span>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  FILE ANALYSIS COMPONENTS                                       */
/* ═══════════════════════════════════════════════════════════════ */

function ConfidenceBar({ value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, background: "#111", borderRadius: 2, height: 6, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.4s ease" }} />
      </div>
      <span style={{ fontSize: 11, color, fontWeight: 600, width: 36, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function FileAnalysisRow({ row, idx }) {
  const cfg = SEV[row.severity] || SEV.NORMAL;
  const isAttack = row.is_attack;
  const isAnomaly = row.is_anomaly;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "28px 40px 1fr 110px 90px 90px 80px",
      gap: 8,
      padding: "8px 14px",
      borderBottom: "0.5px solid #151515",
      fontSize: 11,
      alignItems: "center",
      fontFamily: "'IBM Plex Mono', monospace",
      background: idx % 2 === 0 ? "#090909" : "#0b0b0b",
      borderLeft: `2px solid ${cfg.border}`,
    }}>
      <span style={{ color: cfg.glyphColor, fontSize: 9 }}>{cfg.glyph}</span>
      <span style={{ color: "#444", fontSize: 10 }}>#{row.row_index ?? idx}</span>
      <div>
        <div style={{ color: "#888", fontSize: 10, marginBottom: 2 }}>
          {row.src_ip && <span>{row.src_ip} → {row.dst_ip}</span>}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {isAttack && (
            <span style={{ background: "#501313", color: "#F09595", borderRadius: 3, padding: "0 6px", fontSize: 9, fontWeight: 700 }}>
              XGB: {row.attack_type}
            </span>
          )}
          {!isAttack && (
            <span style={{ background: "#1a2a1a", color: "#5DCAA5", borderRadius: 3, padding: "0 6px", fontSize: 9 }}>
              XGB: Normal
            </span>
          )}
          {isAnomaly && (
            <span style={{ background: "#412402", color: "#FAC775", borderRadius: 3, padding: "0 6px", fontSize: 9, fontWeight: 700 }}>
              KitNET: Zero-Day
            </span>
          )}
          {!isAnomaly && row.rmse !== undefined && (
            <span style={{ background: "#111", color: "#555", borderRadius: 3, padding: "0 6px", fontSize: 9 }}>
              KitNET: OK
            </span>
          )}
        </div>
      </div>
      <span style={{ background: cfg.pill, color: cfg.pillText, borderRadius: 4, padding: "1px 8px", fontSize: 9, fontWeight: 700, textAlign: "center", fontFamily: "sans-serif" }}>{row.severity}</span>
      <div>
        <div style={{ fontSize: 9, color: "#444", marginBottom: 2 }}>XGBoost</div>
        <ConfidenceBar value={row.confidence ?? 0} color="#378ADD" />
      </div>
      <div>
        <div style={{ fontSize: 9, color: "#444", marginBottom: 2 }}>KitNET RMSE</div>
        <div style={{ fontSize: 11, color: isAnomaly ? "#EF9F27" : "#5DCAA5", fontWeight: 600 }}>
          {row.rmse?.toFixed(4) ?? "—"}
        </div>
      </div>
      <span style={{ color: isAttack || isAnomaly ? "#F09595" : "#5DCAA5", textAlign: "center", fontSize: 10 }}>
        {isAttack || isAnomaly ? "⚠ Alerte" : "✓ Normal"}
      </span>
    </div>
  );
}

function SummaryStats({ results }) {
  if (!results?.length) return null;
  const total    = results.length;
  const attacks  = results.filter(r => r.is_attack).length;
  const anomalies= results.filter(r => r.is_anomaly && !r.is_attack).length;
  const normals  = total - attacks - anomalies;
  const byType   = {};
  results.filter(r => r.is_attack && r.attack_type).forEach(r => {
    byType[r.attack_type] = (byType[r.attack_type] || 0) + 1;
  });

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
      <div style={{ background: "#0d0d0d", border: "0.5px solid #1f1f1f", borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6 }}>Total lignes</div>
        <div style={{ fontSize: 32, fontWeight: 700, color: "#85B7EB" }}>{total}</div>
      </div>
      <div style={{ background: "#0d0d0d", border: "0.5px solid #1f1f1f", borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6 }}>Attaques XGBoost</div>
        <div style={{ fontSize: 32, fontWeight: 700, color: "#F09595" }}>{attacks}</div>
        <div style={{ fontSize: 10, color: "#555", marginTop: 4 }}>{total ? Math.round(attacks/total*100) : 0}%</div>
      </div>
      <div style={{ background: "#0d0d0d", border: "0.5px solid #1f1f1f", borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6 }}>Zero-Day KitNET</div>
        <div style={{ fontSize: 32, fontWeight: 700, color: "#EF9F27" }}>{anomalies}</div>
        <div style={{ fontSize: 10, color: "#555", marginTop: 4 }}>{total ? Math.round(anomalies/total*100) : 0}%</div>
      </div>
      <div style={{ background: "#0d0d0d", border: "0.5px solid #1f1f1f", borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6 }}>Normal</div>
        <div style={{ fontSize: 32, fontWeight: 700, color: "#5DCAA5" }}>{normals}</div>
        <div style={{ fontSize: 10, color: "#555", marginTop: 4 }}>{total ? Math.round(normals/total*100) : 0}%</div>
      </div>
    </div>
  );
}

function AccuracyPanel({ results, hasLabels }) {
  if (!hasLabels || !results?.length) return null;
  const withLabel = results.filter(r => r.true_label !== undefined && r.true_label !== null);
  if (!withLabel.length) return null;

  let tp = 0, tn = 0, fp = 0, fn = 0;
  withLabel.forEach(r => {
    const predicted = r.is_attack || r.is_anomaly;
    const actual    = r.true_label === 1 || r.true_label === "1" || r.true_label?.toLowerCase?.() !== "normal";
    if (predicted && actual)   tp++;
    if (!predicted && !actual) tn++;
    if (predicted && !actual)  fp++;
    if (!predicted && actual)  fn++;
  });
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1        = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  const accuracy  = (tp + tn) / withLabel.length;

  return (
    <div style={{ background: "#0d0d0d", border: "0.5px solid #1a2a1a", borderRadius: 10, padding: 16, marginBottom: 16 }}>
      <div style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 14 }}>
        📊 Métriques de Performance (labels détectés)
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {[
          { label: "Accuracy",  value: accuracy,  color: "#5DCAA5" },
          { label: "Précision", value: precision, color: "#85B7EB" },
          { label: "Rappel",    value: recall,    color: "#EF9F27" },
          { label: "F1-Score",  value: f1,        color: "#B5D4F4" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "#555", marginBottom: 8 }}>{label}</div>
            <svg width={80} height={80} viewBox="0 0 80 80" style={{ display: "block", margin: "0 auto" }}>
              <circle cx={40} cy={40} r={30} fill="none" stroke="#1a1a1a" strokeWidth={8} />
              <circle cx={40} cy={40} r={30} fill="none" stroke={color} strokeWidth={8}
                strokeDasharray={`${value * 188.5} 188.5`} strokeLinecap="round"
                transform="rotate(-90 40 40)" style={{ transition: "stroke-dasharray 0.8s ease" }} />
              <text x={40} y={46} textAnchor="middle" fill={color} fontSize={14} fontWeight={700} fontFamily="IBM Plex Mono">
                {Math.round(value * 100)}%
              </text>
            </svg>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 10, color: "#444" }}>
        <span>TP: <span style={{ color: "#5DCAA5" }}>{tp}</span></span>
        <span>TN: <span style={{ color: "#85B7EB" }}>{tn}</span></span>
        <span>FP: <span style={{ color: "#EF9F27" }}>{fp}</span></span>
        <span>FN: <span style={{ color: "#F09595" }}>{fn}</span></span>
        <span style={{ marginLeft: "auto" }}>Sur {withLabel.length} échantillons labellisés</span>
      </div>
    </div>
  );
}

function RmseScatterPlot({ results }) {
  if (!results?.length) return null;
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const pad = 32;
    const rmses = results.map(r => r.rmse ?? 0).filter(v => isFinite(v));
    if (!rmses.length) return;
    const maxR = Math.max(...rmses, 0.001) * 1.1;
    const thr  = results[0]?.threshold ?? 0;

    // Grid
    ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad + (H - pad * 2) * (i / 4);
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
      ctx.fillStyle = "#333"; ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText((maxR * (1 - i/4)).toFixed(3), pad - 4, y + 3);
    }

    // Threshold line
    if (thr > 0) {
      const ty = pad + (H - pad * 2) * (1 - thr / maxR);
      ctx.strokeStyle = "#854F0B"; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad, ty); ctx.lineTo(W - pad, ty); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#854F0B"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`seuil ${thr.toFixed(3)}`, pad + 4, ty - 3);
    }

    // Points
    results.forEach((r, i) => {
      const x = pad + (i / (results.length - 1 || 1)) * (W - pad * 2);
      const y = pad + (H - pad * 2) * (1 - (r.rmse ?? 0) / maxR);
      const isAlert = r.is_attack || r.is_anomaly;
      ctx.beginPath();
      ctx.arc(x, y, isAlert ? 3 : 2, 0, Math.PI * 2);
      ctx.fillStyle = r.is_attack ? "#E24B4A" : r.is_anomaly ? "#EF9F27" : "#378ADD55";
      ctx.fill();
    });
  }, [results]);
  return (
    <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16, marginBottom: 16 }}>
      <div style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>
        Score RMSE KitNET par échantillon
        <span style={{ marginLeft: 20, color: "#E24B4A" }}>● Attaque XGB</span>
        <span style={{ marginLeft: 12, color: "#EF9F27" }}>● Zero-Day</span>
        <span style={{ marginLeft: 12, color: "#378ADD55" }}>● Normal</span>
      </div>
      <canvas ref={ref} width={900} height={160} style={{ width: "100%", height: 160, display: "block" }} />
    </div>
  );
}

/* ─── FILE UPLOAD PANEL ────────────────────────────────────────── */
function FileAnalysisPanel() {
  const [dragOver, setDragOver]     = useState(false);
  const [file, setFile]             = useState(null);
  const [loading, setLoading]       = useState(false);
  const [progress, setProgress]     = useState(0);
  const [results, setResults]       = useState(null);
  const [error, setError]           = useState(null);
  const [filterSev, setFilterSev]   = useState("ALL");
  const [hasLabels, setHasLabels]   = useState(false);
  const fileRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f); setResults(null); setError(null);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setProgress(0); setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(p => Math.min(p + Math.random() * 15, 90));
      }, 300);

      const res = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: formData });
      clearInterval(progressInterval);
      setProgress(100);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Erreur serveur");
      }
      const data = await res.json();
      setResults(data.results);
      setHasLabels(data.has_labels);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const filtered = results
    ? (filterSev === "ALL" ? results : results.filter(r => {
        if (filterSev === "ATTACKS") return r.is_attack;
        if (filterSev === "ANOMALY") return r.is_anomaly && !r.is_attack;
        if (filterSev === "NORMAL")  return !r.is_attack && !r.is_anomaly;
        return r.severity === filterSev;
      }))
    : [];

  return (
    <div>
      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => fileRef.current?.click()}
        style={{
          border: `1.5px dashed ${dragOver ? "#378ADD" : file ? "#1D9E75" : "#2a2a2a"}`,
          borderRadius: 12,
          padding: "32px 20px",
          textAlign: "center",
          cursor: "pointer",
          background: dragOver ? "#050d14" : file ? "#050e08" : "#090909",
          marginBottom: 16,
          transition: "all 0.2s ease",
        }}
      >
        <input ref={fileRef} type="file" accept=".csv,.pcap,.pcapng" style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])} />
        <div style={{ fontSize: 28, marginBottom: 8 }}>{file ? "✓" : "⊕"}</div>
        {file ? (
          <>
            <div style={{ fontSize: 13, color: "#5DCAA5", fontWeight: 600 }}>{file.name}</div>
            <div style={{ fontSize: 11, color: "#444", marginTop: 4 }}>{(file.size / 1024).toFixed(1)} KB</div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 13, color: "#555" }}>Glissez un fichier CSV ici</div>
            <div style={{ fontSize: 10, color: "#333", marginTop: 6 }}>Format UNSW-NB15 ou features extraites (.csv)</div>
          </>
        )}
      </div>

      {/* Instructions */}
      {!results && !loading && (
        <div style={{ background: "#0a0d0a", border: "0.5px solid #1a2a1a", borderRadius: 10, padding: "12px 16px", marginBottom: 16, fontSize: 11, color: "#555" }}>
          <div style={{ color: "#3a6a3a", marginBottom: 8, fontWeight: 600 }}>Format CSV attendu :</div>
          <div>Colonnes UNSW-NB15 : <span style={{ color: "#444" }}>dur, sbytes, dbytes, sttl, dttl, Spkts, Dpkts, Sload, Dload, ...</span></div>
          <div style={{ marginTop: 6 }}>Colonnes optionnelles : <span style={{ color: "#444" }}>label (0/1 ou Normal/Attack), attack_cat, srcip, dstip</span></div>
          <div style={{ marginTop: 6, color: "#2a4a2a" }}>Si une colonne <span style={{ color: "#5a8a5a" }}>label</span> est présente, les métriques de performance seront calculées automatiquement.</div>
        </div>
      )}

      {/* Action button */}
      {file && !loading && (
        <button onClick={analyze} style={{
          display: "block", width: "100%",
          background: "#0a1a2a", border: "0.5px solid #185FA5",
          color: "#85B7EB", borderRadius: 8, padding: "12px",
          cursor: "pointer", fontSize: 12, fontFamily: "inherit",
          letterSpacing: "0.1em", fontWeight: 600,
          transition: "all 0.2s", marginBottom: 16,
        }}>
          ▶ ANALYSER LE FICHIER
        </button>
      )}

      {/* Progress */}
      {loading && (
        <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 20, marginBottom: 16, textAlign: "center" }}>
          <div style={{ fontSize: 12, color: "#555", marginBottom: 12 }}>Analyse en cours avec XGBoost + KitNET…</div>
          <div style={{ background: "#1a1a1a", borderRadius: 2, height: 4, overflow: "hidden", marginBottom: 8 }}>
            <div style={{ width: `${progress}%`, height: "100%", background: "#378ADD", transition: "width 0.3s ease", boxShadow: "0 0 8px #378ADD66" }} />
          </div>
          <div style={{ fontSize: 10, color: "#333" }}>{Math.round(progress)}%</div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ background: "#180808", border: "0.5px solid #A32D2D", borderRadius: 10, padding: "12px 16px", marginBottom: 16, fontSize: 11, color: "#F09595" }}>
          ✗ {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <>
          <SummaryStats results={results} />
          <AccuracyPanel results={results} hasLabels={hasLabels} />
          <RmseScatterPlot results={results} />

          {/* Attack type breakdown */}
          {results.some(r => r.is_attack) && (
            <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 14 }}>
                Répartition des attaques détectées
              </div>
              <DonutChart counts={results.filter(r => r.is_attack && r.attack_type).reduce((acc, r) => {
                acc[r.attack_type] = (acc[r.attack_type] || 0) + 1;
                return acc;
              }, {})} />
            </div>
          )}

          {/* Filter + Table */}
          <div style={{ marginBottom: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 10, color: "#444" }}>FILTRER :</span>
            {["ALL", "ATTACKS", "ANOMALY", "NORMAL"].map(f => (
              <button key={f} onClick={() => setFilterSev(f)} style={{
                background: filterSev === f ? "#0a1a2a" : "none",
                border: filterSev === f ? "0.5px solid #185FA5" : "0.5px solid #222",
                color: filterSev === f ? "#85B7EB" : "#444",
                borderRadius: 4, padding: "4px 12px", cursor: "pointer",
                fontSize: 10, fontFamily: "inherit",
              }}>{f}</button>
            ))}
            <span style={{ marginLeft: "auto", fontSize: 10, color: "#333" }}>
              {filtered.length} / {results.length} lignes
            </span>
          </div>

          <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "28px 40px 1fr 110px 90px 90px 80px", gap: 8, padding: "6px 14px", borderBottom: "0.5px solid #1a1a1a", fontSize: 9, color: "#333", textTransform: "uppercase", letterSpacing: "0.14em", background: "#0d0d0d" }}>
              <span /><span>#</span><span>Flux / Détection</span><span>Sévérité</span><span>XGBoost</span><span>KitNET</span><span>Résultat</span>
            </div>
            <div style={{ maxHeight: 420, overflowY: "auto" }}>
              {filtered.length
                ? filtered.slice(0, 500).map((r, i) => <FileAnalysisRow key={i} row={r} idx={i} />)
                : (
                  <div style={{ padding: "32px 0", textAlign: "center", color: "#2a2a2a", fontSize: 12 }}>
                    <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.3 }}>◎</div>
                    Aucun résultat pour ce filtre
                  </div>
                )}
            </div>
          </div>

          {filtered.length > 500 && (
            <div style={{ padding: "8px 14px", fontSize: 10, color: "#444", textAlign: "center" }}>
              Affichage limité aux 500 premières lignes — {results.length} total
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  MAIN APP                                                        */
/* ═══════════════════════════════════════════════════════════════ */

const NAV_TABS = [
  { id: "live",   label: "Live Capture", icon: "◉" },
  { id: "file",   label: "Analyse Fichier", icon: "⊞" },
];

const LIVE_TABS = [
  { label: "Alertes",       icon: "⚠" },
  { label: "RMSE / KitNET",icon: "~" },
  { label: "Répartition",  icon: "◕" },
];

export default function App() {
  const [connected, setConnected]   = useState(false);
  const [data, setData]             = useState(null);
  const [alerts, setAlerts]         = useState([]);
  const [rmseSeries, setRmseSeries] = useState([]);
  const [ppsSeries, setPpsSeries]   = useState([]);
  const [navTab, setNavTab]         = useState("live");
  const [liveTab, setLiveTab]       = useState(0);
  const [clock, setClock]           = useState(new Date());
  const wsRef  = useRef(null);
  const retryRef = useRef(null);

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen  = () => { setConnected(true); clearTimeout(retryRef.current); };
    ws.onclose = () => { setConnected(false); retryRef.current = setTimeout(connect, 3000); };
    ws.onerror = () => ws.close();
    ws.onmessage = e => {
      try {
        const p = JSON.parse(e.data);
        setData(p);
        setAlerts(p.alerts || []);
        if (p.rmse_series?.length) setRmseSeries(prev => [...prev, ...p.rmse_series.slice(-5)].slice(-400));
        if (p.pps_series?.length)  setPpsSeries(prev => [...prev, ...p.pps_series.slice(-5)].slice(-400));
      } catch (_) {}
    };
  }, []);

  useEffect(() => { connect(); return () => wsRef.current?.close(); }, [connect]);

  const stats    = data?.stats         || {};
  const kitnet   = data?.kitnet        || {};
  const capStats = data?.capture_stats || {};
  const counts   = data?.attack_counts || {};
  const pps      = ppsSeries.at(-1)?.pps  ?? 0;
  const rmse     = rmseSeries.at(-1)?.rmse ?? 0;
  const critAlerts = alerts.filter(a => a.severity === "CRITICAL").length;

  return (
    <div style={{ background: "#050505", minHeight: "100vh", color: "#ccc", fontFamily: "'IBM Plex Mono', 'Courier New', monospace", fontSize: 12 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @keyframes blink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} 99%{opacity:0} }
        .nav-tab:hover { color: #85B7EB !important; border-color: #222 !important; }
        .tab-btn:hover { color: #85B7EB !important; }
      `}</style>

      {/* ── HEADER ────────────────────────────────────────────────── */}
      <header style={{ borderBottom: "0.5px solid #1a1a1a", padding: "10px 20px", display: "flex", alignItems: "center", gap: 14, position: "sticky", top: 0, background: "#050505", zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.18em", color: "#e8e8e8" }}>NIDS</span>
          <span style={{ fontSize: 9, color: "#444", letterSpacing: "0.14em" }}>NETWORK INTRUSION DETECTION</span>
        </div>

        {critAlerts > 0 && (
          <div style={{ background: "#501313", color: "#F09595", padding: "2px 10px", borderRadius: 4, fontSize: 10, fontWeight: 700, animation: "blink 1s step-start infinite", letterSpacing: "0.1em" }}>
            ⚠ {critAlerts} CRITICAL
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Nav tabs */}
        <div style={{ display: "flex", gap: 4 }}>
          {NAV_TABS.map(t => (
            <button key={t.id} className="nav-tab" onClick={() => setNavTab(t.id)} style={{
              background: navTab === t.id ? "#0a1620" : "none",
              border: navTab === t.id ? "0.5px solid #185FA5" : "0.5px solid transparent",
              color: navTab === t.id ? "#85B7EB" : "#444",
              borderRadius: 6, padding: "5px 14px", cursor: "pointer",
              fontSize: 11, fontFamily: "inherit", letterSpacing: "0.06em",
              transition: "all 0.15s",
            }}>
              <span style={{ marginRight: 6, opacity: 0.7 }}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: connected ? "#1D9E75" : "#A32D2D", display: "inline-block", animation: connected ? "pulse 2s ease-in-out infinite" : "none" }} />
          <span style={{ fontSize: 10, color: connected ? "#5DCAA5" : "#E24B4A", letterSpacing: "0.08em" }}>{connected ? "LIVE" : "OFFLINE"}</span>
        </div>
        <span style={{ fontSize: 10, color: "#333" }}>{clock.toLocaleTimeString()}</span>
      </header>

      {/* ── LIVE MODE ─────────────────────────────────────────────── */}
      {navTab === "live" && (
        <>
          <Ticker alerts={alerts} />
          <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Stat cards */}
            <div style={{ display: "flex", gap: 10 }}>
              <Card label="Flux analysés"    value={(stats.total_flows||0).toLocaleString()} sub={`${(capStats.total_pkts||0).toLocaleString()} paquets`} accent="#85B7EB" icon="⇆" />
              <Card label="Alertes totales"  value={(stats.total_alerts||0).toLocaleString()} accent="#F09595" icon="⚠" />
              <Card label="Attaques connues" value={(stats.attacks||0).toLocaleString()} accent="#EF9F27" icon="◈" />
              <Card label="Anomalies 0-Day"  value={(stats.anomalies||0).toLocaleString()} accent="#5DCAA5" icon="∿" />
              <Card label="RMSE actuel"      value={rmse.toFixed(4)} sub={kitnet.trained ? `seuil: ${(kitnet.threshold||0).toFixed(4)}` : "apprentissage…"} accent={kitnet.trained && rmse > (kitnet.threshold||0) ? "#F09595" : "#5DCAA5"} icon="⊕" />
              <Card label="Confiance moy."   value={alerts.length ? `${Math.round(alerts.slice(0,20).reduce((s,a)=>s+a.confidence,0)/Math.min(alerts.length,20)*100)}%` : "—"} accent="#B5D4F4" icon="%" />
            </div>

            {/* KitNET bar */}
            <div style={{ background: "#0d0d0d", border: "0.5px solid #1a1a1a", borderRadius: 10, padding: "12px 16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em" }}>KitNET Engine</span>
                  <span style={{ fontSize: 10, color: "#333" }}>{(kitnet.packet_count||0).toLocaleString()} paquets</span>
                  <span style={{ fontSize: 10, color: "#333" }}>•</span>
                  <span style={{ fontSize: 10, color: "#333" }}>{kitnet.n_features||0} features</span>
                  <span style={{ fontSize: 10, color: "#333" }}>•</span>
                  <span style={{ fontSize: 10, color: "#333" }}>seuil {(kitnet.threshold||0).toFixed(4)}</span>
                </div>
                <span style={{ fontSize: 10, fontWeight: 700, color: kitnet.trained ? "#1D9E75" : "#EF9F27", letterSpacing: "0.12em" }}>
                  {kitnet.trained ? "● ACTIF" : "○ APPRENTISSAGE"}
                </span>
              </div>
              <TrainingBar progress={kitnet.progress||0} trained={kitnet.trained||false} />
            </div>

            {/* Sparklines */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ background: "#0d0d0d", border: "0.5px solid #1a1a1a", borderRadius: 10, padding: "10px 14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em" }}>Paquets / seconde</span>
                  <span style={{ fontSize: 12, color: "#378ADD", fontWeight: 600 }}>{pps}</span>
                </div>
                <Spark data={ppsSeries} valueKey="pps" color="#378ADD" h={44} />
              </div>
              <div style={{ background: "#0d0d0d", border: "0.5px solid #1a1a1a", borderRadius: 10, padding: "10px 14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em" }}>Score d'anomalie RMSE</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: kitnet.trained && rmse > kitnet.threshold ? "#F09595" : "#5DCAA5" }}>{rmse.toFixed(5)}</span>
                </div>
                <Spark data={rmseSeries} valueKey="rmse" color={kitnet.trained && rmse > (kitnet.threshold||0) ? "#E24B4A" : "#1D9E75"} h={44} />
              </div>
            </div>

            {/* Live tabs */}
            <div style={{ borderBottom: "0.5px solid #1a1a1a", display: "flex", gap: 0 }}>
              {LIVE_TABS.map((t, i) => (
                <button key={t.label} className="tab-btn" onClick={() => setLiveTab(i)} style={{
                  background: "none", border: "none",
                  borderBottom: liveTab === i ? "2px solid #378ADD" : "2px solid transparent",
                  color: liveTab === i ? "#85B7EB" : "#444",
                  padding: "8px 20px", cursor: "pointer", fontSize: 11,
                  fontFamily: "inherit", letterSpacing: "0.08em", transition: "color 0.15s",
                }}>
                  <span style={{ marginRight: 6, opacity: 0.6 }}>{t.icon}</span>{t.label}
                </button>
              ))}
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 10, color: "#2a2a2a", alignSelf: "center", paddingRight: 14 }}>{alerts.length} alertes</span>
            </div>

            {liveTab === 0 && (
              <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, overflow: "hidden" }}>
                <div style={{ display: "grid", gridTemplateColumns: "16px 62px 96px 1fr 100px 72px 62px", gap: 10, padding: "6px 14px", borderBottom: "0.5px solid #1a1a1a", fontSize: 9, color: "#333", textTransform: "uppercase", letterSpacing: "0.14em", background: "#0d0d0d" }}>
                  <span /><span>Heure</span><span>Sévérité</span><span>Flux src → dst</span><span>Type</span><span>Proto</span><span>Conf.</span>
                </div>
                <div style={{ maxHeight: 380, overflowY: "auto" }}>
                  {alerts.length
                    ? alerts.map((a, i) => <AlertRow key={i} alert={a} idx={i} />)
                    : <div style={{ padding: "32px 0", textAlign: "center", color: "#2a2a2a", fontSize: 12 }}><div style={{ fontSize: 24, marginBottom: 8, opacity: 0.3 }}>◎</div>Aucune alerte — trafic normal</div>}
                </div>
              </div>
            )}

            {liveTab === 1 && (
              <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <span style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em" }}>RMSE KitNET — 100 dernières valeurs</span>
                  <div style={{ display: "flex", gap: 16, fontSize: 10 }}>
                    <span style={{ color: "#378ADD" }}>─── RMSE</span>
                    <span style={{ color: "#854F0B" }}>- - seuil</span>
                    <span style={{ color: "#A32D2D" }}>░ anomalie</span>
                  </div>
                </div>
                <RmseTimeline data={rmseSeries} threshold={kitnet.threshold} />
                <div style={{ marginTop: 12, display: "flex", gap: 20, fontSize: 10, color: "#444" }}>
                  <span>Traité : <span style={{ color: "#888" }}>{(kitnet.packet_count||0).toLocaleString()}</span></span>
                  <span>Phase : <span style={{ color: "#888" }}>{kitnet.trained ? "Surveillance" : "Apprentissage"}</span></span>
                  <span>Seuil : <span style={{ color: "#EF9F27" }}>{(kitnet.threshold||0).toFixed(5)}</span></span>
                </div>
              </div>
            )}

            {liveTab === 2 && (
              <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 16 }}>Répartition des types d'attaque</div>
                <DonutChart counts={counts} />
              </div>
            )}

            <div style={{ paddingTop: 8, borderTop: "0.5px solid #111", display: "flex", gap: 20, fontSize: 10, color: "#2a2a2a" }}>
              <span>Interface : {capStats.interface || "eth0"}</span>
              <span style={{ flex: 1 }} />
              <span>NIDS Dashboard v2.0 · Pipeline A: KitNET + AfterImage · Pipeline B: XGBoost UNSW-NB15</span>
            </div>
          </div>
        </>
      )}

      {/* ── FILE ANALYSIS MODE ─────────────────────────────────────── */}
      {navTab === "file" && (
        <div style={{ padding: "20px" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#e8e8e8", letterSpacing: "0.1em", marginBottom: 6 }}>
              Analyse de Fichier — Mode Démonstration
            </div>
            <div style={{ fontSize: 11, color: "#555" }}>
              Uploadez un fichier CSV UNSW-NB15 pour analyser les flux avec les deux pipelines (XGBoost supervisé + KitNET non-supervisé).
              Idéal pour valider les modèles avec des données de test connues.
            </div>
          </div>

          {/* Pipeline legend */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
            <div style={{ background: "#080d12", border: "0.5px solid #185FA5", borderRadius: 10, padding: "14px 16px" }}>
              <div style={{ fontSize: 10, color: "#378ADD", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Pipeline A — Supervisé</div>
              <div style={{ fontSize: 11, color: "#666", lineHeight: 1.6 }}>
                <div>• XGBoost + Random Forest (UNSW-NB15)</div>
                <div>• Détection binaire attaque / normal</div>
                <div>• Classification multi-classe (type d'attaque)</div>
                <div>• Confiance par probabilité de classe</div>
              </div>
            </div>
            <div style={{ background: "#0d0a06", border: "0.5px solid #854F0B", borderRadius: 10, padding: "14px 16px" }}>
              <div style={{ fontSize: 10, color: "#EF9F27", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Pipeline B — Non-Supervisé</div>
              <div style={{ fontSize: 11, color: "#666", lineHeight: 1.6 }}>
                <div>• KitNET (autoencodeur en ligne)</div>
                <div>• Score RMSE par flux (reconstruction)</div>
                <div>• Détection d'anomalies Zero-Day</div>
                <div>• Seuil calibré automatiquement</div>
              </div>
            </div>
          </div>

          <FileAnalysisPanel />
        </div>
      )}
    </div>
  );
}