import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";

/* ─── Severity palette ─────────────────────────────────────────── */
const SEV = {
  CRITICAL: { pill: "#501313", pillText: "#F09595", row: "#1a0808", border: "#A32D2D", glyph: "◈", glyphColor: "#E24B4A" },
  HIGH:     { pill: "#412402", pillText: "#FAC775", row: "#180e00", border: "#854F0B", glyph: "▲", glyphColor: "#EF9F27" },
  MEDIUM:   { pill: "#04342C", pillText: "#5DCAA5", row: "#021a14", border: "#0F6E56", glyph: "◆", glyphColor: "#1D9E75" },
  LOW:      { pill: "#042C53", pillText: "#85B7EB", row: "#02111f", border: "#185FA5", glyph: "●", glyphColor: "#378ADD" },
  NORMAL:   { pill: "#2C2C2A", pillText: "#B4B2A9", row: "#111", border: "#444441", glyph: "○", glyphColor: "#5F5E5A" },
};

/* ─── Sparkline hook ───────────────────────────────────────────── */
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
    const pts = vals.map((v, i) => [
      (i / Math.max(vals.length - 1, 1)) * w,
      h - (v / max) * h * 0.85 - 2,
    ]);
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = color + "18";
    ctx.fill();
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
      ctx.beginPath();
      ctx.moveTo(80, 80);
      ctx.arc(80, 80, 68, angle, angle + slice);
      ctx.closePath();
      ctx.fillStyle = DONUT_COLORS[i % DONUT_COLORS.length];
      ctx.fill();
      ctx.strokeStyle = "#0a0a0a"; ctx.lineWidth = 2; ctx.stroke();
      angle += slice;
    });
    ctx.beginPath();
    ctx.arc(80, 80, 42, 0, Math.PI * 2);
    ctx.fillStyle = "#0a0a0a"; ctx.fill();
    ctx.fillStyle = "#ccc"; ctx.font = "bold 20px 'IBM Plex Mono'"; ctx.textAlign = "center";
    ctx.fillText(total, 80, 86);
    ctx.fillStyle = "#555"; ctx.font = "10px sans-serif";
    ctx.fillText("total", 80, 100);
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

    // Grid
    ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 0.5;
    [0.25, 0.5, 0.75].forEach(f => {
      const y = h - f * h * 0.88;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    });

    // Threshold band
    if (threshold) {
      const ty = h - (threshold / maxV) * h * 0.88;
      ctx.fillStyle = "#854f0b11";
      ctx.fillRect(0, 0, w, ty);
      ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(w, ty);
      ctx.strokeStyle = "#854F0B"; ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Anomaly markers
    pts.forEach((d, i) => {
      if (d.rmse > threshold && threshold) {
        const x = (i / (pts.length - 1)) * w;
        ctx.fillStyle = "#A32D2D22";
        ctx.fillRect(x - 2, 0, 4, h);
      }
    });

    // RMSE line
    ctx.beginPath();
    pts.forEach((d, i) => {
      const x = (i / Math.max(pts.length - 1, 1)) * w;
      const y = h - (d.rmse / maxV) * h * 0.88;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#378ADD"; ctx.lineWidth = 2; ctx.stroke();

    // Area fill
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = "#378ADD14"; ctx.fill();
  }, [data, threshold]);

  return (
    <canvas ref={ref} width={800} height={130}
      style={{ width: "100%", height: 130, display: "block" }} />
  );
}

/* ─── KitNET training bar ──────────────────────────────────────── */
function TrainingBar({ progress, trained }) {
  const pct = Math.min(Math.round((progress || 0) * 100), 100);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11 }}>
        <span style={{ color: "#666" }}>
          {trained ? "Surveillance active — modèle entraîné" : `Phase d'apprentissage… ${pct}%`}
        </span>
        <span style={{ color: trained ? "#1D9E75" : "#EF9F27", fontWeight: 600 }}>
          {trained ? "ACTIF" : `${pct}%`}
        </span>
      </div>
      <div style={{ background: "#1a1a1a", borderRadius: 2, height: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: trained ? "#1D9E75" : "#EF9F27",
          transition: "width 0.5s ease",
          boxShadow: trained ? "0 0 6px #1D9E7588" : "0 0 6px #EF9F2788"
        }} />
      </div>
    </div>
  );
}

/* ─── Stat card ────────────────────────────────────────────────── */
function Card({ label, value, sub, accent = "#85B7EB", icon }) {
  return (
    <div style={{
      background: "#0d0d0d",
      border: "0.5px solid #1f1f1f",
      borderRadius: 10,
      padding: "14px 16px",
      flex: "1 1 0",
      minWidth: 0,
      position: "relative",
      overflow: "hidden",
    }}>
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
    <div style={{
      display: "grid",
      gridTemplateColumns: "16px 62px 96px 1fr 100px 72px 62px",
      gap: 10,
      padding: "7px 14px",
      borderBottom: "0.5px solid #151515",
      fontSize: 11,
      alignItems: "center",
      fontFamily: "'IBM Plex Mono', monospace",
      background: idx % 2 === 0 ? "#0a0a0a" : "#0c0c0c",
      borderLeft: `2px solid ${cfg.border}`,
      transition: "background 0.15s",
    }}>
      <span style={{ color: cfg.glyphColor, fontSize: 9 }}>{cfg.glyph}</span>
      <span style={{ color: "#444", fontSize: 10 }}>{alert.ts_human}</span>
      <span style={{
        background: cfg.pill, color: cfg.pillText,
        borderRadius: 4, padding: "1px 7px",
        fontSize: 9, fontWeight: 700,
        textAlign: "center", fontFamily: "sans-serif",
        letterSpacing: "0.08em"
      }}>{alert.severity}</span>
      <span style={{ color: "#bbb", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        <span style={{ color: "#666" }}>{alert.src_ip}</span>
        <span style={{ color: "#333", margin: "0 4px" }}>→</span>
        <span style={{ color: "#777" }}>{alert.dst_ip}</span>
        {alert.dport ? <span style={{ color: "#444" }}>:{alert.dport}</span> : null}
      </span>
      <span style={{ color: alert.is_attack ? "#F09595" : "#5DCAA5", fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {alert.attack_type || (alert.is_attack ? "Unknown" : "Anomalie")}
      </span>
      <span style={{ color: "#555", textAlign: "center" }}>{alert.proto}</span>
      <span style={{ color: "#888", textAlign: "right" }}>{(alert.confidence * 100).toFixed(0)}%</span>
    </div>
  );
}

/* ─── Live ticker ──────────────────────────────────────────────── */
function Ticker({ alerts }) {
  const latest = alerts.slice(0, 1)[0];
  if (!latest) return null;
  const cfg = SEV[latest.severity] || SEV.NORMAL;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "5px 14px",
      background: "#0e0e0e",
      borderBottom: "0.5px solid #181818",
      fontSize: 11, fontFamily: "monospace",
      overflow: "hidden",
    }}>
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

const TABS = [
  { label: "Alertes", icon: "⚠" },
  { label: "RMSE / KitNET", icon: "~" },
  { label: "Répartition", icon: "◕" },
];

/* ─── App ──────────────────────────────────────────────────────── */
export default function App() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [rmseSeries, setRmseSeries] = useState([]);
  const [ppsSeries, setPpsSeries] = useState([]);
  const [tab, setTab] = useState(0);
  const [clock, setClock] = useState(new Date());
  const wsRef = useRef(null);
  const retryRef = useRef(null);

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => { setConnected(true); clearTimeout(retryRef.current); };
    ws.onclose = () => {
      setConnected(false);
      retryRef.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = e => {
      try {
        const p = JSON.parse(e.data);
        setData(p);
        setAlerts(p.alerts || []);
        if (p.rmse_series?.length)
          setRmseSeries(prev => [...prev, ...p.rmse_series.slice(-5)].slice(-400));
        if (p.pps_series?.length)
          setPpsSeries(prev => [...prev, ...p.pps_series.slice(-5)].slice(-400));
      } catch (_) {}
    };
  }, []);

  useEffect(() => { connect(); return () => wsRef.current?.close(); }, [connect]);

  const stats   = data?.stats        || {};
  const kitnet  = data?.kitnet       || {};
  const capStats= data?.capture_stats|| {};
  const counts  = data?.attack_counts|| {};
  const pps     = ppsSeries.at(-1)?.pps  ?? 0;
  const rmse    = rmseSeries.at(-1)?.rmse ?? 0;
  const critAlerts = alerts.filter(a => a.severity === "CRITICAL").length;

  return (
    <div style={{
      background: "#050505",
      minHeight: "100vh",
      color: "#ccc",
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      fontSize: 12,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @keyframes blink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} 99%{opacity:0} }
        .tab-btn:hover { color: #85B7EB !important; }
      `}</style>

      {/* Header */}
      <header style={{
        borderBottom: "0.5px solid #1a1a1a",
        padding: "10px 20px",
        display: "flex", alignItems: "center", gap: 14,
        position: "sticky", top: 0,
        background: "#050505",
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.18em", color: "#e8e8e8" }}>NIDS</span>
          <span style={{ fontSize: 9, color: "#444", letterSpacing: "0.14em" }}>NETWORK INTRUSION DETECTION</span>
        </div>

        {critAlerts > 0 && (
          <div style={{
            background: "#501313", color: "#F09595",
            padding: "2px 10px", borderRadius: 4,
            fontSize: 10, fontWeight: 700,
            animation: "blink 1s step-start infinite",
            letterSpacing: "0.1em"
          }}>
            ⚠ {critAlerts} CRITICAL
          </div>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ display: "flex", gap: 16, alignItems: "center", fontSize: 11 }}>
          <span style={{ color: "#333" }}>PKT/s</span>
          <span style={{ color: "#85B7EB", fontWeight: 600 }}>{pps}</span>
          <span style={{ color: "#333" }}>FLOWS</span>
          <span style={{ color: "#85B7EB", fontWeight: 600 }}>{(stats.total_flows || 0).toLocaleString()}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: connected ? "#1D9E75" : "#A32D2D",
            display: "inline-block",
            animation: connected ? "pulse 2s ease-in-out infinite" : "none"
          }} />
          <span style={{ fontSize: 10, color: connected ? "#5DCAA5" : "#E24B4A", letterSpacing: "0.08em" }}>
            {connected ? "LIVE" : "OFFLINE"}
          </span>
        </div>

        <span style={{ fontSize: 10, color: "#333", letterSpacing: "0.05em" }}>
          {clock.toLocaleTimeString()}
        </span>
      </header>

      {/* Ticker */}
      <Ticker alerts={alerts} />

      <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Stat cards */}
        <div style={{ display: "flex", gap: 10 }}>
          <Card label="Flux analysés"    value={(stats.total_flows || 0).toLocaleString()}    sub={`${(capStats.total_pkts || 0).toLocaleString()} paquets`} accent="#85B7EB" icon="⇆" />
          <Card label="Alertes totales"  value={(stats.total_alerts || 0).toLocaleString()}   accent="#F09595" icon="⚠" />
          <Card label="Attaques connues" value={(stats.attacks || 0).toLocaleString()}         accent="#EF9F27" icon="◈" />
          <Card label="Anomalies 0-Day"  value={(stats.anomalies || 0).toLocaleString()}       accent="#5DCAA5" icon="∿" />
          <Card label="RMSE actuel"      value={rmse.toFixed(4)}
            sub={kitnet.trained ? `seuil: ${(kitnet.threshold || 0).toFixed(4)}` : "apprentissage…"}
            accent={kitnet.trained && rmse > (kitnet.threshold || 0) ? "#F09595" : "#5DCAA5"} icon="⊕" />
          <Card label="Confiance moy."
            value={alerts.length ? `${Math.round(alerts.slice(0,20).reduce((s,a)=>s+a.confidence,0)/Math.min(alerts.length,20)*100)}%` : "—"}
            accent="#B5D4F4" icon="%" />
        </div>

        {/* KitNET status bar */}
        <div style={{
          background: "#0d0d0d",
          border: "0.5px solid #1a1a1a",
          borderRadius: 10,
          padding: "12px 16px",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: "#555", textTransform: "uppercase", letterSpacing: "0.12em" }}>KitNET Engine</span>
              <span style={{ fontSize: 10, color: "#333" }}>{(kitnet.packet_count || 0).toLocaleString()} paquets</span>
              <span style={{ fontSize: 10, color: "#333" }}>•</span>
              <span style={{ fontSize: 10, color: "#333" }}>{kitnet.n_features || 0} features</span>
              <span style={{ fontSize: 10, color: "#333" }}>•</span>
              <span style={{ fontSize: 10, color: "#333" }}>seuil {(kitnet.threshold || 0).toFixed(4)}</span>
            </div>
            <span style={{
              fontSize: 10, fontWeight: 700,
              color: kitnet.trained ? "#1D9E75" : "#EF9F27",
              letterSpacing: "0.12em"
            }}>
              {kitnet.trained ? "● ACTIF" : "○ APPRENTISSAGE"}
            </span>
          </div>
          <TrainingBar progress={kitnet.progress || 0} trained={kitnet.trained || false} />
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
              <span style={{ fontSize: 12, fontWeight: 600, color: kitnet.trained && rmse > kitnet.threshold ? "#F09595" : "#5DCAA5" }}>
                {rmse.toFixed(5)}
              </span>
            </div>
            <Spark data={rmseSeries} valueKey="rmse"
              color={kitnet.trained && rmse > (kitnet.threshold||0) ? "#E24B4A" : "#1D9E75"} h={44} />
          </div>
        </div>

        {/* Tabs */}
        <div style={{ borderBottom: "0.5px solid #1a1a1a", display: "flex", gap: 0 }}>
          {TABS.map((t, i) => (
            <button key={t.label} className="tab-btn" onClick={() => setTab(i)} style={{
              background: "none", border: "none",
              borderBottom: tab === i ? "2px solid #378ADD" : "2px solid transparent",
              color: tab === i ? "#85B7EB" : "#444",
              padding: "8px 20px",
              cursor: "pointer", fontSize: 11,
              fontFamily: "inherit",
              letterSpacing: "0.08em",
              transition: "color 0.15s",
            }}>
              <span style={{ marginRight: 6, opacity: 0.6 }}>{t.icon}</span>{t.label}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 10, color: "#2a2a2a", alignSelf: "center", paddingRight: 14 }}>
            {alerts.length} alertes en mémoire
          </span>
        </div>

        {/* Tab: Alertes */}
        {tab === 0 && (
          <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, overflow: "hidden" }}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "16px 62px 96px 1fr 100px 72px 62px",
              gap: 10,
              padding: "6px 14px",
              borderBottom: "0.5px solid #1a1a1a",
              fontSize: 9, color: "#333",
              textTransform: "uppercase", letterSpacing: "0.14em",
              background: "#0d0d0d",
            }}>
              <span />
              <span>Heure</span><span>Sévérité</span>
              <span>Flux src → dst</span>
              <span>Type</span><span>Proto</span><span>Conf.</span>
            </div>
            <div style={{ maxHeight: 380, overflowY: "auto" }}>
              {alerts.length
                ? alerts.map((a, i) => <AlertRow key={i} alert={a} idx={i} />)
                : (
                  <div style={{ padding: "32px 0", textAlign: "center", color: "#2a2a2a", fontSize: 12 }}>
                    <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.3 }}>◎</div>
                    Aucune alerte — trafic normal
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Tab: RMSE */}
        {tab === 1 && (
          <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                RMSE KitNET — 100 dernières valeurs
              </span>
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
              <span>n_features : <span style={{ color: "#888" }}>{kitnet.n_features || "—"}</span></span>
            </div>
            {!kitnet.trained && (
              <div style={{
                marginTop: 12, fontSize: 11, color: "#555",
                background: "#0d0d0d", borderRadius: 6, padding: "8px 12px",
                borderLeft: "2px solid #854F0B",
              }}>
                KitNET est en phase d'apprentissage ({Math.round((kitnet.progress||0)*100)}% / 100%). La détection d'anomalies sera activée après la phase de grâce.
              </div>
            )}
          </div>
        )}

        {/* Tab: Répartition */}
        {tab === 2 && (
          <div style={{ background: "#090909", border: "0.5px solid #141414", borderRadius: 10, padding: 16 }}>
            <div style={{ fontSize: 10, color: "#444", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 16 }}>
              Répartition des types d'attaque
            </div>
            <DonutChart counts={counts} />
          </div>
        )}

        {/* Footer */}
        <div style={{ paddingTop: 8, borderTop: "0.5px solid #111", display: "flex", gap: 20, fontSize: 10, color: "#2a2a2a" }}>
          <span>Interface : {capStats.interface || "eth0"}</span>
          <span>Uptime : —</span>
          <span style={{ flex: 1 }} />
          <span>NIDS Dashboard v2.0 · Pipeline A: KitNET + AfterImage · Pipeline B: XGBoost UNSW-NB15</span>
        </div>
      </div>
    </div>
  );
}