import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, RadarChart,
  PolarGrid, PolarAngleAxis, Radar,
} from "recharts";

// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL   = import.meta.env.VITE_WS_URL  || "ws://localhost:8000/ws";

// ── Color palette ────────────────────────────────────────────────────────────
const C = {
  blue:   "#2563EB",
  teal:   "#0D9488",
  indigo: "#4F46E5",
  amber:  "#D97706",
  red:    "#DC2626",
  green:  "#16A34A",
  slate:  "#64748B",
  bg:     "#F8FAFC",
  card:   "#FFFFFF",
  border: "#E2E8F0",
  text:   "#0F172A",
  sub:    "#64748B",
};

const BANK_COLORS = ["#2563EB","#0D9488","#D97706","#7C3AED","#DC2626"];

// ── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");
const pct = (v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—");

async function apiPost(path, body) {
  const r = await fetch(`${API_BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function apiGet(path) {
  const r = await fetch(`${API_BASE}/api${path}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status }) {
  const map = {
    idle:      { bg: "#F1F5F9", color: C.slate,  dot: "#94A3B8", label: "Idle" },
    running:   { bg: "#DBEAFE", color: C.blue,   dot: C.blue,    label: "Training" },
    completed: { bg: "#DCFCE7", color: C.green,  dot: C.green,   label: "Completed" },
    error:     { bg: "#FEE2E2", color: C.red,    dot: C.red,     label: "Error" },
  };
  const s = map[status] || map.idle;
  return (
    <span style={{ display:"inline-flex", alignItems:"center", gap:6,
      background:s.bg, color:s.color, borderRadius:20,
      padding:"4px 12px", fontSize:12, fontWeight:600, letterSpacing:"0.03em" }}>
      <span style={{ width:7, height:7, borderRadius:"50%", background:s.dot,
        animation: status==="running" ? "pulse 1.4s infinite" : "none" }} />
      {s.label}
    </span>
  );
}

function KpiCard({ label, value, sub, accent, delta }) {
  return (
    <div style={{ background:C.card, border:`1px solid ${C.border}`,
      borderRadius:12, padding:"20px 24px", position:"relative", overflow:"hidden" }}>
      <div style={{ position:"absolute", top:0, left:0, width:4,
        height:"100%", background:accent || C.blue, borderRadius:"12px 0 0 12px" }} />
      <div style={{ marginLeft:8 }}>
        <div style={{ fontSize:12, color:C.sub, fontWeight:500,
          textTransform:"uppercase", letterSpacing:"0.06em", marginBottom:6 }}>{label}</div>
        <div style={{ fontSize:28, fontWeight:700, color:C.text, lineHeight:1 }}>{value}</div>
        {sub && <div style={{ fontSize:12, color:C.sub, marginTop:4 }}>{sub}</div>}
        {delta !== undefined && (
          <div style={{ fontSize:12, fontWeight:600, color: delta >= 0 ? C.green : C.red, marginTop:4 }}>
            {delta >= 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)}% vs baseline
          </div>
        )}
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize:11, fontWeight:700, color:C.sub, letterSpacing:"0.08em",
      textTransform:"uppercase", marginBottom:12, marginTop:8 }}>
      {children}
    </div>
  );
}

function ChartCard({ title, children, height = 240 }) {
  return (
    <div style={{ background:C.card, border:`1px solid ${C.border}`,
      borderRadius:12, padding:"20px 24px" }}>
      <div style={{ fontSize:13, fontWeight:600, color:C.text, marginBottom:16 }}>{title}</div>
      <div style={{ height }}>{children}</div>
    </div>
  );
}

// ── Control Panel ─────────────────────────────────────────────────────────────

function ControlPanel({ onStart, disabled }) {
  const [cfg, setCfg] = useState({
    num_rounds: 15, local_epochs: 5, strategy: "fedavg",
    dp_noise_multiplier: 0.0, mu: 0.01, learning_rate: 0.001,
    dropout_rate: 0.3,
  });

  const update = (k, v) => setCfg(p => ({ ...p, [k]: v }));
  const num = (k, label, min, max, step, note) => (
    <div key={k} style={{ marginBottom:14 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
        <label style={{ fontSize:12, fontWeight:500, color:C.text }}>{label}</label>
        <span style={{ fontSize:12, color:C.blue, fontWeight:600 }}>{cfg[k]}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={cfg[k]}
        onChange={e => update(k, parseFloat(e.target.value))}
        disabled={disabled}
        style={{ width:"100%", accentColor:C.blue }} />
      {note && <div style={{ fontSize:10, color:C.sub, marginTop:2 }}>{note}</div>}
    </div>
  );

  return (
    <div style={{ background:C.card, border:`1px solid ${C.border}`,
      borderRadius:12, padding:"24px" }}>
      <div style={{ fontSize:15, fontWeight:700, color:C.text, marginBottom:20 }}>
        Training Configuration
      </div>

      <SectionLabel>Strategy</SectionLabel>
      <div style={{ display:"flex", gap:8, marginBottom:20 }}>
        {["fedavg","fedprox","fedadam"].map(s => (
          <button key={s} onClick={() => update("strategy", s)} disabled={disabled}
            style={{ flex:1, padding:"8px 0", borderRadius:8, border:"none",
              background: cfg.strategy===s ? C.blue : "#F1F5F9",
              color: cfg.strategy===s ? "#fff" : C.text,
              fontWeight:600, fontSize:12, cursor:disabled?"not-allowed":"pointer",
              textTransform:"uppercase", letterSpacing:"0.04em" }}>
            {s}
          </button>
        ))}
      </div>

      <SectionLabel>Hyperparameters</SectionLabel>
      {num("num_rounds", "FL Rounds", 5, 50, 1)}
      {num("local_epochs", "Local Epochs", 1, 20, 1)}
      {num("learning_rate", "Learning Rate", 0.0001, 0.01, 0.0001, "Adam optimizer LR")}
      {num("dropout_rate", "Dropout Rate", 0, 0.6, 0.05)}

      <SectionLabel>Privacy & Regularization</SectionLabel>
      {num("dp_noise_multiplier", "DP Noise Multiplier", 0, 1, 0.05,
        "0 = no privacy; higher = more private, less accurate")}
      {num("mu", "FedProx μ", 0, 0.1, 0.005, "Proximal regularization (FedProx only)")}

      <button onClick={() => onStart(cfg)} disabled={disabled}
        style={{ width:"100%", marginTop:8, padding:"12px 0", borderRadius:10,
          background: disabled ? "#CBD5E1" : C.blue, border:"none", color:"#fff",
          fontSize:14, fontWeight:700, cursor: disabled?"not-allowed":"pointer",
          letterSpacing:"0.04em" }}>
        {disabled ? "Training in Progress…" : "▶ Start Training"}
      </button>
    </div>
  );
}

// ── Bank Profile Table ────────────────────────────────────────────────────────

function BankTable({ profiles }) {
  if (!profiles) return null;
  const banks = Object.entries(profiles);
  return (
    <div style={{ background:C.card, border:`1px solid ${C.border}`,
      borderRadius:12, overflow:"hidden" }}>
      <div style={{ padding:"16px 20px", borderBottom:`1px solid ${C.border}`,
        fontSize:13, fontWeight:700, color:C.text }}>
        Participating Banks
      </div>
      <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12 }}>
        <thead>
          <tr style={{ background:"#F8FAFC" }}>
            {["Bank","Transactions","Fraud Rate","Geography","Train Size"].map(h => (
              <th key={h} style={{ padding:"10px 16px", textAlign:"left",
                color:C.sub, fontWeight:600, letterSpacing:"0.04em",
                borderBottom:`1px solid ${C.border}`, textTransform:"uppercase",
                fontSize:10 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {banks.map(([id, p], i) => (
            <tr key={id} style={{ borderBottom:`1px solid ${C.border}` }}>
              <td style={{ padding:"10px 16px", fontWeight:600, color:C.text }}>
                <span style={{ display:"inline-block", width:8, height:8,
                  borderRadius:"50%", background:BANK_COLORS[i], marginRight:8 }} />
                {id}
              </td>
              <td style={{ padding:"10px 16px", color:C.sub }}>{p.n_transactions?.toLocaleString()}</td>
              <td style={{ padding:"10px 16px" }}>
                <span style={{ background:"#FEF3C7", color:"#92400E",
                  borderRadius:4, padding:"2px 6px", fontWeight:600 }}>
                  {pct(p.fraud_rate)}
                </span>
              </td>
              <td style={{ padding:"10px 16px", color:C.sub, textTransform:"capitalize" }}>{p.geography}</td>
              <td style={{ padding:"10px 16px", color:C.sub }}>{p.train_size?.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Progress Bar ──────────────────────────────────────────────────────────────

function ProgressBar({ current, total, status }) {
  const pct = total > 0 ? (current / total) * 100 : 0;
  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between",
        fontSize:12, color:C.sub, marginBottom:6 }}>
        <span>Round {current} / {total}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div style={{ height:6, background:"#E2E8F0", borderRadius:3, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${pct}%`,
          background: status==="completed" ? C.green : C.blue,
          borderRadius:3, transition:"width 0.4s ease" }} />
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [status, setStatus]   = useState("idle");
  const [history, setHistory] = useState([]);
  const [baseline, setBaseline] = useState(null);
  const [final, setFinal]     = useState(null);
  const [profiles, setProfiles] = useState(null);
  const [currentRound, setCurrentRound] = useState(0);
  const [totalRounds, setTotalRounds]   = useState(0);
  const [error, setError]     = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");
  const wsRef = useRef(null);

  // ── WebSocket ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let retryTimer;
    function connect() {
      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;
        ws.onopen = () => setWsConnected(true);
        ws.onclose = () => {
          setWsConnected(false);
          retryTimer = setTimeout(connect, 3000);
        };
        ws.onerror = () => ws.close();
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          handleWsMessage(msg);
        };
      } catch (_) {}
    }
    connect();
    return () => { clearTimeout(retryTimer); wsRef.current?.close(); };
  }, []);

  // Keep alive ping
  useEffect(() => {
    const t = setInterval(() => {
      if (wsRef.current?.readyState === 1) wsRef.current.send("ping");
    }, 25000);
    return () => clearInterval(t);
  }, []);

  const handleWsMessage = useCallback((msg) => {
    if (msg.type === "init" || msg.state) {
      const s = msg.state || msg;
      setStatus(s.status || "idle");
      setCurrentRound(s.current_round || 0);
      setTotalRounds(s.total_rounds || 0);
      if (s.baseline) setBaseline(s.baseline);
      if (s.final) setFinal(s.final);
      if (s.bank_profiles) setProfiles(s.bank_profiles);
    }
    if (msg.type === "round_update" && msg.data) {
      setHistory(h => {
        const exists = h.find(r => r.round === msg.data.round);
        return exists ? h.map(r => r.round === msg.data.round ? msg.data : r) : [...h, msg.data];
      });
      setCurrentRound(msg.data.round);
      if (msg.state?.baseline) setBaseline(msg.state.baseline);
      if (msg.state?.bank_profiles) setProfiles(msg.state.bank_profiles);
      setStatus("running");
    }
    if (msg.type === "training_complete") {
      setStatus("completed");
      if (msg.data?.final) setFinal(msg.data.final);
    }
    if (msg.type === "error") {
      setStatus("error");
      setError(msg.message);
    }
  }, []);

  // ── Actions ───────────────────────────────────────────────────────────────
  const handleStart = async (cfg) => {
    setError(null);
    setHistory([]);
    setBaseline(null);
    setFinal(null);
    setCurrentRound(0);
    setTotalRounds(cfg.num_rounds);
    setStatus("running");
    try {
      await apiPost("/start-training", cfg);
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  };

  const handleReset = async () => {
    try {
      await apiPost("/reset", {});
      setStatus("idle"); setHistory([]); setBaseline(null);
      setFinal(null); setCurrentRound(0); setTotalRounds(0); setError(null);
    } catch (e) { setError(e.message); }
  };

  const downloadResults = () => {
    const blob = new Blob([JSON.stringify({ history, baseline, final, profiles }, null, 2)], { type:"application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "fedfraud_results.json"; a.click();
    URL.revokeObjectURL(url);
  };

  // ── Derived data ──────────────────────────────────────────────────────────
  const latest = history[history.length - 1];
  const evalData = final || latest;
  const f1Delta  = evalData && baseline ? evalData.f1  - baseline.f1  : undefined;
  const aucDelta = evalData && baseline ? evalData.auc - baseline.auc : undefined;

  // Per-client radar data from latest round
  const clientRadar = latest?.client_metrics?.map(c => ({
    bank: c.bank_id.replace("Bank","").replace("Neo","Neo"),
    F1: +(c.f1 * 100).toFixed(1),
    AUC: +(c.auc * 100).toFixed(1),
    Recall: +(c.recall * 100).toFixed(1),
    Precision: +(c.precision * 100).toFixed(1),
  })) || [];

  const tabs = ["dashboard", "charts", "clients", "config"];

  return (
    <div style={{ minHeight:"100vh", background:C.bg, fontFamily:"'DM Sans', system-ui, sans-serif",
      color:C.text }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        input[type=range]:disabled { opacity:0.4; }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.3 } }
        ::-webkit-scrollbar { width:6px } ::-webkit-scrollbar-track { background:#F1F5F9 }
        ::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius:3px }
      `}</style>

      {/* Header */}
      <header style={{ background:C.card, borderBottom:`1px solid ${C.border}`,
        padding:"0 32px", display:"flex", alignItems:"center",
        justifyContent:"space-between", height:60, position:"sticky", top:0, zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{ width:32, height:32, borderRadius:8,
            background:`linear-gradient(135deg, ${C.blue}, ${C.indigo})`,
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:16 }}>🏦</div>
          <div>
            <div style={{ fontSize:15, fontWeight:700, color:C.text }}>FedFraud</div>
            <div style={{ fontSize:10, color:C.sub, letterSpacing:"0.06em" }}>FEDERATED FRAUD DETECTION</div>
          </div>
        </div>

        <nav style={{ display:"flex", gap:4 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setActiveTab(t)}
              style={{ padding:"6px 14px", borderRadius:8, border:"none",
                background: activeTab===t ? "#EFF6FF" : "transparent",
                color: activeTab===t ? C.blue : C.sub,
                fontWeight: activeTab===t ? 600 : 400,
                fontSize:13, cursor:"pointer", textTransform:"capitalize" }}>
              {t}
            </button>
          ))}
        </nav>

        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <span style={{ fontSize:11, color: wsConnected ? C.green : C.red,
            fontWeight:600, display:"flex", alignItems:"center", gap:4 }}>
            <span style={{ width:6, height:6, borderRadius:"50%",
              background: wsConnected ? C.green : C.red }} />
            {wsConnected ? "Live" : "Offline"}
          </span>
          <StatusBadge status={status} />
          {history.length > 0 && (
            <button onClick={downloadResults}
              style={{ padding:"6px 12px", borderRadius:8, border:`1px solid ${C.border}`,
                background:C.card, fontSize:12, fontWeight:500, cursor:"pointer", color:C.text }}>
              ↓ Export
            </button>
          )}
          {status !== "running" && status !== "idle" && (
            <button onClick={handleReset}
              style={{ padding:"6px 12px", borderRadius:8, border:`1px solid ${C.border}`,
                background:C.card, fontSize:12, fontWeight:500, cursor:"pointer", color:C.sub }}>
              Reset
            </button>
          )}
        </div>
      </header>

      <main style={{ maxWidth:1400, margin:"0 auto", padding:"28px 32px" }}>
        {error && (
          <div style={{ background:"#FEF2F2", border:`1px solid #FECACA`,
            borderRadius:10, padding:"12px 16px", marginBottom:20,
            color:C.red, fontSize:13, display:"flex", justifyContent:"space-between" }}>
            <span>⚠ {error}</span>
            <button onClick={() => setError(null)}
              style={{ background:"none", border:"none", cursor:"pointer", color:C.red, fontSize:16 }}>×</button>
          </div>
        )}

        {/* Progress (when running) */}
        {status === "running" && (
          <div style={{ background:C.card, border:`1px solid ${C.border}`,
            borderRadius:12, padding:"16px 24px", marginBottom:24 }}>
            <ProgressBar current={currentRound} total={totalRounds} status={status} />
          </div>
        )}

        {/* ── Dashboard Tab ── */}
        {activeTab === "dashboard" && (
          <div style={{ display:"grid", gridTemplateColumns:"320px 1fr", gap:24 }}>
            <div>
              <ControlPanel onStart={handleStart} disabled={status === "running"} />
              <div style={{ marginTop:20 }}>
                <BankTable profiles={profiles} />
              </div>
            </div>

            <div style={{ display:"flex", flexDirection:"column", gap:20 }}>
              {/* KPI row */}
              <div style={{ display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:16 }}>
                <KpiCard label="F1-Score" value={evalData ? fmt(evalData.f1) : "—"}
                  accent={C.blue} delta={f1Delta}
                  sub="Harmonic precision/recall" />
                <KpiCard label="ROC-AUC" value={evalData ? fmt(evalData.auc) : "—"}
                  accent={C.teal} delta={aucDelta}
                  sub="Fraud discrimination" />
                <KpiCard label="Recall" value={evalData ? pct(evalData.recall) : "—"}
                  accent={C.indigo}
                  sub="Fraud detection rate" />
              </div>

              <div style={{ display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:16 }}>
                <KpiCard label="Precision" value={evalData ? pct(evalData.precision) : "—"}
                  accent={C.amber} sub="Flag accuracy" />
                <KpiCard label="Loss" value={evalData ? fmt(evalData.loss, 4) : "—"}
                  accent={C.red} sub="Cross-entropy" />
                <KpiCard label="Rounds Completed"
                  value={`${currentRound} / ${totalRounds || "—"}`}
                  accent={C.green} sub={`Strategy: ${latest ? "active" : "pending"}`} />
              </div>

              {/* Quick F1 chart */}
              {history.length > 0 && (
                <ChartCard title="F1-Score Progress" height={200}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis dataKey="round" tick={{ fontSize:10, fill:C.sub }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize:10, fill:C.sub }} />
                      <Tooltip formatter={(v) => fmt(v)} contentStyle={{ fontSize:12, borderRadius:8, border:`1px solid ${C.border}` }} />
                      {baseline && <Line type="monotone" dataKey={() => baseline.f1} stroke="#CBD5E1" strokeDasharray="4 2" dot={false} name="Baseline" />}
                      <Line type="monotone" dataKey="f1" stroke={C.blue} strokeWidth={2.5} dot={false} name="Global F1" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

              {history.length === 0 && status === "idle" && (
                <div style={{ background:C.card, border:`1px solid ${C.border}`,
                  borderRadius:12, padding:48, textAlign:"center" }}>
                  <div style={{ fontSize:40, marginBottom:12 }}>🔐</div>
                  <div style={{ fontSize:15, fontWeight:600, color:C.text, marginBottom:6 }}>
                    Ready to train
                  </div>
                  <div style={{ fontSize:13, color:C.sub }}>
                    Configure your FL experiment in the panel and click Start Training.
                    <br/>Raw transaction data never leaves each bank.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Charts Tab ── */}
        {activeTab === "charts" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
            <ChartCard title="F1-Score vs Rounds" height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="round" tick={{ fontSize:11, fill:C.sub }} label={{ value:"Round", position:"insideBottom", offset:-2, fontSize:11 }} />
                  <YAxis domain={[0,1]} tick={{ fontSize:11, fill:C.sub }} />
                  <Tooltip formatter={v=>fmt(v)} contentStyle={{ fontSize:12, borderRadius:8 }} />
                  <Legend wrapperStyle={{ fontSize:12 }} />
                  <Line type="monotone" dataKey="f1" stroke={C.blue} strokeWidth={2.5} dot={{ r:2 }} name="F1" />
                  <Line type="monotone" dataKey="recall" stroke={C.teal} strokeWidth={2} dot={{ r:2 }} name="Recall" />
                  <Line type="monotone" dataKey="precision" stroke={C.indigo} strokeWidth={2} dot={{ r:2 }} name="Precision" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="ROC-AUC vs Rounds" height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="round" tick={{ fontSize:11, fill:C.sub }} />
                  <YAxis domain={[0.5,1]} tick={{ fontSize:11, fill:C.sub }} />
                  <Tooltip formatter={v=>fmt(v)} contentStyle={{ fontSize:12, borderRadius:8 }} />
                  <Line type="monotone" dataKey="auc" stroke={C.teal} strokeWidth={2.5} dot={{ r:2 }} name="AUC" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Training Loss Curve" height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="round" tick={{ fontSize:11, fill:C.sub }} />
                  <YAxis tick={{ fontSize:11, fill:C.sub }} />
                  <Tooltip formatter={v=>fmt(v,5)} contentStyle={{ fontSize:12, borderRadius:8 }} />
                  <Line type="monotone" dataKey="loss" stroke={C.red} strokeWidth={2.5} dot={{ r:2 }} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Accuracy vs Rounds" height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="round" tick={{ fontSize:11, fill:C.sub }} />
                  <YAxis domain={[0,1]} tick={{ fontSize:11, fill:C.sub }} />
                  <Tooltip formatter={v=>fmt(v)} contentStyle={{ fontSize:12, borderRadius:8 }} />
                  <Line type="monotone" dataKey="accuracy" stroke={C.amber} strokeWidth={2.5} dot={{ r:2 }} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Baseline vs Final comparison */}
            {baseline && evalData && (
              <div style={{ gridColumn:"1/-1" }}>
                <ChartCard title="Baseline vs Final — Global Metrics" height={220}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={["f1","auc","recall","precision","accuracy"].map(k=>({
                        metric: k.toUpperCase(),
                        Baseline: +(baseline[k]*100).toFixed(1),
                        Final: +(evalData[k]*100).toFixed(1),
                      }))}
                      margin={{ top:4, right:8, bottom:0, left:-10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis dataKey="metric" tick={{ fontSize:11, fill:C.sub }} />
                      <YAxis unit="%" tick={{ fontSize:11, fill:C.sub }} domain={[0,100]} />
                      <Tooltip formatter={v=>`${v}%`} contentStyle={{ fontSize:12, borderRadius:8 }} />
                      <Legend wrapperStyle={{ fontSize:12 }} />
                      <Bar dataKey="Baseline" fill="#CBD5E1" radius={[4,4,0,0]} />
                      <Bar dataKey="Final" fill={C.blue} radius={[4,4,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              </div>
            )}
          </div>
        )}

        {/* ── Clients Tab ── */}
        {activeTab === "clients" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
            {/* Per-client bar chart from latest round */}
            {clientRadar.length > 0 && (
              <ChartCard title="Latest Round — Per-Bank F1 & AUC" height={280}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={clientRadar} margin={{ top:4, right:8, bottom:20, left:-10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                    <XAxis dataKey="bank" tick={{ fontSize:11, fill:C.sub }} angle={-15} textAnchor="end" />
                    <YAxis unit="%" tick={{ fontSize:11, fill:C.sub }} domain={[0,100]} />
                    <Tooltip formatter={v=>`${v}%`} contentStyle={{ fontSize:12, borderRadius:8 }} />
                    <Legend wrapperStyle={{ fontSize:12 }} />
                    <Bar dataKey="F1" fill={C.blue} radius={[4,4,0,0]} />
                    <Bar dataKey="AUC" fill={C.teal} radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {clientRadar.length > 0 && (
              <ChartCard title="Bank Performance Radar" height={280}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={clientRadar}>
                    <PolarGrid stroke="#E2E8F0" />
                    <PolarAngleAxis dataKey="bank" tick={{ fontSize:11, fill:C.sub }} />
                    <Radar name="F1" dataKey="F1" stroke={C.blue} fill={C.blue} fillOpacity={0.15} />
                    <Radar name="Recall" dataKey="Recall" stroke={C.teal} fill={C.teal} fillOpacity={0.1} />
                    <Legend wrapperStyle={{ fontSize:12 }} />
                    <Tooltip formatter={v=>`${v}%`} contentStyle={{ fontSize:12, borderRadius:8 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {/* Client metrics over time for each bank */}
            {latest?.client_metrics?.map((cm, i) => (
              <ChartCard key={cm.bank_id} title={`${cm.bank_id} — Metrics Trend`} height={200}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={history.map(r => {
                      const c = r.client_metrics?.find(x => x.bank_id === cm.bank_id);
                      return c ? { round:r.round, f1:c.f1, auc:c.auc, recall:c.recall } : { round:r.round };
                    })}
                    margin={{ top:4, right:8, bottom:0, left:-20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                    <XAxis dataKey="round" tick={{ fontSize:10, fill:C.sub }} />
                    <YAxis domain={[0,1]} tick={{ fontSize:10, fill:C.sub }} />
                    <Tooltip formatter={v=>fmt(v)} contentStyle={{ fontSize:12, borderRadius:8 }} />
                    <Line type="monotone" dataKey="f1" stroke={BANK_COLORS[i]} strokeWidth={2} dot={false} name="F1" />
                    <Line type="monotone" dataKey="auc" stroke={BANK_COLORS[i]} strokeWidth={1.5} strokeDasharray="3 2" dot={false} name="AUC" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            ))}

            {clientRadar.length === 0 && (
              <div style={{ gridColumn:"1/-1", background:C.card, border:`1px solid ${C.border}`,
                borderRadius:12, padding:40, textAlign:"center", color:C.sub, fontSize:13 }}>
                Client metrics will appear after training starts.
              </div>
            )}
          </div>
        )}

        {/* ── Config Tab ── */}
        {activeTab === "config" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
            <div style={{ background:C.card, border:`1px solid ${C.border}`,
              borderRadius:12, padding:24 }}>
              <div style={{ fontSize:14, fontWeight:700, marginBottom:16 }}>Current Experiment Config</div>
              {latest ? (
                <pre style={{ fontFamily:"'DM Mono', monospace", fontSize:12,
                  color:C.text, lineHeight:1.8, whiteSpace:"pre-wrap",
                  background:"#F8FAFC", borderRadius:8, padding:16 }}>
                  {JSON.stringify({
                    strategy: "see header",
                    rounds_completed: history.length,
                    total_rounds: totalRounds,
                  }, null, 2)}
                </pre>
              ) : (
                <div style={{ color:C.sub, fontSize:13 }}>No experiment running yet.</div>
              )}
            </div>

            <div style={{ background:C.card, border:`1px solid ${C.border}`,
              borderRadius:12, padding:24 }}>
              <div style={{ fontSize:14, fontWeight:700, marginBottom:16 }}>Bank Profiles</div>
              {profiles ? (
                <pre style={{ fontFamily:"'DM Mono', monospace", fontSize:11,
                  color:C.text, lineHeight:1.8, whiteSpace:"pre-wrap",
                  background:"#F8FAFC", borderRadius:8, padding:16, maxHeight:400, overflowY:"auto" }}>
                  {JSON.stringify(profiles, null, 2)}
                </pre>
              ) : (
                <div style={{ color:C.sub, fontSize:13 }}>Start training to load bank profiles.</div>
              )}
            </div>

            <div style={{ gridColumn:"1/-1", background:C.card, border:`1px solid ${C.border}`,
              borderRadius:12, padding:24 }}>
              <div style={{ fontSize:14, fontWeight:700, marginBottom:6 }}>About FedFraud</div>
              <div style={{ fontSize:13, color:C.sub, lineHeight:1.8, maxWidth:700 }}>
                FedFraud enables multiple banking institutions to collaboratively train a fraud detection model 
                without sharing raw customer data. Each bank trains locally on private transaction data; 
                only encrypted model weights are exchanged with the central aggregation server.
                <br/><br/>
                <strong style={{ color:C.text }}>Strategies:</strong> FedAvg (weighted averaging) · 
                FedProx (proximal regularization for non-IID) · FedAdam (server-side adaptive optimizer)
                <br/>
                <strong style={{ color:C.text }}>Privacy:</strong> Differential Privacy via DP-SGD — 
                gradient clipping + calibrated Gaussian noise ensures plausible deniability.
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
