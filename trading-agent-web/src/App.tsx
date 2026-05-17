import {
  Activity,
  ArrowRight,
  BarChart3,
  Bot,
  CheckCircle2,
  CircleDollarSign,
  Cpu,
  Database,
  Play,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp
} from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  Backtest,
  ModelStatus,
  PaperRun,
  Prediction,
  apiBaseUrl,
  getModelStatus,
  getPrediction,
  runBacktest,
  runPaperOnce
} from "./lib/api";

const sampleEquity = [
  { label: "01", equity: 10000 },
  { label: "04", equity: 10480 },
  { label: "08", equity: 10190 },
  { label: "12", equity: 10960 },
  { label: "16", equity: 11240 },
  { label: "20", equity: 11850 }
];

const coinPresets = [
  { label: "BTC", symbol: "BTCUSD", coinId: "bitcoin" },
  { label: "ETH", symbol: "ETHUSD", coinId: "ethereum" },
  { label: "SOL", symbol: "SOLUSD", coinId: "solana" },
  { label: "BNB", symbol: "BNBUSD", coinId: "binancecoin" },
  { label: "XRP", symbol: "XRPUSD", coinId: "ripple" },
  { label: "DOGE", symbol: "DOGEUSD", coinId: "dogecoin" }
];

type RunState = "idle" | "loading" | "ready" | "error";

export function App() {
  const [symbol, setSymbol] = useState("BTCUSD");
  const [coinId, setCoinId] = useState("bitcoin");
  const [days, setDays] = useState(30);
  const [status, setStatus] = useState<RunState>("idle");
  const [error, setError] = useState("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [paper, setPaper] = useState<PaperRun | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);

  const signalTone = useMemo(() => {
    const side = paper?.signal.side;
    if (side === "buy") return "positive";
    if (side === "sell") return "negative";
    return "neutral";
  }, [paper]);

  async function runFullCheck() {
    setStatus("loading");
    setError("");

    try {
      const coinRequest = {
        symbol,
        coin_id: coinId,
        vs_currency: "usd",
        days,
        data_source: "coingecko" as const
      };
      const [model, nextPrediction, nextPaper, nextBacktest] = await Promise.all([
        getModelStatus(),
        getPrediction(coinRequest),
        runPaperOnce(coinRequest),
        runBacktest(coinRequest)
      ]);
      setModelStatus(model);
      setPrediction(nextPrediction);
      setPaper(nextPaper);
      setBacktest(nextBacktest);
      setStatus("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown request error");
      setStatus("error");
    }
  }

  return (
    <main className="app-shell">
      <aside className="side-panel">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Bot size={22} aria-hidden="true" />
          </div>
          <div>
            <p className="eyebrow">Autonomous paper agent</p>
            <h1>Signal Deck</h1>
          </div>
        </div>

        <nav className="nav-stack" aria-label="Dashboard sections">
          <a className="nav-item active" href="#run">
            <Activity size={18} aria-hidden="true" />
            Run
          </a>
          <a className="nav-item" href="#risk">
            <ShieldCheck size={18} aria-hidden="true" />
            Risk
          </a>
          <a className="nav-item" href="#model">
            <Cpu size={18} aria-hidden="true" />
            Model
          </a>
        </nav>

        <div className="api-pill">
          <Radio size={16} aria-hidden="true" />
          <span>{apiBaseUrl.replace("https://", "")}</span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Paper trading control surface</p>
            <h2>AI signal, risk approval, and backtest pulse</h2>
          </div>
          <div className={`status-chip ${status}`}>
            {status === "loading" ? <RefreshCw size={16} /> : <CheckCircle2 size={16} />}
            <span>{status === "idle" ? "Ready to connect" : status}</span>
          </div>
        </header>

        <section id="run" className="command-band">
          <div className="preset-row" aria-label="Coin presets">
            {coinPresets.map((coin) => (
              <button
                key={coin.coinId}
                className={coinId === coin.coinId ? "preset active" : "preset"}
                onClick={() => {
                  setSymbol(coin.symbol);
                  setCoinId(coin.coinId);
                }}
                type="button"
              >
                {coin.label}
              </button>
            ))}
          </div>
          <div className="symbol-control">
            <label htmlFor="symbol">Symbol</label>
            <input
              id="symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              placeholder="BTCUSD"
            />
          </div>
          <div className="symbol-control">
            <label htmlFor="coinId">CoinGecko</label>
            <input
              id="coinId"
              value={coinId}
              onChange={(event) => setCoinId(event.target.value.toLowerCase())}
              placeholder="bitcoin"
            />
          </div>
          <div className="symbol-control compact">
            <label htmlFor="days">Days</label>
            <input
              id="days"
              value={days}
              min={1}
              max={365}
              type="number"
              onChange={(event) => setDays(Number(event.target.value))}
            />
          </div>
          <button className="primary-action" onClick={runFullCheck} disabled={status === "loading"}>
            {status === "loading" ? <RefreshCw size={18} /> : <Play size={18} />}
            Run agent check
          </button>
        </section>

        {error && (
          <div className="error-strip" role="alert">
            {error}
          </div>
        )}

        <section className="metric-grid" aria-label="Trading summary">
          <Metric
            icon={<Sparkles size={20} />}
            label="Prediction"
            value={prediction ? formatScore(prediction.direction_score) : "--"}
            detail={prediction?.rationale || `CoinGecko coin: ${coinId}`}
          />
          <Metric
            icon={signalTone === "negative" ? <TrendingDown size={20} /> : <TrendingUp size={20} />}
            label="Signal"
            value={paper?.signal.side.toUpperCase() || "--"}
            detail={paper ? `strength ${toPercent(paper.signal.strength)}` : "Paper signal not run"}
            tone={signalTone}
          />
          <Metric
            icon={<ShieldCheck size={20} />}
            label="Risk"
            value={paper?.risk.approved ? "APPROVED" : paper ? "BLOCKED" : "--"}
            detail={paper?.risk.reason || "Risk manager standing by"}
          />
          <Metric
            icon={<CircleDollarSign size={20} />}
            label="Backtest"
            value={backtest ? toPercent(backtest.total_return_pct / 100) : "--"}
            detail={backtest ? `${backtest.fills} fills from demo data` : "No backtest yet"}
          />
        </section>

        <section className="insight-layout">
          <div className="chart-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Equity curve</p>
                <h3>{backtest ? "Latest demo backtest" : "Preview trend"}</h3>
              </div>
              <BarChart3 size={20} aria-hidden="true" />
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sampleEquity}>
                  <defs>
                    <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#27d7a2" stopOpacity={0.42} />
                      <stop offset="95%" stopColor="#27d7a2" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1c3032" strokeDasharray="3 6" vertical={false} />
                  <XAxis dataKey="label" stroke="#8da4a6" tickLine={false} axisLine={false} />
                  <YAxis stroke="#8da4a6" tickLine={false} axisLine={false} width={52} />
                  <Tooltip
                    contentStyle={{
                      background: "#0d191c",
                      border: "1px solid #244044",
                      borderRadius: 8,
                      color: "#eef7f5"
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="#27d7a2"
                    strokeWidth={3}
                    fill="url(#equityFill)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="decision-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Execution preview</p>
                <h3>Paper fill</h3>
              </div>
              <ArrowRight size={20} aria-hidden="true" />
            </div>

            <div className="fill-card">
              <span>Side</span>
              <strong>{paper?.fill?.side.toUpperCase() || "NONE"}</strong>
            </div>
            <div className="fill-card">
              <span>Quantity</span>
              <strong>{paper?.fill ? paper.fill.quantity.toFixed(6) : "--"}</strong>
            </div>
            <div className="fill-card">
              <span>Notional</span>
              <strong>{paper?.fill ? currency(paper.fill.notional) : "--"}</strong>
            </div>
          </div>
        </section>

        <section id="risk" className="system-strip">
          <SystemItem icon={<Database size={18} />} label="Data" value="CoinGecko OHLC" />
          <SystemItem icon={<Cpu size={18} />} label="Horizon" value={prediction ? `${prediction.horizon_candles} candles` : "--"} />
          <SystemItem icon={<ShieldCheck size={18} />} label="Mode" value="paper only" />
        </section>
      </section>
    </main>
  );
}

function Metric({
  icon,
  label,
  value,
  detail,
  tone = "neutral"
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone?: "positive" | "negative" | "neutral";
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}

function SystemItem({
  icon,
  label,
  value
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="system-item">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function toPercent(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 1
  }).format(value);
}

function currency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2
  }).format(value);
}

function formatScore(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}
