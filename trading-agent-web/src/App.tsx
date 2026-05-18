import {
  Activity,
  ArrowRight,
  BarChart3,
  Bot,
  Brain,
  CheckCircle2,
  CircleDollarSign,
  Cpu,
  Database,
  GraduationCap,
  Play,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Zap
} from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
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
  AutoTrainSummary,
  Backtest,
  CoinRequest,
  LiveMarket,
  ModelStatus,
  PaperRun,
  Prediction,
  PredictorList,
  PredictorName,
  apiBaseUrl,
  getLiveMarket,
  getModelStatus,
  getPrediction,
  getPredictors,
  runBacktest,
  runPaperOnce,
  trainAuto
} from "./lib/api";

const coinPresets = [
  { label: "BTC", symbol: "BTCUSDT", coingeckoSymbol: "BTCUSD", coinId: "bitcoin" },
  { label: "ETH", symbol: "ETHUSDT", coingeckoSymbol: "ETHUSD", coinId: "ethereum" },
  { label: "SOL", symbol: "SOLUSDT", coingeckoSymbol: "SOLUSD", coinId: "solana" },
  { label: "BNB", symbol: "BNBUSDT", coingeckoSymbol: "BNBUSD", coinId: "binancecoin" },
  { label: "XRP", symbol: "XRPUSDT", coingeckoSymbol: "XRPUSD", coinId: "ripple" },
  { label: "DOGE", symbol: "DOGEUSDT", coingeckoSymbol: "DOGEUSD", coinId: "dogecoin" }
];

const intervalOptions = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];

type RunState = "idle" | "loading" | "ready" | "error";
type DataSource = "live" | "binance" | "coingecko";

export function App() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [coingeckoSymbol, setCoingeckoSymbol] = useState("BTCUSD");
  const [coinId, setCoinId] = useState("bitcoin");
  const [days, setDays] = useState(30);
  const [dataSource, setDataSource] = useState<DataSource>("live");
  const [predictor, setPredictor] = useState<PredictorName>("auto");
  const [predictors, setPredictors] = useState<PredictorList | null>(null);
  // Per-coin candle interval. Persists across the dashboard so the user can mix
  // BTCUSDT@1m and ETHUSDT@5m and the autotrain knows which interval to use.
  const [intervalsBySymbol, setIntervalsBySymbol] = useState<Record<string, string>>({
    BTCUSDT: "1h"
  });
  const intervalForSymbol = (target: string) => intervalsBySymbol[target] ?? "1h";
  const interval = intervalForSymbol(symbol);
  const setIntervalForSymbol = (target: string, next: string) =>
    setIntervalsBySymbol((current) => ({ ...current, [target]: next }));
  const [status, setStatus] = useState<RunState>("idle");
  const [error, setError] = useState("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [paper, setPaper] = useState<PaperRun | null>(null);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [live, setLive] = useState<LiveMarket | null>(null);
  const [liveStatus, setLiveStatus] = useState<RunState>("idle");
  const [trainStatus, setTrainStatus] = useState<RunState>("idle");
  const [trainSummary, setTrainSummary] = useState<AutoTrainSummary | null>(null);

  // Pull the predictor list once so the dropdown reflects the backend's reality.
  useEffect(() => {
    getPredictors()
      .then(setPredictors)
      .catch(() => setPredictors(null));
  }, []);

  // Refresh the live ticker every 10 seconds while the page is open.
  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      setLiveStatus("loading");
      try {
        const snapshot = await getLiveMarket({ symbol, interval });
        if (!cancelled) {
          setLive(snapshot);
          setLiveStatus("ready");
        }
      } catch (err) {
        if (!cancelled) {
          setLive(null);
          setLiveStatus("error");
          console.warn("live market refresh failed", err);
        }
      }
    }

    refresh();
    // Poll once every 30s. Slow enough to stay under CoinGecko free-tier limits
    // when the deploy host cannot reach Binance directly. Backend caches further.
    const handle = window.setInterval(refresh, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [symbol, interval]);

  const signalTone = useMemo(() => {
    const side = paper?.signal.side;
    if (side === "buy") return "positive";
    if (side === "sell") return "negative";
    return "neutral";
  }, [paper]);
  const equityData = useMemo(() => {
    if (!backtest) return [];
    return [
      { label: "Start", equity: backtest.starting_cash },
      { label: "End", equity: backtest.ending_equity }
    ];
  }, [backtest]);

  function buildRequest(): CoinRequest {
    if (dataSource === "coingecko") {
      return {
        symbol: coingeckoSymbol,
        coin_id: coinId,
        vs_currency: "usd",
        days,
        data_source: "coingecko",
        predictor
      };
    }
    return {
      symbol,
      days,
      interval,
      data_source: dataSource,
      predictor
    };
  }

  async function runFullCheck() {
    setStatus("loading");
    setError("");
    setModelStatus(null);
    setPrediction(null);
    setPaper(null);
    setBacktest(null);

    try {
      const request = buildRequest();
      const [model, nextPrediction, nextPaper, nextBacktest] = await Promise.all([
        getModelStatus(),
        getPrediction(request),
        runPaperOnce(request),
        runBacktest(request)
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

  async function runTrainNow() {
    setTrainStatus("loading");
    try {
      const summary = await trainAuto({
        symbol: dataSource === "coingecko" ? coingeckoSymbol : symbol,
        data_source: dataSource === "live" ? "binance" : (dataSource as "binance" | "coingecko"),
        days,
        interval,
        coin_id: dataSource === "coingecko" ? coinId : null,
        output_path: `models/${(dataSource === "coingecko" ? coingeckoSymbol : symbol).toLowerCase()}_${interval}_auto_nb.json`,
        label_threshold: 0.005
      });
      setTrainSummary(summary);
      setTrainStatus("ready");
      try {
        setModelStatus(await getModelStatus());
      } catch {
        // ignore – the train summary already shows the path.
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training request failed");
      setTrainStatus("error");
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
          <a className="nav-item" href="#live">
            <Radio size={18} aria-hidden="true" />
            Live
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
                className={symbol === coin.symbol ? "preset active" : "preset"}
                onClick={() => {
                  setSymbol(coin.symbol);
                  setCoingeckoSymbol(coin.coingeckoSymbol);
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
              placeholder="BTCUSDT"
            />
          </div>
          <div className="symbol-control">
            <label htmlFor="coinId">Coin ID</label>
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
          <div className="symbol-control compact">
            <label htmlFor="interval">Interval</label>
            <select
              id="interval"
              value={interval}
              onChange={(event) => setIntervalForSymbol(symbol, event.target.value)}
            >
              {intervalOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <div className="symbol-control">
            <label htmlFor="data-source">Source</label>
            <select
              id="data-source"
              value={dataSource}
              onChange={(event) => setDataSource(event.target.value as DataSource)}
            >
              <option value="live">live (binance, fallback coingecko)</option>
              <option value="binance">binance</option>
              <option value="coingecko">coingecko (fallback only)</option>
            </select>
          </div>
          <div className="symbol-control">
            <label htmlFor="predictor">Predictor</label>
            <select
              id="predictor"
              value={predictor}
              onChange={(event) => setPredictor(event.target.value as PredictorName)}
            >
              {(predictors?.predictors || ["auto", "baseline", "multi", "ml", "llm"]).map(
                (name) => (
                  <option
                    key={name}
                    value={name}
                    disabled={name === "llm" && predictors !== null && !predictors.llm_available}
                  >
                    {name}
                    {name === "llm" && predictors !== null && !predictors.llm_available
                      ? " (no API key)"
                      : ""}
                  </option>
                )
              )}
            </select>
          </div>
          <button className="primary-action" onClick={runFullCheck} disabled={status === "loading"}>
            {status === "loading" ? <RefreshCw size={18} /> : <Play size={18} />}
            Run agent check
          </button>
          <button
            className="secondary-action"
            onClick={runTrainNow}
            disabled={trainStatus === "loading"}
            type="button"
          >
            {trainStatus === "loading" ? <RefreshCw size={18} /> : <GraduationCap size={18} />}
            Train now
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
            detail={
              prediction
                ? `${prediction.predictor || predictor} • ${prediction.rationale}`.slice(0, 110)
                : "Pick a predictor and run"
            }
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
            label="Latest Price"
            value={
              live?.ticker.last_price
                ? currency(live.ticker.last_price)
                : backtest?.latest_price
                ? currency(backtest.latest_price)
                : "--"
            }
            detail={
              live
                ? `${live.source} • ${
                    live.ticker.change_24h_pct != null
                      ? `${live.ticker.change_24h_pct.toFixed(2)}% 24h`
                      : "live tick"
                  }`
                : backtest
                ? `${toPercent(backtest.total_return_pct / 100)} backtest, ${backtest.fills} fills`
                : "Waiting for live tick"
            }
          />
        </section>

        <section id="live" className="insight-layout">
          <div className="chart-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Live pulse</p>
                <h3>
                  {symbol}
                  <span className="interval-tag">@{interval}</span>
                </h3>
              </div>
              <Zap
                size={20}
                aria-hidden="true"
                className={liveStatus === "loading" ? "spin" : undefined}
              />
            </div>
            {live ? (
              <div className="live-grid">
                <LiveStat label="Last" value={currency(live.ticker.last_price)} />
                <LiveStat
                  label="24h change"
                  value={
                    live.ticker.change_24h_pct != null
                      ? `${live.ticker.change_24h_pct.toFixed(2)}%`
                      : "--"
                  }
                  tone={
                    live.ticker.change_24h_pct == null
                      ? "neutral"
                      : live.ticker.change_24h_pct >= 0
                      ? "positive"
                      : "negative"
                  }
                />
                <LiveStat
                  label="24h high"
                  value={live.ticker.high_24h != null ? currency(live.ticker.high_24h) : "--"}
                />
                <LiveStat
                  label="24h low"
                  value={live.ticker.low_24h != null ? currency(live.ticker.low_24h) : "--"}
                />
                <LiveStat
                  label="RSI 14"
                  value={fmtFixed(live.indicators.rsi_14, 1)}
                  tone={rsiTone(live.indicators.rsi_14)}
                />
                <LiveStat
                  label="MACD hist"
                  value={fmtFixed(live.indicators.macd_histogram, 2)}
                  tone={signTone(live.indicators.macd_histogram)}
                />
                <LiveStat
                  label="EMA 12 / 26"
                  value={`${fmtFixed(live.indicators.ema_12, 1)} / ${fmtFixed(
                    live.indicators.ema_26,
                    1
                  )}`}
                />
                <LiveStat
                  label="Bollinger %B"
                  value={fmtFixed(live.indicators.bollinger_percent, 2)}
                  tone={bollingerTone(live.indicators.bollinger_percent)}
                />
              </div>
            ) : (
              <div className="empty-chart">
                {liveStatus === "loading"
                  ? "Loading live ticker..."
                  : "Live feed unavailable. Confirm the API host can reach Binance or CoinGecko."}
              </div>
            )}
          </div>

          <div className="decision-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Self-training</p>
                <h3>Latest auto-train</h3>
              </div>
              <Brain size={20} aria-hidden="true" />
            </div>
            {trainSummary ? (
              <>
                <div className="fill-card">
                  <span>Model</span>
                  <strong title={trainSummary.output_path}>
                    {trainSummary.output_path.split("/").pop()}
                  </strong>
                </div>
                <div className="fill-card">
                  <span>Test accuracy</span>
                  <strong>{toPercent(trainSummary.test_accuracy)}</strong>
                </div>
                <div className="fill-card">
                  <span>Samples</span>
                  <strong>
                    {trainSummary.samples} ({trainSummary.train_samples}/
                    {trainSummary.test_samples})
                  </strong>
                </div>
                <div className="fill-card">
                  <span>Source</span>
                  <strong>
                    {trainSummary.data_source} • {trainSummary.candle_count} candles
                  </strong>
                </div>
                {trainSummary.warnings.length > 0 && (
                  <div className="warning-strip">
                    {trainSummary.warnings.slice(0, 2).join("; ")}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-chart small">
                Click <strong>Train now</strong> to fetch fresh data and retrain the JSON model.
              </div>
            )}
          </div>
        </section>

        <section className="insight-layout">
          <div className="chart-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Equity curve</p>
                <h3>{backtest ? `${symbol} paper backtest` : "Run a backtest"}</h3>
              </div>
              <BarChart3 size={20} aria-hidden="true" />
            </div>
            <div className="chart-wrap">
              {equityData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={equityData}>
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
              ) : (
                <div className="empty-chart">
                  Run the agent to load real candles and an equity curve.
                </div>
              )}
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
          <SystemItem
            icon={<Database size={18} />}
            label="Data"
            value={live ? `live (${live.source})` : "no live tick yet"}
          />
          <SystemItem
            icon={<Cpu size={18} />}
            label="Predictor"
            value={prediction?.predictor || predictor}
          />
          <SystemItem
            icon={<Brain size={18} />}
            label="Model"
            value={modelStatus ? (modelStatus.exists ? "trained" : "missing") : "--"}
          />
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

function LiveStat({
  label,
  value,
  tone = "neutral"
}: {
  label: string;
  value: string;
  tone?: "positive" | "negative" | "neutral";
}) {
  return (
    <div className={`live-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
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

function fmtFixed(value: number | null | undefined, digits: number): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

function rsiTone(value: number | null | undefined): "positive" | "negative" | "neutral" {
  if (value == null) return "neutral";
  if (value <= 30) return "positive"; // oversold tends to bounce
  if (value >= 70) return "negative"; // overbought tends to fade
  return "neutral";
}

function signTone(value: number | null | undefined): "positive" | "negative" | "neutral" {
  if (value == null) return "neutral";
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "neutral";
}

function bollingerTone(value: number | null | undefined): "positive" | "negative" | "neutral" {
  if (value == null) return "neutral";
  if (value <= 0.1) return "positive";
  if (value >= 0.9) return "negative";
  return "neutral";
}
