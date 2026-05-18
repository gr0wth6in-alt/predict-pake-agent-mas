export type PredictorName = "auto" | "baseline" | "multi" | "ml" | "llm";

export type Prediction = {
  market_source?: string;
  predictor?: string;
  candle_count?: number;
  latest_price?: number;
  latest_timestamp?: string;
  symbol: string;
  direction_score: number;
  confidence: number;
  horizon_candles: number;
  rationale: string;
};

export type PaperRun = {
  market_source?: string;
  predictor?: string;
  candle_count?: number;
  latest_price?: number;
  latest_timestamp?: string;
  signal: {
    symbol: string;
    side: "buy" | "sell" | "hold";
    strength: number;
    reason: string;
  };
  risk: {
    approved: boolean;
    reason: string;
  };
  fill: null | {
    symbol: string;
    side: "buy" | "sell";
    quantity: number;
    price: number;
    notional: number;
    reason: string;
  };
  paper_cash: number;
};

export type Backtest = {
  market_source?: string;
  candle_count?: number;
  latest_price?: number;
  latest_timestamp?: string;
  starting_cash: number;
  ending_equity: number;
  total_return_pct: number;
  fills: number;
  equity_points: number;
  last_equity_point: null | {
    timestamp: string;
    equity: number;
    close: number;
  };
};

export type ModelStatus = {
  model_path: string;
  exists: boolean;
  message: string;
};

export type PredictorList = {
  predictors: PredictorName[];
  default: PredictorName;
  llm_available: boolean;
};

export type IndicatorSnapshot = {
  close?: number | null;
  sma_20?: number | null;
  ema_12?: number | null;
  ema_26?: number | null;
  ema_50?: number | null;
  rsi_14?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_histogram?: number | null;
  atr_14?: number | null;
  bollinger_lower?: number | null;
  bollinger_middle?: number | null;
  bollinger_upper?: number | null;
  bollinger_percent?: number | null;
  return_1?: number | null;
  return_5?: number | null;
  return_20?: number | null;
};

export type LiveTicker = {
  symbol: string;
  last_price: number;
  timestamp: string;
  bid?: number | null;
  ask?: number | null;
  high_24h?: number | null;
  low_24h?: number | null;
  volume_24h?: number | null;
  change_24h_pct?: number | null;
  source: string;
};

export type LiveMarket = {
  source: string;
  fallbacks: string[];
  symbol: string;
  interval: string;
  candles: number;
  ticker: LiveTicker;
  indicators: IndicatorSnapshot;
};

export type AutoTrainSummary = {
  started_at: string;
  finished_at: string;
  symbol: string;
  data_source: string;
  candle_count: number;
  output_path: string;
  samples: number;
  train_samples: number;
  test_samples: number;
  label_distribution: Record<string, number>;
  train_accuracy: number;
  test_accuracy: number;
  warnings: string[];
};

export type AutoTrainRequest = {
  symbol: string;
  output_path?: string;
  data_source?: "binance" | "coingecko" | "csv";
  interval?: string;
  limit?: number;
  days?: number;
  vs_currency?: string;
  coin_id?: string | null;
  csv_path?: string | null;
  lookback?: number;
  horizon?: number;
  label_threshold?: number;
  train_fraction?: number;
};

export type CoinRequest = {
  symbol: string;
  coin_id?: string | null;
  vs_currency?: string;
  days?: number;
  interval?: string;
  limit?: number;
  data_source: "coingecko" | "binance" | "live" | "csv";
  predictor?: PredictorName;
  model_path?: string | null;
};

const fallbackUrl = "https://YOUR_USERNAME.pythonanywhere.com";

export const apiBaseUrl =
  import.meta.env.VITE_AGENT_API_BASE_URL?.replace(/\/$/, "") || fallbackUrl;

async function getJson<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${apiBaseUrl}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }
  const response = await fetch(url.toString());
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getModelStatus(): Promise<ModelStatus> {
  return getJson<ModelStatus>("/model/status");
}

export function getPredictors(): Promise<PredictorList> {
  return getJson<PredictorList>("/predictors");
}

export function getLiveMarket(params: {
  symbol: string;
  interval?: string;
  limit?: number;
  days?: number | null;
  vs_currency?: string;
  coin_id?: string | null;
}): Promise<LiveMarket> {
  return getJson<LiveMarket>("/market/live", {
    symbol: params.symbol,
    interval: params.interval,
    limit: params.limit,
    days: params.days ?? undefined,
    vs_currency: params.vs_currency,
    coin_id: params.coin_id ?? undefined
  });
}

export function getPrediction(request: CoinRequest): Promise<Prediction> {
  return postJson<Prediction>("/predict", request);
}

export function runPaperOnce(request: CoinRequest): Promise<PaperRun> {
  return postJson<PaperRun>("/paper/run-once", request);
}

export function runBacktest(request: CoinRequest): Promise<Backtest> {
  return postJson<Backtest>("/backtest", request);
}

export function trainAuto(request: AutoTrainRequest): Promise<AutoTrainSummary> {
  return postJson<AutoTrainSummary>("/train/auto", request);
}
