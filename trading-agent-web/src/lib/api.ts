export type Prediction = {
  market_source?: string;
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

export type CoinRequest = {
  symbol: string;
  coin_id: string;
  vs_currency: string;
  days: number;
  interval?: string;
  limit?: number;
  data_source: "coingecko";
};

const fallbackUrl = "https://YOUR_USERNAME.pythonanywhere.com";

export const apiBaseUrl =
  import.meta.env.VITE_AGENT_API_BASE_URL?.replace(/\/$/, "") || fallbackUrl;

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

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await fetch(`${apiBaseUrl}/model/status`);
  if (!response.ok) {
    throw new Error(`Model status failed with ${response.status}`);
  }
  return response.json() as Promise<ModelStatus>;
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
