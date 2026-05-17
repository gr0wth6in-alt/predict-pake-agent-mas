export type Prediction = {
  symbol: string;
  direction_score: number;
  confidence: number;
  horizon_candles: number;
  rationale: string;
};

export type PaperRun = {
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

export function getPrediction(symbol: string): Promise<Prediction> {
  return postJson<Prediction>("/predict", { symbol });
}

export function runPaperOnce(symbol: string): Promise<PaperRun> {
  return postJson<PaperRun>("/paper/run-once", { symbol });
}

export function runBacktest(symbol: string): Promise<Backtest> {
  return postJson<Backtest>("/backtest", { symbol });
}
