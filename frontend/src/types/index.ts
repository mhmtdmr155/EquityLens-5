export interface Security {
  id: number;
  ticker: string;
  exchange: string;
  valid_from: string;
  valid_to: string | null;
  company_id: number;
}

export interface Company {
  id: number;
  company_name: string;
  sec_cik: string;
  created_at: string;
  securities: Security[];
}

export interface PriceBarDetail {
  session_date: string;
  open: number | string;
  high: number | string;
  low: number | string;
  close: number | string;
  volume: number;
}

export interface PriceSeriesResponse {
  ticker: string;
  company_name: string;
  provider: string;
  feed: string;
  count: number;
  bars: PriceBarDetail[];
}

export interface PredictionResponse {
  ticker: string;
  company_name: string;
  model_name: string;
  model_version: string;
  horizon_trading_days: number;
  as_of_date: string;
  predicted_direction: number; // 1 = UP, 0 = FLAT/DOWN
  probability_up: number;
  data_source: string;
  warning: string;
}

export interface ModelScoreDetail {
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number | null;
}

export interface ExperimentMetricsResponse {
  status: string;
  as_of_date: string;
  retrospective_models: Record<string, ModelScoreDetail>;
  expanding_cv_averages: Record<string, ModelScoreDetail>;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
}

export interface DatabaseHealthResponse {
  status: string;
  connected: boolean;
  dialect: string;
  error: string | null;
}

export interface SystemHealth {
  isOnline: boolean;
  dbConnected: boolean;
  dialect: string;
  version: string;
  lastChecked: string;
}

export type TimeHorizon = 30 | 90 | 180 | 500;
