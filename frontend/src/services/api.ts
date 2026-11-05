import {
  Company,
  DatabaseHealthResponse,
  ExperimentMetricsResponse,
  HealthResponse,
  PredictionResponse,
  PriceSeriesResponse,
} from '../types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Keep default message if json parse fails
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`, { signal });
  return handleResponse<HealthResponse>(res);
}

export async function fetchDatabaseHealth(signal?: AbortSignal): Promise<DatabaseHealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health/db`, { signal });
  return handleResponse<DatabaseHealthResponse>(res);
}

export async function fetchCompanies(signal?: AbortSignal): Promise<Company[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/companies`, { signal });
  return handleResponse<Company[]>(res);
}

export async function fetchPrices(
  ticker: string,
  limit: number = 180,
  order: 'asc' | 'desc' = 'desc',
  signal?: AbortSignal
): Promise<PriceSeriesResponse> {
  const url = `${API_BASE_URL}/api/v1/prices/${encodeURIComponent(ticker)}?limit=${limit}&order=${order}`;
  const res = await fetch(url, { signal });
  return handleResponse<PriceSeriesResponse>(res);
}

export async function fetchPrediction(
  ticker: string,
  signal?: AbortSignal
): Promise<PredictionResponse> {
  const url = `${API_BASE_URL}/api/v1/predictions/${encodeURIComponent(ticker)}`;
  const res = await fetch(url, { signal });
  return handleResponse<PredictionResponse>(res);
}

export async function fetchExperimentMetrics(
  signal?: AbortSignal
): Promise<ExperimentMetricsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/experiments/metrics`, { signal });
  return handleResponse<ExperimentMetricsResponse>(res);
}

