import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { CompanySelector } from './components/CompanySelector';
import { StockSummary } from './components/StockSummary';
import { PriceChart } from './components/PriceChart';
import { PredictionCard } from './components/PredictionCard';
import { ModelPerformance } from './components/ModelPerformance';
import { DataSources } from './components/DataSources';
import { ErrorMessage } from './components/ErrorMessage';
import {
  Company,
  PriceBarDetail,
  PredictionResponse,
  ExperimentMetricsResponse,
  SystemHealth,
  TimeHorizon,
} from './types';
import {
  fetchHealth,
  fetchDatabaseHealth,
  fetchCompanies,
  fetchPrices,
  fetchPrediction,
  fetchExperimentMetrics,
} from './services/api';

export const App: React.FC = () => {
  // Selected State
  const [selectedTicker, setSelectedTicker] = useState<string>('AAPL');
  const [timeHorizon, setTimeHorizon] = useState<TimeHorizon>(180);

  // Data State
  const [companies, setCompanies] = useState<Company[]>([]);
  const [prices, setPrices] = useState<PriceBarDetail[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [metrics, setMetrics] = useState<ExperimentMetricsResponse | null>(null);

  // System Health State
  const [health, setHealth] = useState<SystemHealth>({
    isOnline: false,
    dbConnected: false,
    dialect: 'sqlite',
    version: '0.1.0',
    lastChecked: '',
  });

  // Loading States
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);
  const [isLoadingPrices, setIsLoadingPrices] = useState<boolean>(false);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Error States
  const [priceError, setPriceError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [backendOffline, setBackendOffline] = useState<boolean>(false);

  // In-flight request cancellation reference
  const abortControllerRef = useRef<AbortController | null>(null);

  // Check health and connectivity
  const checkHealth = useCallback(async () => {
    try {
      const [hRes, dbRes] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchDatabaseHealth().catch(() => null),
      ]);

      if (hRes) {
        setHealth({
          isOnline: true,
          dbConnected: dbRes?.connected ?? false,
          dialect: dbRes?.dialect ?? 'sqlite',
          version: hRes.version ?? '0.1.0',
          lastChecked: new Date().toLocaleTimeString('tr-TR'),
        });
        setBackendOffline(false);
      } else {
        setHealth((prev) => ({ ...prev, isOnline: false, dbConnected: false }));
        setBackendOffline(true);
      }
    } catch {
      setHealth((prev) => ({ ...prev, isOnline: false, dbConnected: false }));
      setBackendOffline(true);
    } finally {
      setIsLoadingHealth(false);
    }
  }, []);

  // Fetch company catalog and experiment metrics once
  const loadInitialCatalog = useCallback(async () => {
    try {
      const comps = await fetchCompanies().catch(() => []);
      if (comps.length > 0) {
        setCompanies(comps);
      }
    } catch {
      // Handled gracefully by fallback companies
    }

    try {
      const expMetrics = await fetchExperimentMetrics().catch(() => null);
      if (expMetrics) {
        setMetrics(expMetrics);
      }
    } catch {
      // Handled gracefully by default scores
    }
  }, []);

  // Fetch prices and prediction for selected ticker
  const loadTickerData = useCallback(
    async (ticker: string, horizon: TimeHorizon) => {
      // Cancel any ongoing in-flight requests for previous ticker/horizon
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setIsLoadingPrices(true);
      setIsLoadingPrediction(true);
      setPriceError(null);
      setPredictionError(null);

      // Fetch prices (request with order=desc to get the latest bars)
      try {
        const priceRes = await fetchPrices(ticker, horizon, 'desc', controller.signal);
        setPrices(priceRes.bars || []);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          setPriceError(err.message || 'Fiyat verisi yüklenemedi.');
          setPrices([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingPrices(false);
        }
      }

      // Fetch ML prediction
      try {
        const predRes = await fetchPrediction(ticker, controller.signal);
        setPrediction(predRes);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          setPredictionError(err.message || 'Tahmin verisi yüklenemedi.');
          setPrediction(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingPrediction(false);
        }
      }
    },
    []
  );

  // Initial load
  useEffect(() => {
    checkHealth();
    loadInitialCatalog();
  }, [checkHealth, loadInitialCatalog]);

  // When ticker or horizon changes
  useEffect(() => {
    loadTickerData(selectedTicker, timeHorizon);
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [selectedTicker, timeHorizon, loadTickerData]);

  // Manual refresh handler
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await checkHealth();
    await loadInitialCatalog();
    await loadTickerData(selectedTicker, timeHorizon);
    setIsRefreshing(false);
  };

  // Find active company details
  const activeCompany = companies.find((c) =>
    c.securities?.some((s) => s.ticker === selectedTicker)
  );

  // Identify latest bar and previous bar for summary
  // Since we fetch with order=desc, prices[0] is latest, prices[1] is previous
  const latestBar = prices.length > 0 ? prices[0] : undefined;
  const previousBar = prices.length > 1 ? prices[1] : undefined;

  return (
    <div className="app-layout">
      {/* Navbar */}
      <Header
        health={health}
        onRefresh={handleRefresh}
        isLoading={isRefreshing || isLoadingHealth}
      />

      {/* Main Container */}
      <main className="main-content">
        {/* Backend Offline Warning Banner */}
        {backendOffline && (
          <ErrorMessage
            title="FastAPI Backend Sunucusuna Bağlanılamadı"
            message="Yerel API servisi (http://localhost:8000) yanıt vermiyor. Lütfen backend servisini başlatın."
            onRetry={handleRefresh}
            isBackendOffline={true}
          />
        )}

        {/* 5-Company Selection Bar */}
        <CompanySelector
          companies={companies}
          selectedTicker={selectedTicker}
          onSelectTicker={(ticker) => {
            if (ticker !== selectedTicker) {
              setSelectedTicker(ticker);
            }
          }}
          isLoading={isLoadingPrices || isLoadingPrediction}
        />

        {/* Top Stock Summary Card */}
        <StockSummary
          ticker={selectedTicker}
          companyInfo={activeCompany}
          latestBar={latestBar}
          previousBar={previousBar}
        />

        {/* Central Dashboard Grid (Chart + Prediction) */}
        <div className="dashboard-grid">
          {/* Main Chart Column */}
          <div className="chart-column">
            {priceError ? (
              <ErrorMessage
                title="Fiyat Verisi Alınamadı"
                message={priceError}
                onRetry={() => loadTickerData(selectedTicker, timeHorizon)}
              />
            ) : (
              <PriceChart
                ticker={selectedTicker}
                bars={prices}
                timeHorizon={timeHorizon}
                onHorizonChange={(newHorizon) => setTimeHorizon(newHorizon)}
                isLoading={isLoadingPrices}
              />
            )}
          </div>

          {/* Side ML Prediction Column */}
          <div className="prediction-column">
            <PredictionCard
              prediction={prediction}
              isLoading={isLoadingPrediction}
              error={predictionError}
            />
          </div>
        </div>

        {/* Scientific Model Performance Section */}
        <ModelPerformance metrics={metrics} isLoading={isLoadingHealth} />

        {/* Verified Data Sources Section */}
        <DataSources />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-container">
          <p className="footer-text">
            <strong>EquityLens-5</strong> • Üniversite Makine Öğrenmesi Dersi Projesi • Aşama 07: Web Arayüzü & Demo
          </p>
          <p className="footer-subtext">
            Tüm analizler 30.10.2026 piyasa kapanış verileriyle sınırlıdır. Yatırım tavsiyesi veya getiri vaadi içermez.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;

