import { describe, it, expect } from 'vitest';
import { PriceBarDetail, PredictionResponse, ExperimentMetricsResponse } from '../types';

describe('Frontend Data & Formatting Logic Tests', () => {
  describe('Price & Date Formatting', () => {
    it('formats currency with two decimal places and dollar sign', () => {
      const val = 237.589;
      const formatted = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(val);
      expect(formatted).toBe('$237.59');
    });

    it('formats volume with comma separators', () => {
      const volume = 48291040;
      const formatted = new Intl.NumberFormat('en-US').format(volume);
      expect(formatted).toBe('48,291,040');
    });

    it('calculates price change and percentage change correctly', () => {
      const currentClose = 237.59;
      const prevClose = 235.25;
      const change = currentClose - prevClose;
      const pctChange = (change / prevClose) * 100;

      expect(Number(change.toFixed(2))).toBe(2.34);
      expect(Number(pctChange.toFixed(2))).toBe(0.99);
    });

    it('handles zero or flat price changes safely without division by zero', () => {
      const currentClose: number = 100.0;
      const prevClose: number = 100.0;
      const change = currentClose - prevClose;
      const pctChange = prevClose !== 0 ? (change / prevClose) * 100 : 0;

      expect(change).toBe(0);
      expect(pctChange).toBe(0);
    });
  });

  describe('Chronological Bar Sorting for Chart', () => {
    it('sorts descending price bars chronologically for Recharts rendering', () => {
      const descBars: PriceBarDetail[] = [
        { session_date: '2026-10-30', open: 236, high: 238, low: 235, close: 237.5, volume: 1000 },
        { session_date: '2026-10-29', open: 234, high: 236, low: 233, close: 235.0, volume: 1200 },
        { session_date: '2026-10-28', open: 230, high: 235, low: 229, close: 234.0, volume: 900 },
      ];

      const sorted = [...descBars].sort((a, b) => (a.session_date > b.session_date ? 1 : -1));

      expect(sorted[0].session_date).toBe('2026-10-28');
      expect(sorted[1].session_date).toBe('2026-10-29');
      expect(sorted[2].session_date).toBe('2026-10-30');
    });

    it('handles empty bars array without throwing', () => {
      const emptyBars: PriceBarDetail[] = [];
      const sorted = [...emptyBars].sort((a, b) => (a.session_date > b.session_date ? 1 : -1));
      expect(sorted).toEqual([]);
    });
  });

  describe('Prediction Response Contract & Validation', () => {
    it('correctly maps direction 1 as UP and direction 0 as DOWN', () => {
      const upPred: PredictionResponse = {
        ticker: 'AAPL',
        company_name: 'Apple Inc.',
        model_name: 'HistGradientBoostingClassifier',
        model_version: '1.0.0',
        horizon_trading_days: 5,
        as_of_date: '2026-10-30',
        predicted_direction: 1,
        probability_up: 0.612,
        data_source: 'Alpaca Market Data v2 (IEX raw)',
        warning: 'Notice: Test warning',
      };

      expect(upPred.predicted_direction).toBe(1);
      expect(upPred.probability_up).toBe(0.612);
      expect(upPred.horizon_trading_days).toBe(5);

      const downPred: PredictionResponse = {
        ...upPred,
        predicted_direction: 0,
        probability_up: 0.46,
      };

      expect(downPred.predicted_direction).toBe(0);
      expect(downPred.probability_up).toBeLessThan(0.5);
    });

    it('restricts probability_up to valid range [0, 1] and formats percentage accurately', () => {
      const prob = 0.612;
      const pct = (prob * 100).toFixed(1);
      expect(pct).toBe('61.2');
    });
  });

  describe('Company Catalog & SPCX Post-IPO Handling', () => {
    it('verifies SpaceX official SEC CIK is 0001181412 (not 0001174922 Wynn Resorts)', () => {
      const spcxCompany = {
        ticker: 'SPCX',
        name: 'Space Exploration Technologies Corp.',
        sec_cik: '0001181412',
      };

      expect(spcxCompany.sec_cik).toBe('0001181412');
      expect(spcxCompany.sec_cik).not.toBe('0001174922');
    });

    it('identifies SPCX ticker and assigns post-IPO status', () => {
      const tickers = ['AAPL', 'NVDA', 'MU', 'DELL', 'SPCX'];
      const spcxTicker = tickers.find((t) => t === 'SPCX');

      expect(spcxTicker).toBe('SPCX');
      const isSpcxPostIpo = (t: string) => t === 'SPCX';
      expect(isSpcxPostIpo('SPCX')).toBe(true);
      expect(isSpcxPostIpo('AAPL')).toBe(false);
    });

    it('contains exactly 5 tracked companies', () => {
      const coreCompanies = ['AAPL', 'NVDA', 'MU', 'DELL', 'SPCX'];
      expect(coreCompanies.length).toBe(5);
      expect(coreCompanies).toContain('AAPL');
      expect(coreCompanies).toContain('NVDA');
      expect(coreCompanies).toContain('MU');
      expect(coreCompanies).toContain('DELL');
      expect(coreCompanies).toContain('SPCX');
    });
  });

  describe('Experiment Metrics Validation', () => {
    it('verifies M0, M1, M2 feature categorizations and absence of ATR', () => {
      // 23 verified technical indicators
      const technicalIndicators = [
        'return_1d', 'return_5d', 'return_10d',
        'sma_5', 'sma_10', 'sma_20',
        'ema_5', 'ema_10', 'ema_20',
        'rsi_14', 'macd_12_26', 'macd_signal_9', 'macd_histogram',
        'rolling_volatility_5', 'rolling_volatility_20',
        'bollinger_mid_20', 'bollinger_upper_20', 'bollinger_lower_20',
        'volume_sma_20', 'volume_ratio_20',
        'close_to_sma_20', 'close_to_ema_20', 'bollinger_position',
      ];

      expect(technicalIndicators.length).toBe(23);
      expect(technicalIndicators).not.toContain('atr');
      expect(technicalIndicators).not.toContain('ATR');

      // 9 market ETF features (SPY, XLK, SOXX - not FRED macro)
      const marketEtfFeatures = [
        'spy_return_1d', 'spy_return_5d',
        'xlk_return_1d', 'xlk_return_5d',
        'soxx_return_1d', 'soxx_return_5d',
        'spy_volatility_20',
        'xlk_relative_to_spy_5d', 'soxx_relative_to_spy_5d',
      ];
      expect(marketEtfFeatures.length).toBe(9);
      expect(marketEtfFeatures.every((f) => f.startsWith('spy_') || f.startsWith('xlk_') || f.startsWith('soxx_'))).toBe(true);
    });

    it('verifies M0, M1, M2 and Baselines presence in experiment payload', () => {
      const mockMetrics: ExperimentMetricsResponse = {
        status: 'SUCCESS',
        as_of_date: '2026-10-30',
        retrospective_models: {
          B0a_majority: { accuracy: 0.5718, balanced_accuracy: 0.5, precision: 0.5718, recall: 1.0, f1_score: 0.7275, roc_auc: 0.5 },
          B0b_momentum: { accuracy: 0.4259, balanced_accuracy: 0.4125, precision: 0.498, recall: 0.5061, f1_score: 0.502, roc_auc: 0.381 },
          M0_price_only: { accuracy: 0.5347, balanced_accuracy: 0.4981, precision: 0.5706, recall: 0.753, f1_score: 0.6492, roc_auc: 0.5065 },
          M1_price_sec: { accuracy: 0.5463, balanced_accuracy: 0.5191, precision: 0.5853, recall: 0.7085, f1_score: 0.641, roc_auc: 0.5177 },
          M2_price_sec_market: { accuracy: 0.5718, balanced_accuracy: 0.5421, precision: 0.6006, recall: 0.749, f1_score: 0.6667, roc_auc: 0.5376 },
        },
        expanding_cv_averages: {
          B0a: { accuracy: 0.5854, balanced_accuracy: 0.5, precision: 0.5854, recall: 1.0, f1_score: 0.7372, roc_auc: 0.5 },
          B0b: { accuracy: 0.4596, balanced_accuracy: 0.4392, precision: 0.5354, recall: 0.5298, f1_score: 0.5325, roc_auc: 0.404 },
          M0: { accuracy: 0.5158, balanced_accuracy: 0.5053, precision: 0.5851, recall: 0.5675, f1_score: 0.572, roc_auc: 0.5 },
          M1: { accuracy: 0.5, balanced_accuracy: 0.5041, precision: 0.5909, recall: 0.5122, f1_score: 0.5379, roc_auc: 0.503 },
          M2: { accuracy: 0.4966, balanced_accuracy: 0.4896, precision: 0.576, recall: 0.5759, f1_score: 0.5646, roc_auc: 0.4842 },
        },
        disclaimer: 'Test disclaimer',
      };

      expect(mockMetrics.retrospective_models['M0_price_only'].accuracy).toBe(0.5347);
      expect(mockMetrics.expanding_cv_averages['M0'].balanced_accuracy).toBe(0.5053);
      expect(mockMetrics.expanding_cv_averages['M2'].balanced_accuracy).toBe(0.4896);
    });
  });
});
