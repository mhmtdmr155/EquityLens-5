import React from 'react';
import { PredictionResponse } from '../types';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Clock,
  Sparkles,
} from 'lucide-react';

interface PredictionCardProps {
  prediction: PredictionResponse | null;
  isLoading: boolean;
  error: string | null;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return (
      <div className="prediction-card loading">
        <div className="card-header-sm">
          <Sparkles size={18} className="text-blue-400" />
          <h3>Makine Öğrenmesi Yön Tahmini (5 Günlük)</h3>
        </div>
        <div className="prediction-skeleton">
          <div className="skeleton-line line-lg"></div>
          <div className="skeleton-line line-md"></div>
          <div className="skeleton-line line-sm"></div>
        </div>
      </div>
    );
  }

  if (error || !prediction) {
    return (
      <div className="prediction-card error-state">
        <div className="card-header-sm">
          <AlertTriangle size={18} className="text-amber-400" />
          <h3>Model Tahmini Alınamadı</h3>
        </div>
        <p className="prediction-error-text">
          {error || 'Model servisine bağlanılamadı veya veri yetersiz.'}
        </p>
      </div>
    );
  }

  const isUp = prediction.predicted_direction === 1;
  const confidencePct = (prediction.probability_up * 100).toFixed(1);

  return (
    <div className={`prediction-card ${isUp ? 'pred-up' : 'pred-down'}`}>
      <div className="prediction-card-top">
        <div className="pred-title-group">
          <div className="pred-icon-box">
            <Sparkles size={20} className={isUp ? 'text-emerald-400' : 'text-amber-400'} />
          </div>
          <div>
            <h3 className="prediction-main-title">
              Makine Öğrenmesi Fiyat Yönü Tahmini
            </h3>
            <span className="prediction-subtitle">
              5 İşlem Günü Ufku (5-Day Direction Horizon)
            </span>
          </div>
        </div>

        <div className="pred-horizon-badge">
          <Clock size={13} className="mr-1" />
          <span>Ufuk: 5 İşlem Günü</span>
        </div>
      </div>

      {/* Main Signal Display */}
      <div className="prediction-signal-box">
        <div className={`signal-direction-pill ${isUp ? 'signal-up' : 'signal-down'}`}>
          {isUp ? (
            <>
              <TrendingUp size={28} className="signal-icon" />
              <div>
                <span className="signal-label">Tahmin Edilen Yön:</span>
                <span className="signal-value">YUKARI (Pozitif Eğilim)</span>
              </div>
            </>
          ) : (
            <>
              <TrendingDown size={28} className="signal-icon" />
              <div>
                <span className="signal-label">Tahmin Edilen Yön:</span>
                <span className="signal-value">DÜZ / AŞAĞI (Nötr / Negatif)</span>
              </div>
            </>
          )}
        </div>

        <div className="confidence-metric-box">
          <span className="confidence-label">Yukarı Yön Skoru (Model Skoru)</span>
          <div className="confidence-number-row">
            <span className="confidence-value font-mono">%{confidencePct}</span>
            <span className="confidence-subtext">Algoritmik Skor</span>
          </div>
          {/* Visual progress gauge */}
          <div className="confidence-progress-bg">
            <div
              className={`confidence-progress-bar ${isUp ? 'bg-emerald-500' : 'bg-amber-500'}`}
              style={{ width: `${Math.min(Math.max(prediction.probability_up * 100, 5), 100)}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Model Metadata Grid */}
      <div className="pred-meta-grid">
        <div className="meta-box">
          <span className="meta-label">Kullanılan Model</span>
          <span className="meta-val font-mono">{prediction.model_name}</span>
        </div>
        <div className="meta-box">
          <span className="meta-label">Model Sürümü</span>
          <span className="meta-val font-mono">v{prediction.model_version} (Üretim)</span>
        </div>
        <div className="meta-box">
          <span className="meta-label">Girdi Veri Tarihi</span>
          <span className="meta-val font-mono">{prediction.as_of_date} Kapanış</span>
        </div>
        <div className="meta-box">
          <span className="meta-label">Veri Kaynağı</span>
          <span className="meta-val">{prediction.data_source}</span>
        </div>
      </div>

      {/* Mandatory Scientific Disclaimer Callout */}
      <div className="scientific-disclaimer-callout">
        <div className="disclaimer-header">
          <AlertTriangle size={15} className="text-amber-400 mr-1.5 flex-shrink-0" />
          <span className="font-semibold text-amber-300">Bilimsel Araştırma ve Risk Bildirimi</span>
        </div>
        <p className="disclaimer-body">
          Bu tahmin, <strong>{prediction.as_of_date}</strong> piyasa kapanışı itibarıyla hesaplanan
          23 teknik göstergeye dayalı deneysel araştırma modelidir. <strong>Kesinlikle canlı yatırım,
          al/sat (Buy/Sell) tavsiyesi veya getiri garantisi değildir.</strong> Modelin yukarı yön skoru (%{confidencePct}),
          kalibre edilmiş kesin kazanma olasılığı olmayıp uncalibrated algoritmik bir sınıflandırıcı puanıdır.
        </p>
      </div>
    </div>
  );
};
