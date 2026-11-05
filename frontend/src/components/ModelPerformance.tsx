import React from 'react';
import { ExperimentMetricsResponse } from '../types';
import { Award, CheckCircle2, FlaskConical, ShieldCheck } from 'lucide-react';

interface ModelPerformanceProps {
  metrics: ExperimentMetricsResponse | null;
  isLoading?: boolean;
}

export const ModelPerformance: React.FC<ModelPerformanceProps> = ({ metrics }) => {
  // If backend hasn't loaded metrics yet, fallback to confirmed experimental results
  const retro = metrics?.retrospective_models;
  const cv = metrics?.expanding_cv_averages;

  const b0a_acc = retro?.['B0a_majority']?.accuracy ?? 0.5718;
  const b0a_bacc = cv?.['B0a']?.balanced_accuracy ?? 0.5000;

  const b0b_acc = retro?.['B0b_momentum']?.accuracy ?? 0.4259;
  const b0b_bacc = cv?.['B0b']?.balanced_accuracy ?? 0.4392;

  const m0_acc = retro?.['M0_price_only']?.accuracy ?? 0.5347;
  const m0_bacc = cv?.['M0']?.balanced_accuracy ?? 0.5053;

  const m1_acc = retro?.['M1_price_sec']?.accuracy ?? 0.5463;
  const m1_bacc = cv?.['M1']?.balanced_accuracy ?? 0.5041;

  const m2_acc = retro?.['M2_price_sec_market']?.accuracy ?? 0.5718;
  const m2_bacc = cv?.['M2']?.balanced_accuracy ?? 0.4896;

  const formatPct = (val: number) => `%${(val * 100).toFixed(2)}`;

  return (
    <div className="performance-card">
      <div className="perf-header">
        <div className="perf-title-area">
          <div className="perf-icon-box">
            <FlaskConical size={20} className="text-purple-400" />
          </div>
          <div>
            <h3 className="perf-title">Akademik Model Karşılaştırması ve Deney Sonuçları</h3>
            <span className="perf-subtitle">
              Kronolojik Test ve 3 Dönemli Genişleyen Pencere Çapraz Doğrulaması (Expanding Window CV)
            </span>
          </div>
        </div>

        <div className="perf-badge-active">
          <ShieldCheck size={14} className="text-emerald-400 mr-1" />
          <span>Aktif Model: M0 (23 Teknik Özellik)</span>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="table-responsive">
        <table className="perf-table">
          <thead>
            <tr>
              <th>Model Yapılandırması</th>
              <th>Özellik Kapsamı</th>
              <th>Tekil Test Doğruluğu (Test Accuracy)</th>
              <th>Çapraz Doğrulama (Mean Balanced Acc)</th>
              <th>Model Durumu</th>
            </tr>
          </thead>
          <tbody>
            {/* B0a Majority */}
            <tr>
              <td className="font-semibold text-gray-300">
                B0a (Majority Baseline)
              </td>
              <td>Tarihsel Çoğunluk Sınıfı (Statik)</td>
              <td className="font-mono">{formatPct(b0a_acc)}</td>
              <td className="font-mono text-gray-400">{formatPct(b0a_bacc)}</td>
              <td>
                <span className="status-tag tag-neutral">Referans Taban</span>
              </td>
            </tr>

            {/* B0b Momentum */}
            <tr>
              <td className="font-semibold text-gray-300">
                B0b (Momentum Baseline)
              </td>
              <td>5 Günlük Basit Fiyat Yönü</td>
              <td className="font-mono">{formatPct(b0b_acc)}</td>
              <td className="font-mono text-red-400">{formatPct(b0b_bacc)}</td>
              <td>
                <span className="status-tag tag-neutral">Zayıf Taban</span>
              </td>
            </tr>

            {/* M0 Price Only */}
            <tr className="highlight-row">
              <td className="font-semibold text-blue-400 flex items-center">
                <CheckCircle2 size={16} className="text-blue-400 mr-1.5 inline" />
                M0 (Fiyat & Teknik Model)
              </td>
              <td>23 Teknik İndikatör (RSI, MACD, Bollinger, Volatilite, Hacim)</td>
              <td className="font-mono">{formatPct(m0_acc)}</td>
              <td className="font-mono text-emerald-400 font-bold">
                {formatPct(m0_bacc)} (En Yüksek CV)
              </td>
              <td>
                <span className="status-tag tag-production">ÜRETİM MODELİ</span>
              </td>
            </tr>

            {/* M1 SEC */}
            <tr>
              <td className="font-semibold text-gray-300">
                M1 (Fiyat + SEC Temel)
              </td>
              <td>23 Teknik + 8 SEC Finansal Oran (Point-in-Time)</td>
              <td className="font-mono">{formatPct(m1_acc)}</td>
              <td className="font-mono">{formatPct(m1_bacc)}</td>
              <td>
                <span className="status-tag tag-experiment">Araştırma / Deney</span>
              </td>
            </tr>

            {/* M2 Market & SEC */}
            <tr>
              <td className="font-semibold text-gray-300">
                M2 (Fiyat + SEC + Piyasa/ETF)
              </td>
              <td>23 Teknik + 8 SEC + 9 Piyasa ETF (SPY, XLK, SOXX)</td>
              <td className="font-mono text-emerald-400">{formatPct(m2_acc)} (Retrospektif)</td>
              <td className="font-mono text-amber-400">{formatPct(m2_bacc)} (Overfitting)</td>
              <td>
                <span className="status-tag tag-experiment">Araştırma / Deney</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Honest Scientific Finding Callout */}
      <div className="perf-finding-box">
        <div className="finding-header">
          <Award size={16} className="text-purple-400 mr-2" />
          <h4 className="font-semibold text-purple-200">Bilimsel Değerlendirme ve Karar Gerekçesi</h4>
        </div>
        <p className="finding-text">
          Çok dönemli 3-fold genişleyen pencere (expanding window) çapraz doğrulamasında;
          <strong> M0 modeli (%50,53)</strong> en dengeli genelleme performansını sergilemiştir.
          M2 modeli tekil test periyodunda %57,18 doğruluk (%54,21 dengeli doğruluk) elde etmesine karşın,
          çapraz doğrulamada ortalama %48,96 seviyesine gerileyerek rejim değişikliklerine karşı kırılganlık
          göstermiştir. Bilimsel dürüstlük gereği, aşırı öğrenme (overfitting) şüphesi bulunan karmaşık modeller
          yerine, kararlı performans gösteren <strong>M0 (23 teknik özellikli HistGradientBoosting)</strong> modeli
          üretim çıkarımında tutulmuştur.
        </p>
      </div>
    </div>
  );
};
