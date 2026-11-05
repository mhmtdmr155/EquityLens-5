import React from 'react';
import { FileText, Globe, Layers, Server } from 'lucide-react';

export const DataSources: React.FC = () => {
  return (
    <div className="datasources-card">
      <div className="datasources-header">
        <div className="datasources-icon-box">
          <Layers size={20} className="text-cyan-400" />
        </div>
        <div>
          <h3 className="datasources-title">Doğrulanmış Veri Kaynakları ve Veri Hattı (ETL)</h3>
          <span className="datasources-subtitle">
            EquityLens-5 Sisteminde Aktif Olarak Kullanılan Güvenilir Veri Katmanları
          </span>
        </div>
      </div>

      <div className="datasources-grid">
        {/* Alpaca */}
        <div className="ds-box">
          <div className="ds-box-top">
            <Globe size={18} className="text-blue-400" />
            <span className="ds-provider-name">Alpaca Market Data v2</span>
          </div>
          <p className="ds-desc">
            5 çekirdek şirket için 2.098 adet günlük (1Day) OHLCV fiyat barı. (AAPL, NVDA, MU, DELL: 500'er bar; SPCX: 98 bar).
          </p>
          <div className="ds-badge-row">
            <span className="ds-tag">IEX Ham Veri</span>
            <span className="ds-tag">Look-Ahead Korumalı</span>
          </div>
        </div>

        {/* SEC EDGAR */}
        <div className="ds-box">
          <div className="ds-box-top">
            <FileText size={18} className="text-emerald-400" />
            <span className="ds-provider-name">SEC EDGAR API</span>
          </div>
          <p className="ds-desc">
            Resmî 10-K ve 10-Q bildirimlerinden point-in-time uyumlu 8 temel finansal oran (Borç/Varlık, Cari Oran, Kârlılık vb.).
          </p>
          <div className="ds-badge-row">
            <span className="ds-tag">CompanyFacts JSON</span>
            <span className="ds-tag">Restatement Korumalı</span>
          </div>
        </div>

        {/* Market ETFs */}
        <div className="ds-box">
          <div className="ds-box-top">
            <Layers size={18} className="text-amber-400" />
            <span className="ds-provider-name">Piyasa / Sektör ETF'leri</span>
          </div>
          <p className="ds-desc">
            Geniş piyasa ve sektör korelasyonu için SPY (S&P 500), XLK (Teknoloji) ve SOXX (Yarı İletken) barları (FRED makro verisi değildir).
          </p>
          <div className="ds-badge-row">
            <span className="ds-tag">9 Piyasa/Sektör Özelliği</span>
            <span className="ds-tag">M2 Deney Katmanı</span>
          </div>
        </div>

        {/* SQLite Database */}
        <div className="ds-box">
          <div className="ds-box-top">
            <Server size={18} className="text-purple-400" />
            <span className="ds-provider-name">Yerel SQLite ACID DB</span>
          </div>
          <p className="ds-desc">
            Tüm şirket modelleri, fiyat serileri ve güvenlik kısıtlamaları (SpaceX Pre-IPO kısıtı) yerel ilişkisel veritabanında saklanır.
          </p>
          <div className="ds-badge-row">
            <span className="ds-tag">SQLAlchemy 2.0</span>
            <span className="ds-tag">Alembic Migrations</span>
          </div>
        </div>
      </div>
    </div>
  );
};
