import React from 'react';
import { PriceBarDetail, Company } from '../types';
import { ArrowUpRight, ArrowDownRight, Minus, Calendar, Building2 } from 'lucide-react';

interface StockSummaryProps {
  ticker: string;
  companyInfo?: Company;
  latestBar?: PriceBarDetail;
  previousBar?: PriceBarDetail;
}

export const StockSummary: React.FC<StockSummaryProps> = ({
  ticker,
  companyInfo,
  latestBar,
  previousBar,
}) => {
  const currentClose = latestBar ? Number(latestBar.close) : 0;
  const prevClose = previousBar ? Number(previousBar.close) : currentClose;
  const priceChange = currentClose - prevClose;
  const pctChange = prevClose !== 0 ? (priceChange / prevClose) * 100 : 0;

  const isPositive = priceChange > 0.0001;
  const isNegative = priceChange < -0.0001;

  const formatCurrency = (val: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(val);
  };

  const formatNumber = (val: number) => {
    return new Intl.NumberFormat('en-US').format(val);
  };

  return (
    <div className="stock-summary-card">
      <div className="stock-summary-header">
        <div className="stock-title-area">
          <div className="ticker-badge-lg">{ticker}</div>
          <div>
            <h1 className="company-fullname">
              {companyInfo?.company_name || `${ticker} Corporate`}
            </h1>
            <div className="company-meta-tags">
              <span className="meta-tag">
                <Building2 size={13} className="mr-1" />
                CIK: {companyInfo?.sec_cik || 'N/A'}
              </span>
              <span className="meta-tag">
                <Calendar size={13} className="mr-1" />
                Tarih: {latestBar?.session_date || '2026-10-30'}
              </span>
              {ticker === 'SPCX' && (
                <span className="meta-tag spcx-highlight">
                  Özel Statü: 12.06.2026 Sonrası Halka Arz (Post-IPO)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Current Price Display */}
        <div className="price-display-area">
          <div className="price-main">
            {latestBar ? formatCurrency(currentClose) : '---'}
          </div>
          <div className={`price-change-pill ${isPositive ? 'positive' : isNegative ? 'negative' : 'neutral'}`}>
            {isPositive && <ArrowUpRight size={18} />}
            {isNegative && <ArrowDownRight size={18} />}
            {!isPositive && !isNegative && <Minus size={18} />}
            <span>
              {isPositive ? '+' : ''}
              {formatCurrency(priceChange)} ({isPositive ? '+' : ''}
              {pctChange.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      {/* OHLCV Metrics Bar */}
      {latestBar && (
        <div className="ohlcv-grid">
          <div className="ohlcv-item">
            <span className="ohlcv-label">Açılış (Open)</span>
            <span className="ohlcv-value">{formatCurrency(Number(latestBar.open))}</span>
          </div>
          <div className="ohlcv-item">
            <span className="ohlcv-label">En Yüksek (High)</span>
            <span className="ohlcv-value text-green-400">{formatCurrency(Number(latestBar.high))}</span>
          </div>
          <div className="ohlcv-item">
            <span className="ohlcv-label">En Düşük (Low)</span>
            <span className="ohlcv-value text-red-400">{formatCurrency(Number(latestBar.low))}</span>
          </div>
          <div className="ohlcv-item">
            <span className="ohlcv-label">Kapanış (Close)</span>
            <span className="ohlcv-value font-semibold">{formatCurrency(Number(latestBar.close))}</span>
          </div>
          <div className="ohlcv-item">
            <span className="ohlcv-label">İşlem Hacmi (Volume)</span>
            <span className="ohlcv-value">{formatNumber(Number(latestBar.volume))}</span>
          </div>
        </div>
      )}
    </div>
  );
};
