import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { PriceBarDetail, TimeHorizon } from '../types';
import { BarChart3, TrendingUp, Calendar } from 'lucide-react';

interface PriceChartProps {
  ticker: string;
  bars: PriceBarDetail[];
  timeHorizon: TimeHorizon;
  onHorizonChange: (horizon: TimeHorizon) => void;
  isLoading: boolean;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as PriceBarDetail;
    const formatCurrency = (val: number | string) =>
      new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
      }).format(Number(val));

    return (
      <div className="custom-chart-tooltip">
        <div className="tooltip-date">
          <Calendar size={13} className="mr-1" />
          {label}
        </div>
        <div className="tooltip-row highlight">
          <span>Kapanış (Close):</span>
          <span className="font-bold text-blue-400">{formatCurrency(data.close)}</span>
        </div>
        <div className="tooltip-row">
          <span>Açılış (Open):</span>
          <span>{formatCurrency(data.open)}</span>
        </div>
        <div className="tooltip-row">
          <span>En Yüksek (High):</span>
          <span className="text-green-400">{formatCurrency(data.high)}</span>
        </div>
        <div className="tooltip-row">
          <span>En Düşük (Low):</span>
          <span className="text-red-400">{formatCurrency(data.low)}</span>
        </div>
        <div className="tooltip-row">
          <span>Hacim (Volume):</span>
          <span>{new Intl.NumberFormat('en-US').format(Number(data.volume))}</span>
        </div>
      </div>
    );
  }
  return null;
};

export const PriceChart: React.FC<PriceChartProps> = ({
  ticker,
  bars,
  timeHorizon,
  onHorizonChange,
  isLoading,
}) => {
  // Sort bars chronologically (ascending date) for chart display
  const chartData = React.useMemo(() => {
    const sorted = [...bars].sort((a, b) => (a.session_date > b.session_date ? 1 : -1));
    return sorted.map((b) => ({
      ...b,
      closeNum: Number(b.close),
    }));
  }, [bars]);

  // Compute min/max for dynamic Y-axis scaling
  const minPrice = React.useMemo(() => {
    if (!chartData.length) return 0;
    const min = Math.min(...chartData.map((d) => d.closeNum));
    return Math.floor(min * 0.96);
  }, [chartData]);

  const maxPrice = React.useMemo(() => {
    if (!chartData.length) return 100;
    const max = Math.max(...chartData.map((d) => d.closeNum));
    return Math.ceil(max * 1.04);
  }, [chartData]);

  const formatPriceYAxis = (val: number) => `$${val}`;

  const formatShortDate = (dateStr: string) => {
    if (!dateStr) return '';
    const parts = dateStr.split('-');
    if (parts.length === 3) {
      return `${parts[1]}/${parts[2].slice(-2)}`;
    }
    return dateStr;
  };

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title-area">
          <div className="chart-icon-box">
            <BarChart3 size={20} className="text-blue-400" />
          </div>
          <div>
            <h3 className="chart-title">{ticker} Fiyat Grafiği (Tarihsel Kapanış)</h3>
            <span className="chart-subtitle">
              Alpaca Market Data v2 (IEX Ham Barlar) • Günlük OHLCV Serisi
            </span>
          </div>
        </div>

        {/* Horizon selector buttons */}
        <div className="horizon-btn-group">
          <button
            onClick={() => onHorizonChange(30)}
            className={`horizon-btn ${timeHorizon === 30 ? 'active' : ''}`}
            disabled={isLoading}
          >
            30 Seans
          </button>
          <button
            onClick={() => onHorizonChange(90)}
            className={`horizon-btn ${timeHorizon === 90 ? 'active' : ''}`}
            disabled={isLoading}
          >
            90 Seans
          </button>
          <button
            onClick={() => onHorizonChange(180)}
            className={`horizon-btn ${timeHorizon === 180 ? 'active' : ''}`}
            disabled={isLoading}
          >
            180 Seans
          </button>
          <button
            onClick={() => onHorizonChange(500)}
            className={`horizon-btn ${timeHorizon === 500 ? 'active' : ''}`}
            disabled={isLoading}
          >
            Tümü
          </button>
        </div>
      </div>

      {/* Chart Body */}
      <div className="chart-container">
        {isLoading && (
          <div className="chart-loading-overlay">
            <div className="spinner"></div>
            <span>Grafik yükleniyor...</span>
          </div>
        )}

        {chartData.length === 0 && !isLoading ? (
          <div className="chart-empty-state">
            <TrendingUp size={40} className="text-gray-500 mb-2" />
            <p>Bu dönem için kayıtlı fiyat barı bulunamadı.</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={chartData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="session_date"
                tickFormatter={formatShortDate}
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                tickLine={{ stroke: '#374151' }}
                minTickGap={28}
              />
              <YAxis
                domain={[minPrice, maxPrice]}
                tickFormatter={formatPriceYAxis}
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                tickLine={{ stroke: '#374151' }}
                orientation="right"
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="closeNum"
                stroke="#3b82f6"
                strokeWidth={2.2}
                fillOpacity={1}
                fill="url(#priceGradient)"
                name="Kapanış Fiyatı"
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="chart-footer">
        <span className="chart-note">
          {ticker === 'SPCX'
            ? 'Not: SPCX serisi 12.06.2026 halka arz tarihi sonrası 98 seans barını kapsar.'
            : `Görüntülenen: Son ${chartData.length} işlem günü seansı (Kronolojik sıra).`}
        </span>
        <span className="chart-date-range">
          {chartData.length > 0 &&
            `Aralık: ${chartData[0].session_date} — ${chartData[chartData.length - 1].session_date}`}
        </span>
      </div>
    </div>
  );
};

