import React from 'react';
import { Company } from '../types';
import { Rocket, TrendingUp, Laptop, Cpu, Layers } from 'lucide-react';

interface CompanySelectorProps {
  companies: Company[];
  selectedTicker: string;
  onSelectTicker: (ticker: string) => void;
  isLoading: boolean;
}

const COMPANY_ICONS: Record<string, React.ReactNode> = {
  AAPL: <Laptop size={20} className="text-gray-300" />,
  NVDA: <Cpu size={20} className="text-emerald-400" />,
  MU: <Layers size={20} className="text-blue-400" />,
  DELL: <Laptop size={20} className="text-cyan-400" />,
  SPCX: <Rocket size={20} className="text-amber-400" />,
};

const COMPANY_NICKNAMES: Record<string, string> = {
  AAPL: 'Apple Inc.',
  NVDA: 'NVIDIA Corp.',
  MU: 'Micron Tech.',
  DELL: 'Dell Tech.',
  SPCX: 'SpaceX (SPCX)',
};

export const CompanySelector: React.FC<CompanySelectorProps> = ({
  companies,
  selectedTicker,
  onSelectTicker,
  isLoading,
}) => {
  // If backend hasn't returned companies yet, fallback to the 5 standard tracked companies
  const displayList = companies.length > 0
    ? companies
    : [
        { id: 1, company_name: 'Apple Inc.', sec_cik: '0000320193', created_at: '', securities: [{ id: 1, ticker: 'AAPL', exchange: 'NASDAQ', valid_from: '2024-11-01', valid_to: null, company_id: 1 }] },
        { id: 2, company_name: 'NVIDIA Corporation', sec_cik: '0001045810', created_at: '', securities: [{ id: 2, ticker: 'NVDA', exchange: 'NASDAQ', valid_from: '2024-11-01', valid_to: null, company_id: 2 }] },
        { id: 3, company_name: 'Micron Technology, Inc.', sec_cik: '0000723125', created_at: '', securities: [{ id: 3, ticker: 'MU', exchange: 'NASDAQ', valid_from: '2024-11-01', valid_to: null, company_id: 3 }] },
        { id: 4, company_name: 'Dell Technologies Inc.', sec_cik: '0001571996', created_at: '', securities: [{ id: 4, ticker: 'DELL', exchange: 'NYSE', valid_from: '2024-11-01', valid_to: null, company_id: 4 }] },
        { id: 5, company_name: 'Space Exploration Technologies Corp.', sec_cik: '0001181412', created_at: '', securities: [{ id: 5, ticker: 'SPCX', exchange: 'NASDAQ', valid_from: '2026-06-12', valid_to: null, company_id: 5 }] },
      ];

  return (
    <section className="company-selector-section">
      <div className="section-header-compact">
        <h2 className="section-title-sm">Takip Edilen Şirketler (Core Portföy)</h2>
        <span className="section-subtitle-sm">5 Stratejik Varlık • Gün Sonu EOD Analizi</span>
      </div>

      <div className="company-grid">
        {displayList.map((comp) => {
          const ticker = comp.securities?.[0]?.ticker || 'AAPL';
          const exchange = comp.securities?.[0]?.exchange || 'NASDAQ';
          const isSelected = ticker === selectedTicker;

          return (
            <button
              key={ticker}
              onClick={() => onSelectTicker(ticker)}
              className={`company-card ${isSelected ? 'active' : ''}`}
              disabled={isLoading && isSelected}
            >
              <div className="company-card-top">
                <div className="company-icon-box">
                  {COMPANY_ICONS[ticker] || <TrendingUp size={20} />}
                </div>
                <span className="company-exchange-badge">{exchange}</span>
              </div>

              <div className="company-card-middle">
                <span className="company-ticker">{ticker}</span>
                <span className="company-name">{COMPANY_NICKNAMES[ticker] || comp.company_name}</span>
              </div>

              <div className="company-card-bottom">
                {ticker === 'SPCX' ? (
                  <span className="spcx-tag">Post-IPO (98 Bar)</span>
                ) : (
                  <span className="history-tag">500 Bar (2 Yıl)</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
};

