import React from 'react';
import { Database, RefreshCw, Cpu } from 'lucide-react';
import { SystemHealth } from '../types';

interface HeaderProps {
  health: SystemHealth;
  onRefresh: () => void;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, onRefresh, isLoading }) => {
  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <div className="brand-icon">
            <Cpu className="icon-pulse text-blue-400" size={28} />
          </div>
          <div>
            <div className="brand-title">
              EQUITYLENS-5
              <span className="badge-tag">v1.2 Demo</span>
            </div>
            <p className="brand-subtitle">AI-Powered Financial Intelligence • Makine Öğrenmesi Karar Desteği</p>
          </div>
        </div>

        <div className="navbar-status-group">
          {/* Market As-of Date */}
          <div className="status-pill">
            <span className="pill-dot bg-amber-400"></span>
            <span className="pill-label">Piyasa Verisi:</span>
            <span className="pill-value">30.10.2026 Kapanış</span>
          </div>

          {/* Backend Status */}
          <div className={`status-pill ${health.isOnline ? 'border-green-500/30' : 'border-red-500/30'}`}>
            <span className={`pill-dot ${health.isOnline ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`}></span>
            <span className="pill-label">Backend:</span>
            <span className="pill-value">
              {health.isOnline ? `Çevrimiçi (${health.dialect.toUpperCase()})` : 'Bağlantı Kesildi'}
            </span>
          </div>

          {/* DB Status */}
          <div className="status-pill hidden-sm">
            <Database size={14} className="text-gray-400 mr-1" />
            <span className="pill-label">Veritabanı:</span>
            <span className="pill-value">{health.dbConnected ? 'SQLite (Aktif)' : 'Erişilemiyor'}</span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="btn-refresh"
            title="Verileri ve Backend Durumunu Yenile"
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
            <span className="hidden-sm">Yenile</span>
          </button>
        </div>
      </div>
    </header>
  );
};
