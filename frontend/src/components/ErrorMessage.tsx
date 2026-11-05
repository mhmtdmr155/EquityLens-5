import React from 'react';
import { AlertCircle, RefreshCw, ServerOff } from 'lucide-react';

interface ErrorMessageProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  isBackendOffline?: boolean;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  title = 'Bir Hata Oluştu',
  message,
  onRetry,
  isBackendOffline = false,
}) => {
  return (
    <div className={`error-banner ${isBackendOffline ? 'backend-offline' : ''}`}>
      <div className="error-icon-col">
        {isBackendOffline ? (
          <ServerOff size={26} className="text-red-400" />
        ) : (
          <AlertCircle size={26} className="text-amber-400" />
        )}
      </div>
      <div className="error-content-col">
        <h4 className="error-title">{title}</h4>
        <p className="error-description">{message}</p>
        {isBackendOffline && (
          <p className="error-subnote">
            Lütfen FastAPI sunucusunun çalıştığından emin olun (Örn: <code>uvicorn backend.app.main:app --port 8000</code>).
          </p>
        )}
      </div>
      {onRetry && (
        <div className="error-action-col">
          <button onClick={onRetry} className="btn-retry">
            <RefreshCw size={15} className="mr-1.5" />
            Yeniden Dene
          </button>
        </div>
      )}
    </div>
  );
};

