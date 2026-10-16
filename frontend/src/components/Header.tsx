import React from "react";
import type { ConnectionStatus, HealthResponse } from "../types/api";

interface HeaderProps {
  status: ConnectionStatus;
  healthData: HealthResponse | null;
  onRefresh: () => void;
  isChecking: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  healthData,
  onRefresh,
  isChecking,
}) => {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-icon">🤟</div>
        <div>
          <h1 className="brand-title">SignSpeakAI</h1>
          <p className="brand-subtitle">Indian Sign Language to English Translation</p>
        </div>
      </div>

      <div className="header-status-area">
        <div className={`status-badge status-${status}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {status === "connected" && healthData
              ? `API Online (${healthData.supported_classes_count} Classes)`
              : status === "checking" || isChecking
              ? "Checking API..."
              : "API Disconnected"}
          </span>
        </div>

        <button
          className="btn-icon"
          onClick={onRefresh}
          disabled={isChecking}
          title="Refresh backend status"
          aria-label="Refresh backend status"
        >
          🔄
        </button>
      </div>
    </header>
  );
};
