import React from "react";

interface ErrorAlertProps {
  message: string;
  onDismiss: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onDismiss }) => {
  if (!message) return null;

  return (
    <div className="error-alert" role="alert">
      <div className="error-content">
        <span className="error-icon">❌</span>
        <span className="error-text">{message}</span>
      </div>
      <button
        className="btn-dismiss"
        onClick={onDismiss}
        title="Dismiss error"
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  );
};
