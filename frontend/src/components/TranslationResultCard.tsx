import React from "react";
import type { TranslateResponse } from "../types/api";

interface TranslationResultCardProps {
  result: TranslateResponse | null;
  isLoading: boolean;
  hasInput: boolean;
}

export const TranslationResultCard: React.FC<TranslationResultCardProps> = ({
  result,
  isLoading,
  hasInput,
}) => {
  return (
    <div className="card result-card">
      <div className="card-header">
        <h2 className="card-title">English Translation Output</h2>
        {result && (
          <div className="status-badges-group">
            {result.is_fallback ? (
              <span className="badge badge-fallback" title="No grammar template matched; safe literal fallback used">
                Safe Fallback
              </span>
            ) : (
              <span className="badge badge-rule" title={result.matched_rule || "Rule template matched"}>
                Grammar Rule Matched
              </span>
            )}
          </div>
        )}
      </div>

      <div className="result-body">
        {isLoading ? (
          <div className="loading-state">
            <span className="large-spinner"></span>
            <p>Processing gloss sequence with SignSpeakAI translator...</p>
          </div>
        ) : result ? (
          <div className="output-container">
            <div className="translation-highlight">
              <span className="output-label">English Translation:</span>
              <p className="translation-text">
                {result.translation ? `"${result.translation}"` : <span className="empty-val">(Empty sequence)</span>}
              </p>
            </div>

            <div className="metadata-section">
              <div className="meta-row">
                <span className="meta-label">Normalized Glosses:</span>
                <div className="normalized-chips">
                  {result.normalized_glosses && result.normalized_glosses.length > 0 ? (
                    result.normalized_glosses.map((item, idx) => (
                      <span key={`${item}-${idx}`} className="norm-chip">
                        {item}
                        {idx < result.normalized_glosses.length - 1 && (
                          <span className="chip-sep">→</span>
                        )}
                      </span>
                    ))
                  ) : (
                    <span className="empty-meta">None</span>
                  )}
                </div>
              </div>

              {result.matched_rule && (
                <div className="meta-row">
                  <span className="meta-label">Matched Template:</span>
                  <code className="rule-code">{result.matched_rule}</code>
                </div>
              )}

              {result.is_fallback && (
                <div className="fallback-explanation">
                  ℹ️ <strong>Safe Fallback Notice:</strong> No rule template exists for this specific combination.
                  SignSpeakAI translated mapped words directly while preserving original glosses without inventing unsaid prepositions or auxiliary verbs.
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h3>No translation yet</h3>
            <p>
              {hasInput
                ? "Click 'Translate Sequence' above to generate English translation."
                : "Enter glosses or use the quick insert buttons to start."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
