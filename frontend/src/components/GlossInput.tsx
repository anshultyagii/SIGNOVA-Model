import React from "react";

const QUICK_INSERT_GLOSSES = [
  "I_Me",
  "You",
  "Want_Desire_Wish_For",
  "Water",
  "Food",
  "Help",
  "Where",
  "Milk",
  "Drink",
  "Eat",
  "Go",
  "One",
];

interface GlossInputProps {
  value: string;
  onChange: (val: string) => void;
  onTranslate: () => void;
  onClear: () => void;
  isLoading: boolean;
  parsedGlosses: string[];
}

export const GlossInput: React.FC<GlossInputProps> = ({
  value,
  onChange,
  onTranslate,
  onClear,
  isLoading,
  parsedGlosses,
}) => {
  const handleInsert = (gloss: string) => {
    const trimmed = value.trim();
    if (!trimmed) {
      onChange(gloss);
    } else {
      onChange(`${trimmed} ${gloss}`);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (parsedGlosses.length > 0 && !isLoading) {
        onTranslate();
      }
    }
  };

  return (
    <div className="card input-card">
      <div className="card-header">
        <h2 className="card-title">Gloss Sequence Input</h2>
        <span className="token-counter">
          {parsedGlosses.length} {parsedGlosses.length === 1 ? "token" : "tokens"} detected
        </span>
      </div>

      <div className="input-wrapper">
        <textarea
          className="gloss-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter sign glosses separated by spaces or commas (e.g., I_Me Want_Desire_Wish_For Water)..."
          rows={3}
          disabled={isLoading}
        />
      </div>

      {parsedGlosses.length > 0 && (
        <div className="parsed-chips-preview">
          <span className="chips-label">Sequence:</span>
          <div className="chips-container">
            {parsedGlosses.map((g, idx) => (
              <span key={`${g}-${idx}`} className="gloss-chip">
                {g}
                {idx < parsedGlosses.length - 1 && <span className="chip-arrow">→</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="quick-insert-section">
        <span className="quick-label">Quick Insert:</span>
        <div className="quick-buttons">
          {QUICK_INSERT_GLOSSES.map((gloss) => (
            <button
              key={gloss}
              type="button"
              className="btn-quick"
              onClick={() => handleInsert(gloss)}
              disabled={isLoading}
            >
              + {gloss}
            </button>
          ))}
        </div>
      </div>

      <div className="actions-bar">
        <button
          type="button"
          className="btn-clear"
          onClick={onClear}
          disabled={isLoading || (!value && parsedGlosses.length === 0)}
        >
          Clear
        </button>

        <button
          type="button"
          className="btn-translate"
          onClick={onTranslate}
          disabled={isLoading || parsedGlosses.length === 0}
        >
          {isLoading ? (
            <>
              <span className="spinner"></span> Translating...
            </>
          ) : (
            "Translate Sequence"
          )}
        </button>
      </div>
    </div>
  );
};
