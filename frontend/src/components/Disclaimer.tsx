import React from "react";

export const Disclaimer: React.FC = () => {
  return (
    <div className="disclaimer-banner">
      <div className="disclaimer-icon">⚠️</div>
      <div className="disclaimer-content">
        <strong>Rule-based ISL-to-English prototype; not a linguistically validated translator.</strong>
        <p>
          Indian Sign Language (ISL) possesses complex grammatical structures, spatial referencing,
          and facial/non-manual markers. This interface translates ordered gloss tokens using
          deterministic grammar templates and safe fallbacks.
        </p>
      </div>
    </div>
  );
};
