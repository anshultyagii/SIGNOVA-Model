import React, { useEffect, useMemo, useState } from "react";
import { Header } from "./components/Header";
import { Disclaimer } from "./components/Disclaimer";
import { GlossInput } from "./components/GlossInput";
import { TranslationResultCard } from "./components/TranslationResultCard";
import { ErrorAlert } from "./components/ErrorAlert";
import { WebcamRecognition } from "./components/WebcamRecognition";
import { checkHealth, translateGlosses, ApiError } from "./services/api";
import type { ConnectionStatus, HealthResponse, TranslateResponse } from "./types/api";
import { parseGlossInput } from "./utils/parser";
import "./App.css";

type ActiveMode = "webcam" | "manual";

export const App: React.FC = () => {
  const [activeMode, setActiveMode] = useState<ActiveMode>("webcam");
  const [inputValue, setInputValue] = useState<string>("");
  const [translationResult, setTranslationResult] = useState<TranslateResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("checking");
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);

  // Parse input glosses dynamically
  const parsedGlosses = useMemo(() => parseGlossInput(inputValue), [inputValue]);

  // Check backend health for manual refresh
  const performHealthCheck = async () => {
    setIsCheckingHealth(true);
    try {
      const data = await checkHealth();
      setHealthData(data);
      setConnectionStatus("connected");
      setErrorMessage((prev) => (prev.includes("Cannot connect") ? "" : prev));
    } catch (err: unknown) {
      setConnectionStatus("disconnected");
      setHealthData(null);
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Unable to connect to SignSpeakAI backend.");
      }
    } finally {
      setIsCheckingHealth(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    async function initCheck() {
      setIsCheckingHealth(true);
      try {
        const data = await checkHealth();
        if (!ignore) {
          setHealthData(data);
          setConnectionStatus("connected");
        }
      } catch (err: unknown) {
        if (!ignore) {
          setConnectionStatus("disconnected");
          setHealthData(null);
          if (err instanceof ApiError) {
            setErrorMessage(err.message);
          }
        }
      } finally {
        if (!ignore) {
          setIsCheckingHealth(false);
        }
      }
    }
    initCheck();
    return () => {
      ignore = true;
    };
  }, []);

  const handleTranslate = async () => {
    if (parsedGlosses.length === 0) {
      setErrorMessage("Please enter at least one sign gloss to translate.");
      return;
    }

    setIsLoading(true);
    setErrorMessage("");

    try {
      const result = await translateGlosses(parsedGlosses);
      setTranslationResult(result);
      setConnectionStatus("connected");
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
        if (err.statusCode === 0) {
          setConnectionStatus("disconnected");
        }
      } else {
        setErrorMessage("An unexpected error occurred during translation.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setInputValue("");
    setTranslationResult(null);
    setErrorMessage("");
  };

  return (
    <div className="app-container">
      <Header
        status={connectionStatus}
        healthData={healthData}
        onRefresh={performHealthCheck}
        isChecking={isCheckingHealth}
      />

      <main className="main-content">
        <Disclaimer />

        <ErrorAlert
          message={errorMessage}
          onDismiss={() => setErrorMessage("")}
        />

        {/* Mode Selector Tabs */}
        <div className="mode-nav-tabs">
          <button
            type="button"
            className={`tab-btn ${activeMode === "webcam" ? "tab-active" : ""}`}
            onClick={() => setActiveMode("webcam")}
          >
            <span className="tab-icon">📹</span>
            <span>Live Camera Recognition</span>
            <span className="tab-badge">Real-Time</span>
          </button>

          <button
            type="button"
            className={`tab-btn ${activeMode === "manual" ? "tab-active" : ""}`}
            onClick={() => setActiveMode("manual")}
          >
            <span className="tab-icon">⌨️</span>
            <span>Manual Gloss Translation</span>
          </button>
        </div>

        {/* Active Mode View */}
        {activeMode === "webcam" ? (
          <WebcamRecognition />
        ) : (
          <div className="grid-layout">
            <GlossInput
              value={inputValue}
              onChange={(val) => {
                setInputValue(val);
                if (errorMessage && !errorMessage.includes("Cannot connect")) {
                  setErrorMessage("");
                }
              }}
              onTranslate={handleTranslate}
              onClear={handleClear}
              isLoading={isLoading}
              parsedGlosses={parsedGlosses}
            />

            <TranslationResultCard
              result={translationResult}
              isLoading={isLoading}
              hasInput={parsedGlosses.length > 0}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          SignSpeakAI ISL Translation & Recognition &bull; Backend at{" "}
          <code>{import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}</code>
        </p>
      </footer>
    </div>
  );
};

export default App;
