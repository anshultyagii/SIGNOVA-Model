import React, { useCallback, useEffect, useRef, useState } from "react";
import { getWebSocketRecognizeUrl } from "../services/api";
import type {
  ActiveCandidate,
  FrameResult,
  InitAckMessage,
  LandmarkPoint,
  PauseAckMessage,
  ResumeAckMessage,
  UndoAckMessage,
} from "../types/api";

// Hand landmark bone connections (MediaPipe 21 landmarks)
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

// Upper-body pose landmark connections
const UPPER_POSE_CONNECTIONS: [number, number][] = [
  // Shoulders
  [11, 12],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
];

// Vocabulary gloss stem mapping for identifying contextually inferred words
const GLOSS_STEMS: Record<string, string[]> = {
  "I_Me": ["i", "me", "my", "myself"],
  "You": ["you", "your", "yours", "yourself"],
  "We": ["we", "us", "our", "ours"],
  "My": ["my", "mine"],
  "Want_Desire_Wish_For": ["want", "wants", "wanted", "desire", "wish"],
  "Water": ["water"],
  "Food": ["food"],
  "Milk": ["milk"],
  "Drink": ["drink", "drinks", "drinking", "drank"],
  "Eat": ["eat", "eats", "eating", "ate"],
  "Give": ["give", "gives", "giving", "gave"],
  "Go": ["go", "goes", "going", "went"],
  "Help": ["help", "helps", "helping", "helped"],
  "What": ["what"],
  "Where": ["where"],
  "One": ["one", "1"],
  "Two": ["two", "2"],
  "Three": ["three", "3"],
  "Four": ["four", "4"],
  "Five": ["five", "5"],
  "Six": ["six", "6"],
  "Seven": ["seven", "7"],
  "Eight": ["eight", "8"],
  "Nine": ["nine", "9"],
  "Ten": ["ten", "10"],
};

function getInferredWords(sentence: string | null | undefined, glosses: string[]): string[] {
  if (!sentence || !glosses || glosses.length === 0) return [];
  const known = new Set<string>();
  for (const g of glosses) {
    const stems = GLOSS_STEMS[g];
    if (stems) {
      stems.forEach((s) => known.add(s.toLowerCase()));
    } else {
      known.add(g.toLowerCase());
    }
  }
  const words = sentence.match(/\b[A-Za-z0-9']+\b/g) || [];
  const inferred: string[] = [];
  for (const w of words) {
    const lw = w.toLowerCase();
    if (!known.has(lw) && !inferred.includes(lw)) {
      inferred.push(lw);
    }
  }
  return inferred;
}

export const WebcamRecognition: React.FC = () => {
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");
  const [cameraError, setCameraError] = useState<string>("");
  const [wsErrorMessage, setWsErrorMessage] = useState<string>("");
  const [galleryStatus, setGalleryStatus] = useState<{ loaded: boolean; count: number } | null>(null);

  // Recognition state from backend
  const [segState, setSegState] = useState<"IDLE" | "ACTIVE" | "COOLDOWN" | "PAUSED" | string>("IDLE");
  const [handsDetected, setHandsDetected] = useState<boolean>(false);
  const [inRestZone, setInRestZone] = useState<boolean>(false);
  const [motionEnergy, setMotionEnergy] = useState<number>(0);
  const [gestureFrames, setGestureFrames] = useState<number>(0);
  const [sessionFrameCount, setSessionFrameCount] = useState<number>(0);
  const [latestPrediction, setLatestPrediction] = useState<FrameResult["latest_prediction"]>(null);
  const [currentGlosses, setCurrentGlosses] = useState<string[]>([]);
  const [currentTranslation, setCurrentTranslation] = useState<string | null>(null);
  const [isFinalized, setIsFinalized] = useState<boolean>(false);
  const [finalizedSequence, setFinalizedSequence] = useState<string[] | null>(null);
  const [finalizedTranslation, setFinalizedTranslation] = useState<string | null>(null);

  // Persistent completed sentence vs active in-progress sequence
  const [completedSentence, setCompletedSentence] = useState<{
    glosses: string[];
    translation: string | null;
  } | null>(null);
  const [inProgressGlosses, setInProgressGlosses] = useState<string[]>([]);
  const [inProgressTranslation, setInProgressTranslation] = useState<string | null>(null);
  const [handSource, setHandSource] = useState<string>("none");
  const [reachedMatcher, setReachedMatcher] = useState<boolean>(false);
  const [rejectionReason, setRejectionReason] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [videoAspectRatio, setVideoAspectRatio] = useState<number>(4 / 3);

  // Recognition control & Hotkeys (Spacebar, P, Backspace)
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isHotkeysEnabled, setIsHotkeysEnabled] = useState<boolean>(true);
  const [activeCandidate, setActiveCandidate] = useState<ActiveCandidate | null>(null);

  // Diagnostic metrics
  const [missedHandsCount, setMissedHandsCount] = useState<number>(0);
  const [rejectedSegmentsCount, setRejectedSegmentsCount] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);

  // Inferred words & uncertainty
  const [inferredWords, setInferredWords] = useState<string[]>([]);
  const [uncertaintyFlag, setUncertaintyFlag] = useState<boolean>(false);

  // Element refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Keyboard shortcut tracking refs
  const isPausedRef = useRef<boolean>(false);
  const isHotkeysEnabledRef = useRef<boolean>(true);

  // Networking refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const isAwaitingFrameRef = useRef<boolean>(false);
  const frameTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const isManualStopRef = useRef<boolean>(false);
  const isCameraActiveRef = useRef<boolean>(false);
  const connectWebSocketRef = useRef<() => void>(() => {});
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);

  const handleVideoMetadata = () => {
    if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
      setVideoAspectRatio(videoRef.current.videoWidth / videoRef.current.videoHeight);
    }
  };

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    isHotkeysEnabledRef.current = isHotkeysEnabled;
  }, [isHotkeysEnabled]);

  const togglePause = useCallback(() => {
    if (!isCameraActiveRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    const nextPaused = !isPausedRef.current;
    isPausedRef.current = nextPaused;
    setIsPaused(nextPaused);
    try {
      wsRef.current.send(JSON.stringify({ type: nextPaused ? "pause" : "resume" }));
    } catch (err) {
      console.error("Failed to send pause/resume message:", err);
    }
  }, []);

  const handleUndoLastSign = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    try {
      wsRef.current.send(JSON.stringify({ type: "undo" }));
    } catch (err) {
      console.error("Failed to send undo message:", err);
    }
  }, []);

  // Keyboard shortcut listener: Space to Pause/Resume, P to toggle hotkeys, Backspace to undo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      // 1. Never intercept when typing in form inputs, textareas, or contentEditable fields
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }

      // 2. 'P' or 'p' key toggles hotkeys feature on / off
      if (e.key === "p" || e.key === "P") {
        e.preventDefault();
        setIsHotkeysEnabled((prev) => !prev);
        return;
      }

      // 3. If hotkeys are disabled, Space and Backspace do nothing
      if (!isHotkeysEnabledRef.current) {
        return;
      }

      // 4. Spacebar toggles start / stop (pause / resume) recognition
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault(); // Prevent page scroll and button activation
        if (isCameraActiveRef.current) {
          togglePause();
        }
        return;
      }

      // 5. Backspace removes the last accepted sign
      if (e.code === "Backspace" || e.key === "Backspace") {
        e.preventDefault(); // Prevent browser back navigation
        handleUndoLastSign();
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [togglePause, handleUndoLastSign]);

  // ── Landmark Overlay Rendering ────────────────────────────────────────────

  const drawLandmarks = useCallback((landmarks: FrameResult["landmarks"], width: number, height: number) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw upper-body pose skeleton
    if (landmarks.pose && landmarks.pose.length >= 17) {
      ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
      ctx.lineWidth = 3;
      UPPER_POSE_CONNECTIONS.forEach(([start, end]) => {
        const p1 = landmarks.pose![start];
        const p2 = landmarks.pose![end];
        if (p1 && p2 && p1.x > 0 && p2.x > 0) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      });

      // Draw pose joints
      ctx.fillStyle = "#10b981";
      [11, 12, 13, 14, 15, 16].forEach((idx) => {
        const p = landmarks.pose![idx];
        if (p && p.x > 0) {
          ctx.beginPath();
          ctx.arc(p.x * width, p.y * height, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    }

    // Helper to draw a single hand
    const renderHand = (points: LandmarkPoint[] | null | undefined, boneColor: string, jointColor: string) => {
      if (!points || points.length !== 21) return;

      // Draw bones
      ctx.strokeStyle = boneColor;
      ctx.lineWidth = 2.5;
      HAND_CONNECTIONS.forEach(([start, end]) => {
        const p1 = points[start];
        const p2 = points[end];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      });

      // Draw joints
      ctx.fillStyle = jointColor;
      points.forEach((p, idx) => {
        ctx.beginPath();
        const radius = idx === 0 || idx % 4 === 0 ? 4 : 2.5;
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        ctx.fill();
      });
    };

    // Left hand (cyan)
    renderHand(landmarks.left_hand, "rgba(56, 189, 248, 0.8)", "#38bdf8");

    // Right hand (purple)
    renderHand(landmarks.right_hand, "rgba(168, 85, 247, 0.8)", "#c084fc");
  }, []);

  // ── Frame Transmission Loop ───────────────────────────────────────────────

  const sendNextFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    if (isAwaitingFrameRef.current) {
      return;
    }

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    // Setup offscreen capture canvas
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    const canvas = captureCanvasRef.current;
    const targetWidth = 640;
    const targetHeight = Math.round((video.videoHeight / video.videoWidth) * targetWidth);

    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    // Draw UNFLIPPED frame to canvas for MediaPipe extraction
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Compress to JPEG data URL with high fidelity for fine finger keypoints
    const jpegDataUrl = canvas.toDataURL("image/jpeg", 0.75);

    isAwaitingFrameRef.current = true;
    try {
      wsRef.current.send(
        JSON.stringify({
          type: "frame",
          data: jpegDataUrl,
          timestamp: Date.now() / 1000,
        })
      );
    } catch {
      isAwaitingFrameRef.current = false;
    }
  }, []);

  // ── WebSocket Initialization & Reconnection ───────────────────────────────

  const connectWebSocket = useCallback(() => {
    if (isManualStopRef.current) return;

    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // Ignore close error on stale socket
      }
      wsRef.current = null;
    }

    const url = getWebSocketRecognizeUrl();
    setWsStatus("connecting");
    setWsErrorMessage("");

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
      wsRef.current = ws;
    } catch (err: unknown) {
      setWsStatus("error");
      setWsErrorMessage(`Failed to initiate WebSocket to ${url}: ${String(err)}`);
      return;
    }

    ws.onopen = () => {
      setWsStatus("connected");
      setWsErrorMessage("");
      reconnectAttemptsRef.current = 0;
      isAwaitingFrameRef.current = false;
      try {
        ws.send(JSON.stringify({ type: "ping" }));
      } catch {
        // Ignore
      }
    };

    ws.onmessage = (event) => {
      isAwaitingFrameRef.current = false;
      try {
        const data = JSON.parse(event.data);

        if (data.type === "init_ack") {
          const initAck = data as InitAckMessage;
          setGalleryStatus({
            loaded: Boolean(initAck.gallery_loaded),
            count: Number(initAck.enrolled_classes || 0),
          });
          if (!initAck.gallery_loaded) {
            setWsErrorMessage("Warning: Exemplar gallery is not loaded on backend.");
          }
        } else if (data.type === "pause_ack") {
          const pauseMsg = data as PauseAckMessage;
          setIsPaused(pauseMsg.is_paused);
          isPausedRef.current = pauseMsg.is_paused;
          setSegState("PAUSED");
          setActiveCandidate(null);
        } else if (data.type === "resume_ack") {
          const resumeMsg = data as ResumeAckMessage;
          setIsPaused(resumeMsg.is_paused);
          isPausedRef.current = resumeMsg.is_paused;
          setSegState("IDLE");
        } else if (data.type === "undo_ack") {
          const undoMsg = data as UndoAckMessage;
          const updatedGlosses = undoMsg.current_glosses || [];
          const updatedTrans = undoMsg.current_translation || null;
          setCurrentGlosses(updatedGlosses);
          setCurrentTranslation(updatedTrans);
          setIsFinalized(Boolean(undoMsg.is_finalized));

          if (undoMsg.in_progress_sequence !== undefined) {
            setInProgressGlosses(undoMsg.in_progress_sequence);
            setInProgressTranslation(undoMsg.in_progress_translation || null);
          } else {
            setInProgressGlosses(updatedGlosses);
            setInProgressTranslation(updatedTrans);
          }

          if (undoMsg.completed_sequence && undoMsg.completed_sequence.length > 0) {
            setCompletedSentence({
              glosses: undoMsg.completed_sequence,
              translation: undoMsg.completed_translation || null,
            });
          } else if (updatedGlosses.length === 0) {
            setCompletedSentence(null);
            setFinalizedSequence(null);
            setFinalizedTranslation(null);
          }

          if (updatedGlosses.length === 0 && (!undoMsg.completed_sequence || undoMsg.completed_sequence.length === 0)) {
            setFinalizedSequence(null);
            setFinalizedTranslation(null);
            setInferredWords([]);
            setUncertaintyFlag(false);
          } else {
            setInferredWords(getInferredWords(updatedTrans || undoMsg.completed_translation, updatedGlosses.length > 0 ? updatedGlosses : (undoMsg.completed_sequence || [])));
          }
        } else if (data.type === "frame_result") {
          const res = data as FrameResult;
          setSegState(res.state);
          setHandsDetected(res.hands_detected);
          if (res.is_paused !== undefined) {
            setIsPaused(res.is_paused);
            isPausedRef.current = res.is_paused;
          }
          if (res.active_candidate !== undefined) {
            setActiveCandidate(res.active_candidate);
          }
          if (res.diagnostics) {
            if (res.diagnostics.latency_ms !== undefined) {
              setLatencyMs(res.diagnostics.latency_ms);
            }
            if (res.diagnostics.missed_hands_count !== undefined) {
              setMissedHandsCount(res.diagnostics.missed_hands_count);
            }
            if (res.diagnostics.rejected_segments_count !== undefined) {
              setRejectedSegmentsCount(res.diagnostics.rejected_segments_count);
            }
          }
          if (res.hand_source) {
            setHandSource(res.hand_source);
          }
          if (res.reached_matcher !== undefined) {
            setReachedMatcher(res.reached_matcher);
          }
          if (res.rejection_reason !== undefined) {
            setRejectionReason(res.rejection_reason);
          }
          if (res.in_rest_zone !== undefined) {
            setInRestZone(res.in_rest_zone);
          }
          if (res.motion_energy !== undefined) {
            setMotionEnergy(res.motion_energy);
          }
          setGestureFrames(res.gesture_frames);

          if (res.frame_count !== undefined) {
            setSessionFrameCount(res.frame_count);
          }

          if (res.latest_prediction) {
            setLatestPrediction(res.latest_prediction);
          }

          // Update persistent completed sentence when finalized
          if (res.completed_sequence && res.completed_sequence.length > 0) {
            setCompletedSentence({
              glosses: res.completed_sequence,
              translation: res.completed_translation || res.finalized_translation || null,
            });
          } else if (res.finalized_sequence && res.finalized_sequence.length > 0) {
            setCompletedSentence({
              glosses: res.finalized_sequence,
              translation: res.finalized_translation || null,
            });
          }

          // Update active in-progress sequence
          if (res.in_progress_sequence !== undefined) {
            setInProgressGlosses(res.in_progress_sequence);
            setInProgressTranslation(res.in_progress_translation || null);
          }

          if (res.current_glosses) {
            setCurrentGlosses(res.current_glosses);
          }
          if (res.current_translation !== undefined) {
            setCurrentTranslation(res.current_translation);
            setInferredWords(getInferredWords(res.current_translation, res.current_glosses || currentGlosses));
          }
          if (res.is_finalized !== undefined) {
            setIsFinalized(res.is_finalized);
          }
          if (res.finalized_sequence && res.finalized_sequence.length > 0) {
            setFinalizedSequence(res.finalized_sequence);
            setFinalizedTranslation(res.finalized_translation || null);
            setInferredWords(getInferredWords(res.finalized_translation, res.finalized_sequence));
          }

          // Render landmarks on overlay canvas with 1:1 geometry match
          if (videoRef.current && overlayCanvasRef.current && res.landmarks) {
            const w = videoRef.current.clientWidth || 640;
            const h = videoRef.current.clientHeight || 480;
            if (overlayCanvasRef.current.width !== w || overlayCanvasRef.current.height !== h) {
              overlayCanvasRef.current.width = w;
              overlayCanvasRef.current.height = h;
            }
            drawLandmarks(res.landmarks, w, h);
          }

          // Calculate streaming FPS
          frameCountRef.current += 1;
          const now = performance.now();
          if (lastFrameTimeRef.current === 0) {
            lastFrameTimeRef.current = now;
          } else if (now - lastFrameTimeRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastFrameTimeRef.current = now;
          }
        } else if (data.type === "reset_ack") {
          setCompletedSentence(null);
          setInProgressGlosses([]);
          setInProgressTranslation(null);
          setCurrentGlosses([]);
          setCurrentTranslation(null);
          setIsFinalized(false);
          setLatestPrediction(null);
          setFinalizedSequence(null);
          setFinalizedTranslation(null);
          setActiveCandidate(null);
          setInferredWords([]);
          setUncertaintyFlag(false);
        } else if (data.type === "error") {
          console.warn("Backend recognition error:", data.message);
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket connection error:", err);
      setWsStatus("error");
      setWsErrorMessage(
        `WebSocket connection to ${url} failed. Ensure the FastAPI backend is running on port 8000.`
      );
    };

    ws.onclose = (event) => {
      isAwaitingFrameRef.current = false;
      if (isManualStopRef.current) {
        setWsStatus("disconnected");
        setWsErrorMessage("");
        return;
      }

      setWsStatus("error");
      const reason = event.reason ? `: ${event.reason}` : "";
      setWsErrorMessage(
        `Connection closed (code ${event.code}${reason}). Check if FastAPI backend is running.`
      );

      if (isCameraActiveRef.current && reconnectAttemptsRef.current < 5) {
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(1000 * reconnectAttemptsRef.current, 4000);
        console.log(`Reconnecting to WebSocket (attempt ${reconnectAttemptsRef.current}/5 in ${delay}ms)...`);
        reconnectTimerRef.current = window.setTimeout(() => {
          if (isCameraActiveRef.current && !isManualStopRef.current) {
            connectWebSocketRef.current();
          }
        }, delay);
      }
    };
  }, [drawLandmarks]);

  useEffect(() => {
    connectWebSocketRef.current = connectWebSocket;
  }, [connectWebSocket]);

  // ── Camera Control ────────────────────────────────────────────────────────

  const startCamera = async () => {
    setCameraError("");
    setWsErrorMessage("");
    isManualStopRef.current = false;
    reconnectAttemptsRef.current = 0;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
          setVideoAspectRatio(videoRef.current.videoWidth / videoRef.current.videoHeight);
        }
      }

      setIsPaused(false);
      isPausedRef.current = false;
      setActiveCandidate(null);
      setIsCameraActive(true);
      isCameraActiveRef.current = true;
      connectWebSocket();
    } catch (err: unknown) {
      let msg = "Failed to access webcam.";
      if (err instanceof DOMException) {
        if (err.name === "NotAllowedError") {
          msg = "Camera permission denied. Please allow camera access in your browser settings.";
        } else if (err.name === "NotFoundError") {
          msg = "No camera found on this device.";
        } else if (err.name === "NotReadableError") {
          msg = "Camera is currently in use by another application.";
        }
      }
      setCameraError(msg);
      setIsCameraActive(false);
      isCameraActiveRef.current = false;
    }
  };

  const stopCamera = () => {
    isManualStopRef.current = true;
    isCameraActiveRef.current = false;

    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (frameTimerRef.current) {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close(1000, "User stopped camera");
      } catch {
        // Ignore
      }
      wsRef.current = null;
    }

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }

    setIsCameraActive(false);
    setIsPaused(false);
    isPausedRef.current = false;
    setActiveCandidate(null);
    setWsStatus("disconnected");
    setWsErrorMessage("");
    setSegState("IDLE");
    setHandsDetected(false);
    setInRestZone(false);
    setFps(0);
  };

  // Run throttled streaming loop (~13 FPS)
  useEffect(() => {
    if (isCameraActive && wsStatus === "connected") {
      frameTimerRef.current = window.setInterval(() => {
        sendNextFrame();
      }, 75);
    } else {
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
      }
    }

    return () => {
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
      }
    };
  }, [isCameraActive, wsStatus, sendNextFrame]);

  // Clean cleanup on component unmount
  useEffect(() => {
    return () => {
      isManualStopRef.current = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (wsRef.current) {
        try {
          wsRef.current.close(1000, "Component unmounted");
        } catch {
          // Ignore
        }
      }
    };
  }, []);

  const handleClearSequence = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "clear" }));
    }
    setCompletedSentence(null);
    setInProgressGlosses([]);
    setInProgressTranslation(null);
    setCurrentGlosses([]);
    setCurrentTranslation(null);
    setIsFinalized(false);
    setFinalizedSequence(null);
    setFinalizedTranslation(null);
    setLatestPrediction(null);
    setActiveCandidate(null);
    setInferredWords([]);
    setUncertaintyFlag(false);
    setReachedMatcher(false);
    setRejectionReason(null);
  };

  const handleManualReconnect = () => {
    reconnectAttemptsRef.current = 0;
    connectWebSocket();
  };

  return (
    <div className="webcam-recognition-container">
      {/* 2-Column Responsive Layout Grid */}
      <div className="webcam-layout-grid">
        {/* Left Column: Enlarged Camera Viewport */}
        <div className="camera-column">
          <div className="camera-card">
            <div className="camera-header">
              <div className="camera-title-group">
                <h3>Live Camera Feed</h3>
                <span className={`ws-badge ws-${wsStatus}`}>
                  {wsStatus === "connected" && `● Connected (${fps} FPS)`}
                  {wsStatus === "connecting" && "◌ Connecting..."}
                  {wsStatus === "disconnected" && "○ Disconnected"}
                  {wsStatus === "error" && "⚠ Connection Error"}
                </span>

                {isCameraActive && (
                  <span className={`rec-status-badge ${isPaused ? "badge-paused" : "badge-listening"}`}>
                    {isPaused ? "⏸ Recognition Paused" : "● Recognition Listening"}
                  </span>
                )}

                <button
                  type="button"
                  className={`btn-hotkeys-toggle ${isHotkeysEnabled ? "hotkeys-enabled" : "hotkeys-disabled"}`}
                  onClick={() => setIsHotkeysEnabled((p) => !p)}
                  title="Click or press 'P' on keyboard to enable/disable hotkeys"
                >
                  <span className="hotkey-dot">{isHotkeysEnabled ? "●" : "○"}</span>
                  Hotkeys [P]: {isHotkeysEnabled ? "Enabled" : "Disabled"}
                </button>

                {galleryStatus && (
                  <span className={`gallery-badge ${galleryStatus.loaded ? "gallery-loaded" : "gallery-error"}`}>
                    {galleryStatus.loaded
                      ? `Gallery: ${galleryStatus.count} classes`
                      : "Gallery: Not loaded"}
                  </span>
                )}
              </div>

              <div className="camera-actions">
                {!isCameraActive ? (
                  <button
                    type="button"
                    className="btn btn-primary btn-camera"
                    onClick={startCamera}
                  >
                    <span className="btn-icon-symbol">▶</span> Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      className={`btn btn-secondary ${isPaused ? "btn-resume" : "btn-pause"}`}
                      onClick={togglePause}
                      title="Spacebar starts/stops sign accumulation without interrupting video preview"
                    >
                      <span className="btn-icon-symbol">{isPaused ? "▶" : "⏸"}</span>
                      {isPaused ? "Resume (Space)" : "Pause (Space)"}
                    </button>

                    <button
                      type="button"
                      className="btn btn-secondary btn-undo-sign"
                      onClick={handleUndoLastSign}
                      disabled={currentGlosses.length === 0}
                      title="Backspace removes the last accepted sign from sequence"
                    >
                      <span className="btn-icon-symbol">⌫</span> Undo (Backspace)
                    </button>

                    <button
                      type="button"
                      className="btn btn-danger btn-camera"
                      onClick={stopCamera}
                    >
                      <span className="btn-icon-symbol">■</span> Stop Camera
                    </button>
                  </>
                )}

                <button
                  type="button"
                  className="btn btn-secondary btn-clear-sentence"
                  onClick={handleClearSequence}
                  disabled={
                    !completedSentence &&
                    inProgressGlosses.length === 0 &&
                    currentGlosses.length === 0 &&
                    !currentTranslation &&
                    !finalizedTranslation
                  }
                  title="Clear sentence (reset persistent sentence, active sequence, and translation)"
                >
                  Clear Sentence
                </button>
              </div>
            </div>

            {/* Connection Alert */}
            {wsErrorMessage && (
              <div className="camera-error-banner ws-error-banner">
                <div className="error-text-content">
                  <strong>Connection Alert:</strong> {wsErrorMessage}
                </div>
                {isCameraActive && (
                  <button
                    type="button"
                    className="btn-retry-conn"
                    onClick={handleManualReconnect}
                  >
                    Retry Connection
                  </button>
                )}
              </div>
            )}

            {cameraError && (
              <div className="camera-error-banner">
                <span>{cameraError}</span>
              </div>
            )}

            {/* Enlarged Viewport (~68vh, object-fit: contain, 1:1 aligned mirrored canvas) */}
            <div className="video-viewport">
              <div
                className="video-stage"
                style={{ aspectRatio: `${videoAspectRatio}` }}
              >
                <div className="video-mirror-wrapper">
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    onLoadedMetadata={handleVideoMetadata}
                    onPlay={handleVideoMetadata}
                    className="video-feed"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="landmark-overlay-canvas"
                  />
                </div>
              </div>

              {!isCameraActive && (
                <div className="camera-placeholder">
                  <div className="placeholder-icon">📷</div>
                  <h4>Camera is inactive</h4>
                  <p>Click &ldquo;Start Camera&rdquo; to begin real-time ISL sentence recognition.</p>
                  <div className="camera-tips">
                    <span className="tip-item">💡 Spacebar pauses/resumes sign accumulation instantly</span>
                    <span className="tip-item">💡 Backspace undoes the last sign in active sequence</span>
                    <span className="tip-item">💡 Press 'P' to enable/disable hotkeys at any time</span>
                  </div>
                </div>
              )}

              {/* Real-time State Badges Overlay */}
              {isCameraActive && (
                <div className="hud-overlay">
                  <div className="hud-left">
                    {isPaused ? (
                      <span className="seg-badge seg-paused">
                        ⏸ Recognition Paused (Space to Resume)
                      </span>
                    ) : (
                      <span className={`seg-badge seg-${segState.toLowerCase()}`}>
                        {segState === "IDLE" && (isFinalized ? "Sentence finalized" : "Waiting for sign...")}
                        {segState === "ACTIVE" && `Signing (${gestureFrames}f)`}
                        {segState === "COOLDOWN" && "Analyzing gesture..."}
                      </span>
                    )}

                    <span className={`hands-badge ${handsDetected ? (inRestZone ? "rest-zone" : "detected") : "none"}`}>
                      {handsDetected
                        ? (inRestZone ? "⬇ Wrists resting (chest level)" : "✋ Signing zone")
                        : "Rest (Hands down)"}
                    </span>

                    {sessionFrameCount > 0 && (
                      <span className="frame-counter-badge">
                        Frame #{sessionFrameCount}
                      </span>
                    )}
                  </div>

                  <div className="hud-right">
                    {/* Real-time in-flight candidate tracking during gesture */}
                    {activeCandidate && !isPaused && segState === "ACTIVE" && (
                      <div className="hud-active-candidate-bar">
                        <div className="candidate-info">
                          <span className="candidate-pulse">●</span>
                          <span className="candidate-label">Matching: <strong>{activeCandidate.label}</strong></span>
                          <span className="candidate-pct">{Math.round(activeCandidate.similarity * 100)}%</span>
                        </div>
                        <div className="candidate-progress-track">
                          <div
                            className="candidate-progress-fill"
                            style={{ width: `${Math.min(100, Math.round(activeCandidate.similarity * 100))}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {latestPrediction && (!activeCandidate || segState !== "ACTIVE") && (
                      <div className="live-sign-tag">
                        <span className="sign-label">{latestPrediction.label}</span>
                        <span className="sign-conf">
                          {Math.round(latestPrediction.confidence * 100)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Sticky Recognition Panel & Live Sentence Outputs */}
        <div className="recognition-column">
          {/* Accumulated Sentence Card */}
          <div className="result-panel sequence-card">
            <div className="panel-header">
              <div className="panel-title-group">
                <h4>Sentence & Sign Sequences</h4>
                {completedSentence && <span className="badge-persisted-pill">1 Persisted Sentence</span>}
              </div>
              {isFinalized && <span className="finalized-tag">Idle Finalized</span>}
            </div>

            {/* 1. Persistent Completed Sentence Display */}
            {((completedSentence && completedSentence.glosses && completedSentence.glosses.length > 0) || (finalizedSequence && finalizedSequence.length > 0)) && (
              <div className="completed-persistent-block">
                <div className="section-sub-header">
                  <span className="box-label">Completed Sentence</span>
                  <span className="badge-persisted-pill">Persisted 🔒</span>
                </div>

                <div className="gloss-sequence-chips completed-chips">
                  {(completedSentence?.glosses || finalizedSequence || []).map((gloss, idx) => (
                    <React.Fragment key={`comp-${gloss}-${idx}`}>
                      <span className="gloss-chip completed-chip">
                        {gloss}
                      </span>
                      {idx < (completedSentence?.glosses || finalizedSequence || []).length - 1 && <span className="arrow-sep">→</span>}
                    </React.Fragment>
                  ))}
                </div>

                <blockquote className="english-sentence completed-sentence">
                  &ldquo;{completedSentence?.translation || finalizedTranslation || "..."}&rdquo;
                </blockquote>
                <div className="persisted-hint">
                  Permanently saved until <strong>Clear Sentence</strong> is pressed.
                </div>
              </div>
            )}

            {/* 2. In-Progress Active Sequence Display */}
            <div className="in-progress-block">
              <div className="section-sub-header">
                <span className="box-label">In-Progress Active Sequence</span>
                <span className="badge-in-progress-pill">
                  {inProgressGlosses.length > 0 ? `${inProgressGlosses.length} tokens` : "Ready"}
                </span>
              </div>

              {inProgressGlosses.length === 0 ? (
                <p className="empty-hint">
                  {completedSentence
                    ? "Start signing your next sentence or phrase. New signs will accumulate here in real-time."
                    : "Perform signs sequentially in front of the camera. Each gesture accumulates here in chronological order. Press Space to pause/resume."}
                </p>
              ) : (
                <>
                  <div className="gloss-sequence-chips active-chips">
                    {inProgressGlosses.map((gloss, idx) => (
                      <React.Fragment key={`inp-${gloss}-${idx}`}>
                        <span className={`gloss-chip active-chip ${idx === inProgressGlosses.length - 1 ? "latest-chip" : ""}`}>
                          {gloss}
                        </span>
                        {idx < inProgressGlosses.length - 1 && <span className="arrow-sep">→</span>}
                      </React.Fragment>
                    ))}
                  </div>

                  {inProgressTranslation && (
                    <div className="interim-translation-box">
                      <span className="interim-label">Interim Translation:</span>
                      <span className="interim-text">&ldquo;{inProgressTranslation}&rdquo;</span>
                    </div>
                  )}

                  <div className="active-controls-hint">
                    Press <kbd>Backspace</kbd> to undo last token &bull; Press <kbd>Space</kbd> to pause
                  </div>
                </>
              )}
            </div>

            {/* Contextually Inferred Words Pill Display */}
            {inferredWords && inferredWords.length > 0 && (
              <div className="inferred-words-section">
                <span className="inferred-heading">Contextually Inferred Words:</span>
                <div className="inferred-chips">
                  {inferredWords.map((word, wIdx) => (
                    <span key={wIdx} className="inferred-chip">
                      +{word}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {uncertaintyFlag && (
              <div className="uncertainty-notice">
                ⚠️ Low gesture confidence detected; sentence inferred contextually.
              </div>
            )}
          </div>

          {/* Candidates & Gesture Diagnostics Card */}
          <div className="result-panel diagnostics-card">
            <div className="panel-header">
              <h4>Recognition Status & Candidates</h4>
            </div>

            {latestPrediction && latestPrediction.candidates && latestPrediction.candidates.length > 0 ? (
              <div className="candidates-list">
                <span className="candidates-title">Top Candidates for Last Sign:</span>
                <div className="candidate-tags">
                  {latestPrediction.candidates.map((c, i) => (
                    <span key={i} className="candidate-tag">
                      {c.label}: <strong>{Math.round(c.similarity * 100)}%</strong>
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <p className="empty-hint">Perform a sign to see top candidate similarities.</p>
            )}

            <div className="diagnostics-metrics">
              <div className="metric-row">
                <span className="metric-name">Recognition Mode:</span>
                <span className={`metric-val ${isPaused ? "val-paused" : "val-listening"}`}>
                  {isPaused ? "Paused (Space to Listen)" : "Listening (Space to Pause)"}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Shortcut Controls:</span>
                <span className="metric-val">
                  {isHotkeysEnabled ? "Active [Space: Pause/Resume, Backspace: Undo, P: Toggle]" : "Disabled [Press P to enable]"}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Active Candidate:</span>
                <span className="metric-val">
                  {activeCandidate ? `${activeCandidate.label} (${Math.round(activeCandidate.similarity * 100)}%)` : "None"}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Pipeline Latency:</span>
                <span className="metric-val">{latencyMs > 0 ? `${latencyMs.toFixed(1)} ms` : "—"}</span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Missed Hands Count:</span>
                <span className="metric-val">{missedHandsCount}</span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Rejected Segments:</span>
                <span className="metric-val">{rejectedSegmentsCount}</span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Wrist Motion Energy:</span>
                <span className="metric-val">{motionEnergy.toFixed(3)}</span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Hand Source:</span>
                <span className="metric-val">
                  {handSource === "hand_landmarks"
                    ? "MediaPipe Hands (21 pts)"
                    : handSource === "pose_wrists_only"
                    ? "Pose Wrists Articulation"
                    : "None"}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Posture:</span>
                <span className="metric-val">
                  {inRestZone ? "Rest zone (stationary)" : (handsDetected ? "Active signing zone" : "Hands lowered")}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-name">Matcher / Rejection:</span>
                <span className="metric-val">
                  {rejectionReason
                    ? `Rejected (${rejectionReason})`
                    : (reachedMatcher ? "DTW Matched" : "Monitoring")}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

