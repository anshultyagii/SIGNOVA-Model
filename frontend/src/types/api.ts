// SignSpeakAI Frontend - API Types matching src/api/main.py

export interface HealthResponse {
  status: string;
  supported_classes_count: number;
  version: string;
  recognition_gallery_loaded?: boolean;
  enrolled_gallery_classes?: number;
  translator_mode?: string;
  model_loaded?: boolean;
}

export interface TranslateRequest {
  glosses: string[];
}

export interface TranslateResponse {
  normalized_glosses: string[];
  translation: string;
  is_fallback: boolean;
  matched_rule?: string | null;
  model_mode?: string;
  inferred_words?: string[];
  uncertainty_flag?: boolean;
}

export type ConnectionStatus = "connected" | "disconnected" | "checking";

// ── WebSocket Stream Types ──────────────────────────────────────────────────

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface UILandmarks {
  left_hand?: LandmarkPoint[] | null;
  right_hand?: LandmarkPoint[] | null;
  pose?: LandmarkPoint[] | null;
  has_pose: boolean;
  has_face: boolean;
}

export interface PredictionCandidate {
  label: string;
  similarity: number;
}

export interface LivePrediction {
  label: string;
  confidence: number;
  distance: number;
  added_to_buffer: boolean;
  candidates: PredictionCandidate[];
}

export interface ActiveCandidate {
  label: string;
  similarity: number;
  distance: number;
  gesture_frames: number;
}

export interface SessionDiagnostics {
  frame_count: number;
  is_paused?: boolean;
  has_pose: boolean;
  has_face: boolean;
  has_lh: boolean;
  has_rh: boolean;
  hand_source?: string;
  reached_matcher?: boolean;
  rejection_reason?: string | null;
  in_rest_zone?: boolean;
  motion_energy?: number;
  onset_counter: number;
  offset_counter: number;
  pause_counter?: number;
  rest_counter?: number;
  gesture_frames: number;
  latency_ms?: number;
  missed_hands_count?: number;
  rejected_segments_count?: number;
  active_candidate?: ActiveCandidate | null;
}

export interface FrameResult {
  type: "frame_result";
  frame_count?: number;
  state: "IDLE" | "ACTIVE" | "COOLDOWN" | "PAUSED" | string;
  is_paused?: boolean;
  hands_detected: boolean;
  hand_source?: string;
  reached_matcher?: boolean;
  rejection_reason?: string | null;
  in_rest_zone?: boolean;
  motion_energy?: number;
  gesture_frames: number;
  landmarks: UILandmarks;
  diagnostics?: SessionDiagnostics;
  new_prediction?: LivePrediction | null;
  active_candidate?: ActiveCandidate | null;
  latest_prediction?: LivePrediction | null;
  current_glosses: string[];
  current_translation?: string | null;
  is_finalized?: boolean;
  completed_sequence?: string[];
  completed_translation?: string | null;
  in_progress_sequence?: string[];
  in_progress_translation?: string | null;
  finalized_sequence?: string[] | null;
  finalized_translation?: string | null;
}

export interface InitAckMessage {
  type: "init_ack";
  status: string;
  gallery_loaded: boolean;
  enrolled_classes: number;
}

export interface ResetAckMessage {
  type: "reset_ack";
  current_glosses: string[];
  current_translation?: string | null;
  completed_sequence?: string[];
  completed_translation?: string | null;
  in_progress_sequence?: string[];
  in_progress_translation?: string | null;
  is_finalized?: boolean;
  state: string;
}

export interface PauseAckMessage {
  type: "pause_ack";
  is_paused: boolean;
  state: string;
}

export interface ResumeAckMessage {
  type: "resume_ack";
  is_paused: boolean;
  state: string;
}

export interface UndoAckMessage {
  type: "undo_ack";
  removed_gloss?: string | null;
  current_glosses: string[];
  current_translation?: string | null;
  completed_sequence?: string[];
  completed_translation?: string | null;
  in_progress_sequence?: string[];
  in_progress_translation?: string | null;
  is_finalized?: boolean;
}
