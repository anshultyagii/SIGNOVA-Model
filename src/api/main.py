# -*- coding: utf-8 -*-
"""
SignSpeak AI - FastAPI Backend Application
==========================================
Exposes REST endpoints for ISL-to-English translation and system health,
plus WebSocket endpoint for real-time webcam sign recognition.
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.translation.isl_to_english import (
    ALL_25_CLASSES,
    ISLToEnglishTranslator,
    TranslationResult,
)
from src.translation.model_translator import ConfigurableISLTranslator
from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.stream_adapter import LiveWebcamRecognitionSession

# Root path and default exemplar gallery directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GALLERY_DIR = PROJECT_ROOT / "dataset" / "processed_corrected"

# Maximum sequence length permitted in a single translation request
MAX_GLOSS_SEQUENCE_LEN = 50

# Allowed CORS origins for local development frontend
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app = FastAPI(
    title="SignSpeakAI API",
    description="Backend API for SignSpeakAI - ISL to English Translation & Real-Time Recognition",
    version="0.2.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instances
_translator = ConfigurableISLTranslator()
_engine: Optional[ExemplarRecognitionEngine] = None


def get_recognition_engine() -> ExemplarRecognitionEngine:
    """Lazily instantiates and returns the shared ExemplarRecognitionEngine."""
    global _engine
    if _engine is None:
        if DEFAULT_GALLERY_DIR.exists():
            _engine = ExemplarRecognitionEngine(gallery_dir=DEFAULT_GALLERY_DIR)
        else:
            _engine = ExemplarRecognitionEngine()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field("healthy", description="Application status")
    supported_classes_count: int = Field(..., description="Number of enrolled ISL vocabulary classes")
    version: str = Field("0.2.0", description="API version")
    recognition_gallery_loaded: bool = Field(False, description="Whether DTW exemplar gallery is loaded")
    enrolled_gallery_classes: int = Field(0, description="Count of loaded gallery templates")
    translator_mode: str = Field("model", description="Translation engine mode: 'model' or 'rules'")
    model_loaded: bool = Field(False, description="Whether fine-tuned neural Seq2Seq model is loaded")


class TranslateRequest(BaseModel):
    """Translation request payload."""
    glosses: List[str] = Field(
        ...,
        description="Ordered list of sign glosses to translate.",
        examples=[["I_Me", "Want_Desire_Wish_For", "Water"]]
    )
    confidences: Optional[List[float]] = Field(
        None,
        description="Optional recognition confidence scores corresponding to glosses."
    )

    @field_validator("glosses")
    @classmethod
    def validate_length(cls, v: List[str]) -> List[str]:
        if len(v) > MAX_GLOSS_SEQUENCE_LEN:
            raise ValueError(
                f"Sequence length ({len(v)}) exceeds maximum limit of {MAX_GLOSS_SEQUENCE_LEN} glosses."
            )
        return v


class TranslateResponse(BaseModel):
    """Translation response payload."""
    normalized_glosses: List[str] = Field(..., description="List of cleaned and canonicalized glosses")
    translation: str = Field(..., description="Translated English sentence")
    is_fallback: bool = Field(..., description="True if no template matched and safe fallback was used")
    matched_rule: Optional[str] = Field(None, description="Name of the matched rule template, if any")
    model_mode: Optional[str] = Field("model", description="Translation engine mode used")
    inferred_words: Optional[List[str]] = Field(default_factory=list, description="Words contextually inferred or inserted")
    uncertainty_flag: Optional[bool] = Field(False, description="True if inference involves low confidence or uncertain fallback")


class TranslatorModeRequest(BaseModel):
    """Request payload to switch translation engine mode."""
    mode: str = Field(..., description="Translation mode: 'model' or 'rules'")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health():
    """Returns application status and number of supported ISL classes."""
    engine = get_recognition_engine()
    loaded_classes = len(engine.get_gallery_labels()) if engine.gallery_loaded else 0
    return HealthResponse(
        status="healthy",
        supported_classes_count=len(ALL_25_CLASSES),
        version="0.2.0",
        recognition_gallery_loaded=engine.gallery_loaded,
        enrolled_gallery_classes=loaded_classes,
        translator_mode=getattr(_translator, "mode", "rules"),
        model_loaded=getattr(_translator, "is_model_loaded", False),
    )


@app.get("/translator/mode", tags=["Translation"])
def get_translator_mode():
    """Returns current translator mode ('model' or 'rules') and model loaded status."""
    return {
        "mode": getattr(_translator, "mode", "rules"),
        "model_loaded": getattr(_translator, "is_model_loaded", False),
    }


@app.post("/translator/mode", tags=["Translation"])
def set_translator_mode(request: TranslatorModeRequest):
    """Dynamically switches translator mode between 'model' and 'rules'."""
    try:
        _translator.set_mode(request.mode)
        return {
            "status": "success",
            "mode": _translator.mode,
            "model_loaded": getattr(_translator, "is_model_loaded", False),
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
def post_translate(request: TranslateRequest):
    """
    Translates an ordered gloss sequence into a clean English sentence.
    Handles empty sequences safely and applies documented fallbacks for unknown combinations.
    """
    if len(request.glosses) > MAX_GLOSS_SEQUENCE_LEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sequence exceeds maximum allowed length of {MAX_GLOSS_SEQUENCE_LEN} glosses."
        )

    res: TranslationResult = _translator.translate_detailed(
        request.glosses,
        confidences=request.confidences
    )

    return TranslateResponse(
        normalized_glosses=res.normalized_glosses or [],
        translation=res.text or "",
        is_fallback=res.is_fallback,
        matched_rule=res.matched_rule,
        model_mode=getattr(_translator, "mode", "rules"),
        inferred_words=res.inferred_words or [],
        uncertainty_flag=res.uncertainty_flag,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Recognition Stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    Real-time sign language recognition stream over WebSocket.
    Accepts camera frames (JSON with base64 data URL or raw binary bytes),
    processes landmarks and gesture segmentation, runs DTW matching,
    accumulates ordered glosses, and streams finalized English translations.
    """
    await websocket.accept()
    engine = get_recognition_engine()
    session = LiveWebcamRecognitionSession(engine=engine)

    # Immediately acknowledge connection with server readiness and gallery status
    await websocket.send_json({
        "type": "init_ack",
        "status": "ready",
        "gallery_loaded": engine.gallery_loaded,
        "enrolled_classes": len(engine.get_gallery_labels()) if engine.gallery_loaded else 0,
    })

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            frame_bgr: Optional[np.ndarray] = None

            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                    continue

                msg_type = payload.get("type", "frame")

                if msg_type in ("clear", "reset"):
                    session.reset()
                    await websocket.send_json({
                        "type": "reset_ack",
                        "current_glosses": [],
                        "current_translation": None,
                        "completed_sequence": [],
                        "completed_translation": None,
                        "in_progress_sequence": [],
                        "in_progress_translation": None,
                        "is_finalized": False,
                        "state": "IDLE"
                    })
                    continue

                if msg_type == "pause":
                    session.set_paused(True)
                    await websocket.send_json({
                        "type": "pause_ack",
                        "is_paused": True,
                        "state": "PAUSED"
                    })
                    continue

                if msg_type == "resume":
                    session.set_paused(False)
                    await websocket.send_json({
                        "type": "resume_ack",
                        "is_paused": False,
                        "state": session.seg_state
                    })
                    continue

                if msg_type in ("undo", "backspace"):
                    res = session.pop_last_gloss()
                    await websocket.send_json({
                        "type": "undo_ack",
                        **res
                    })
                    continue

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type == "status":
                    await websocket.send_json({
                        "type": "status_response",
                        "gallery_loaded": engine.gallery_loaded,
                        "enrolled_classes": len(engine.get_gallery_labels()) if engine.gallery_loaded else 0,
                        "session_frame_count": session.frame_count,
                        "state": session.seg_state,
                    })
                    continue

                if msg_type == "frame":
                    data_str = payload.get("data")
                    if not data_str:
                        await websocket.send_json({"type": "error", "message": "Missing frame data"})
                        continue
                    try:
                        if "," in data_str:
                            data_str = data_str.split(",", 1)[1]
                        img_bytes = base64.b64decode(data_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except Exception as err:
                        await websocket.send_json({"type": "error", "message": f"Base64 decode failed: {str(err)}"})
                        continue

            elif "bytes" in message:
                try:
                    np_arr = np.frombuffer(message["bytes"], np.uint8)
                    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as err:
                    await websocket.send_json({"type": "error", "message": f"Image bytes decode failed: {str(err)}"})
                    continue

            if frame_bgr is None:
                await websocket.send_json({"type": "error", "message": "Failed to decode camera frame"})
                continue

            # Process frame in threadpool to avoid blocking event loop
            try:
                result = await asyncio.to_thread(session.process_frame, frame_bgr)
                await websocket.send_json(result)
            except Exception as proc_err:
                print(f"[WebSocket /ws/recognize] Frame processing error: {proc_err}")
                await websocket.send_json({"type": "error", "message": f"Processing error: {str(proc_err)}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WebSocket /ws/recognize] Connection closed: {exc}")
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
