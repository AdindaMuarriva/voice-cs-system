"""
main.py
FastAPI backend untuk sistem Speech-to-Speech Code-Switching.
Pipeline: Audio Input → STT → Normalisasi/Tagging → LLM → TTS → Audio Output
"""

import os
import uuid
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modul lokal
from app.stt import transcribe
from app.llm import generate_response, reset_conversation
from app.tts import synthesize_with_fallback, synthesize_multilingual
from app.utils import (
    normalize_text,
    tag_code_switching,
    detect_dominant_language,
    split_by_language,
    build_tagging_summary,
)

# ── Konfigurasi path ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
TEMP_DIR  = BASE_DIR / "temp"
LOG_DIR   = BASE_DIR / "log"
AUDIO_OUT = BASE_DIR / "log" / "tts_output"

for d in [TEMP_DIR, LOG_DIR, AUDIO_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Inisialisasi FastAPI ─────────────────────────────────────────
app = FastAPI(
    title="Voice-cs-System",
    description=(
        "Sistem pipeline STT → LLM → TTS untuk ujaran code-switching "
        "Bahasa Indonesia, Inggris, dan Arab."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ───────────────────────────────────────────────────────

def _save_log(data: dict, prefix: str = "pipeline") -> str:
    """Simpan log hasil pipeline ke folder log/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    log_path = LOG_DIR / f"{prefix}_{timestamp}_{uid}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(log_path)


def _cleanup_temp(path: str):
    """Hapus file temp setelah digunakan."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# ── Endpoint utama: /voice-chat ──────────────────────────────────

@app.post("/voice-chat", summary="Pipeline lengkap: audio → audio")
async def voice_chat(
    audio: UploadFile = File(..., description="File audio .wav (input ujaran)"),
    mode: str = Form("preserve", description="Mode: preserve | normalize | translate"),
    stt_language: str = Form("auto", description="Bahasa STT: auto | id | en | ar"),
    segment_tts: bool = Form(False, description="Pisah TTS per segmen bahasa"),
    save_log: bool = Form(True, description="Simpan log pipeline"),
):
    """
    Pipeline lengkap Speech-to-Speech:
    1. Terima audio input
    2. Transkripsi (STT)
    3. Normalisasi & language tagging
    4. Generate respons LLM
    5. Sintesis suara (TTS)
    6. Return audio output

    Gunakan mode='preserve' untuk mempertahankan code-switching,
    atau mode='normalize' untuk respons dalam satu bahasa.
    """
    start_total = time.time()
    temp_input = None

    try:
        # 1. Simpan audio upload ke temp
        uid = uuid.uuid4().hex[:8]
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"
        temp_input = str(TEMP_DIR / f"input_{uid}{suffix}")
        
        with open(temp_input, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # ── TAHAP 1: STT ──────────────────────────────────────────
        stt_start = time.time()
        stt_result = transcribe(temp_input, language=stt_language)
        stt_time = round(time.time() - stt_start, 2)
        
        transcript = stt_result["text"]
        if not transcript.strip():
            raise HTTPException(status_code=422, detail="Audio tidak dapat ditranskrip. Periksa kualitas rekaman.")

        # ── TAHAP 2: Normalisasi & Tagging ────────────────────────
        proc_start = time.time()
        normalized = normalize_text(transcript)
        tagged_tokens = tag_code_switching(transcript)
        dominant_lang = detect_dominant_language(transcript)
        tagging_summary = build_tagging_summary(tagged_tokens)
        proc_time = round(time.time() - proc_start, 2)

        # ── TAHAP 3: LLM ──────────────────────────────────────────
        llm_start = time.time()
        llm_result = generate_response(
            transcript=normalized,
            mode=mode,
            tagged_text=tagging_summary,
            keep_history=True,
        )
        llm_time = round(time.time() - llm_start, 2)

        response_text = llm_result["response_text"]

        # ── TAHAP 4: TTS ──────────────────────────────────────────
        tts_start = time.time()
        output_audio = str(AUDIO_OUT / f"response_{uid}.wav")
        
        if segment_tts:
            # Pisah per segmen bahasa untuk pelafalan lebih natural
            segments = split_by_language(response_text)
            tts_result = synthesize_multilingual(segments, output_path=output_audio)
        else:
            tts_result = synthesize_with_fallback(
                text=response_text,
                output_path=output_audio,
                lang=dominant_lang if dominant_lang != "mixed" else "id",
            )
        tts_time = round(time.time() - tts_start, 2)

        total_time = round(time.time() - start_total, 2)

        # ── Logging ───────────────────────────────────────────────
        pipeline_log = {
            "timestamp": datetime.now().isoformat(),
            "input_file": audio.filename,
            "mode": mode,
            "stt": {
                "transcript": transcript,
                "normalized": normalized,
                "detected_language": stt_result.get("language"),
                "audio_duration_s": stt_result.get("duration_s"),
                "processing_time_s": stt_time,
                "source": stt_result.get("source"),
            },
            "processing": {
                "dominant_language": dominant_lang,
                "tagging_summary": tagging_summary,
                "tagged_tokens": tagged_tokens[:20],  # Simpan max 20 token
                "processing_time_s": proc_time,
            },
            "llm": {
                "response_text": response_text,
                "model": llm_result.get("model"),
                "mode": mode,
                "processing_time_s": llm_time,
                "finish_reason": llm_result.get("finish_reason"),
            },
            "tts": {
                "output_path": tts_result.get("audio_path"),
                "duration_s": tts_result.get("duration_s"),
                "processing_time_s": tts_time,
            },
            "latency": {
                "stt_s": stt_time,
                "processing_s": proc_time,
                "llm_s": llm_time,
                "tts_s": tts_time,
                "total_s": total_time,
            }
        }
        
        if save_log:
            log_path = _save_log(pipeline_log, prefix="voice_chat")
            pipeline_log["log_file"] = log_path

        return FileResponse(
            path=output_audio,
            media_type="audio/wav",
            filename=f"response_{uid}.wav",
            headers={
                "X-Transcript": transcript[:200],
                "X-Response-Text": response_text[:200],
                "X-Total-Latency-S": str(total_time),
                "X-Mode": mode,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_input:
            _cleanup_temp(temp_input)


# ── Endpoint teks-saja ───────────────────────────────────────────

@app.post("/text-chat", summary="Pipeline teks: transkrip → respons teks")
async def text_chat(
    text: str = Form(...),
    mode: str = Form("preserve"),
):
    """
    Terima teks langsung (tanpa audio), kirim ke LLM, return respons teks.
    Berguna untuk debug/testing pipeline LLM tanpa STT/TTS.
    """
    normalized = normalize_text(text)
    tagged = tag_code_switching(text)
    summary = build_tagging_summary(tagged)
    
    llm_result = generate_response(
        transcript=normalized,
        mode=mode,
        tagged_text=summary,
    )
    
    return JSONResponse({
        "input": text,
        "normalized": normalized,
        "tagging_summary": summary,
        "response": llm_result["response_text"],
        "mode": mode,
        "model": llm_result["model"],
        "processing_time_s": llm_result["processing_time_s"],
    })


# ── Endpoint STT-saja ────────────────────────────────────────────

@app.post("/transcribe", summary="STT saja: audio → teks")
async def transcribe_only(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transkripsi audio ke teks tanpa LLM/TTS."""
    uid = uuid.uuid4().hex[:8]
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    temp_path = str(TEMP_DIR / f"stt_{uid}{suffix}")
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        result = transcribe(temp_path, language=language)
        tagged = tag_code_switching(result["text"])
        
        return JSONResponse({
            "transcript": result["text"],
            "normalized": normalize_text(result["text"]),
            "detected_language": result.get("language"),
            "dominant_language": detect_dominant_language(result["text"]),
            "tagging": tagged,
            "tagging_summary": build_tagging_summary(tagged),
            "audio_duration_s": result.get("duration_s"),
            "processing_time_s": result.get("processing_time_s"),
        })
    finally:
        _cleanup_temp(temp_path)


# ── Endpoint utilitas ────────────────────────────────────────────

@app.post("/reset-conversation", summary="Reset history percakapan")
async def reset():
    reset_conversation()
    return {"status": "ok", "message": "History percakapan direset."}


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/", summary="Info sistem")
async def root():
    return {
        "system": "Code-Switching Speech-to-Speech",
        "version": "1.0.0",
        "endpoints": {
            "POST /voice-chat": "Pipeline penuh: audio → audio",
            "POST /text-chat": "Teks → respons LLM",
            "POST /transcribe": "Audio → transkripsi",
            "POST /reset-conversation": "Reset history",
            "GET /health": "Status sistem",
        }
    }


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
