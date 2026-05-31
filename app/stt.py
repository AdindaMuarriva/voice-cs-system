"""
stt.py
Speech-to-Text menggunakan whisper.cpp (subprocess) atau OpenAI Whisper Python.
Mendukung transkripsi multilingual ID-EN-AR (code-switching).
"""

import os
import subprocess
import tempfile
import json
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WHISPER_EXEC  = os.getenv("WHISPER_EXEC_PATH",  "models/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.getenv("WHISPER_MODEL_PATH",  "models/whisper.cpp/models/ggml-large-v3-turbo.bin")

# Gunakan BASE_DIR agar path portabel di Windows & Linux
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def transcribe_whisper_cpp(audio_path: str, language: str = "auto") -> dict:
    """
    Transkripsi audio menggunakan whisper.cpp melalui subprocess.
    
    Args:
        audio_path: Path ke file audio (.wav)
        language:   Kode bahasa ('id', 'en', 'ar') atau 'auto' untuk deteksi otomatis

    Returns:
        dict: {
            "text": str,
            "language": str,
            "duration_s": float,
            "processing_time_s": float,
            "segments": list  (kosong jika tidak tersedia)
        }
    """
    exec_path  = _resolve(WHISPER_EXEC)
    model_path = _resolve(WHISPER_MODEL)

    if not exec_path.exists():
        raise FileNotFoundError(
            f"Whisper executable tidak ditemukan: {exec_path}\n"
            "Pastikan sudah build whisper.cpp (lihat README)."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model Whisper tidak ditemukan: {model_path}\n"
            "Download model dengan: ./models/download-ggml-model.sh large-v3-turbo"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

    # Buat file output JSON sementara
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_json = tmp.name

    cmd = [
        str(exec_path),
        "-m", str(model_path),
        "-f", str(audio_path),
        "--output-json",
        "--output-file", output_json.replace(".json", ""),
        "--print-progress", "false",
    ]
    
    # Tambahkan language hint jika bukan auto
    if language != "auto":
        cmd += ["-l", language]
    else:
        # Biarkan Whisper deteksi sendiri - bagus untuk code-switching
        cmd += ["-l", "auto"]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(
                f"Whisper error (kode {result.returncode}):\n{result.stderr}"
            )

        # Baca output JSON
        json_file = Path(output_json + ".json") if not output_json.endswith(".json") else Path(output_json)
        # whisper.cpp menambahkan .json sendiri
        candidate = Path(output_json.replace(".json", "") + ".json")
        if candidate.exists():
            json_file = candidate

        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            
            # Gabungkan semua teks dari segmen
            segments = data.get("transcription", [])
            full_text = " ".join(seg.get("text", "").strip() for seg in segments)
            detected_lang = data.get("result", {}).get("language", language)
            
            # Bersihkan file temp
            json_file.unlink(missing_ok=True)
            
            return {
                "text": full_text.strip(),
                "language": detected_lang,
                "duration_s": _get_audio_duration(str(audio_path)),
                "processing_time_s": round(elapsed, 2),
                "segments": segments,
                "source": "whisper.cpp"
            }
        else:
            # Fallback: baca dari stdout jika JSON tidak tersedia
            text = result.stdout.strip()
            return {
                "text": text,
                "language": language,
                "duration_s": _get_audio_duration(str(audio_path)),
                "processing_time_s": round(elapsed, 2),
                "segments": [],
                "source": "whisper.cpp (stdout fallback)"
            }

    except subprocess.TimeoutExpired:
        raise TimeoutError("Whisper timeout (>120 detik). Coba model yang lebih kecil.")
    finally:
        # Pastikan file temp dihapus
        for f in [output_json, output_json + ".json",
                  output_json.replace(".json", "") + ".json"]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


def transcribe_whisper_python(audio_path: str, model_size: str = "base") -> dict:
    """
    Alternatif: Transkripsi menggunakan openai-whisper Python package.
    Digunakan sebagai fallback jika whisper.cpp belum di-build.
    
    Instalasi: pip install openai-whisper
    Model size: tiny (~39MB), base (~74MB), small (~244MB)
    Gunakan 'base' untuk keseimbangan kecepatan & akurasi di CPU.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper belum terinstall.\n"
            "Jalankan: pip install openai-whisper\n"
            "Atau gunakan whisper.cpp (lebih cepat untuk CPU)."
        )

    # Cache model agar tidak reload tiap file (hemat RAM & waktu)
    global _whisper_model_cache, _whisper_model_size_cache
    if (
        "_whisper_model_cache" not in globals()
        or _whisper_model_size_cache != model_size
    ):
        print(f"[STT] Loading Whisper model '{model_size}'...")
        _whisper_model_cache = whisper.load_model(model_size)
        _whisper_model_size_cache = model_size

    start = time.time()
    result = _whisper_model_cache.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,   # Auto-detect
        verbose=False,
        fp16=False,      # Matikan fp16 agar aman di CPU Windows
    )
    elapsed = time.time() - start

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "duration_s": _get_audio_duration(str(audio_path)),
        "processing_time_s": round(elapsed, 2),
        "segments": result.get("segments", []),
        "source": f"openai-whisper ({model_size})"
    }


def transcribe(audio_path: str, language: str = "auto", use_python_fallback: bool = True) -> dict:
    """
    Entry point utama untuk transkripsi.
    Mencoba whisper.cpp terlebih dahulu; fallback ke openai-whisper jika gagal.
    
    Args:
        audio_path: Path ke file .wav
        language:   'auto' | 'id' | 'en' | 'ar'
        use_python_fallback: Jika True, coba openai-whisper jika whisper.cpp gagal
    
    Returns:
        dict hasil transkripsi
    """
    try:
        return transcribe_whisper_cpp(audio_path, language)
    except (FileNotFoundError, RuntimeError) as e:
        if use_python_fallback:
            print(f"[STT] whisper.cpp tidak tersedia ({e}), mencoba openai-whisper...")
            return transcribe_whisper_python(audio_path)
        raise


def _get_audio_duration(audio_path: str) -> float:
    """Dapatkan durasi audio dalam detik."""
    try:
        data, sr = sf.read(audio_path)
        return round(len(data) / sr, 2)
    except Exception:
        return 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Hitung Word Error Rate (WER) antara referensi dan hipotesis.
    WER = (S + D + I) / N
    """
    try:
        from jiwer import wer
        return round(wer(reference.lower(), hypothesis.lower()), 4)
    except ImportError:
        # Implementasi manual sederhana
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Dynamic programming
        dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return round(dp[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Hitung Character Error Rate (CER).
    """
    try:
        from jiwer import cer
        return round(cer(reference.lower(), hypothesis.lower()), 4)
    except ImportError:
        ref_chars = list(reference.lower().replace(" ", ""))
        hyp_chars = list(hypothesis.lower().replace(" ", ""))
        
        dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        for i in range(len(ref_chars) + 1):
            dp[i][0] = i
        for j in range(len(hyp_chars) + 1):
            dp[0][j] = j
        
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return round(dp[len(ref_chars)][len(hyp_chars)] / max(len(ref_chars), 1), 4)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Penggunaan: python stt.py <path_audio.wav>")
        sys.exit(1)
    
    result = transcribe(sys.argv[1])
    print(f"Teks         : {result['text']}")
    print(f"Bahasa       : {result['language']}")
    print(f"Durasi audio : {result['duration_s']}s")
    print(f"Waktu proses : {result['processing_time_s']}s")
    print(f"Sumber       : {result['source']}")