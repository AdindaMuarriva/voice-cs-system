"""
tts.py
Text-to-Speech menggunakan Coqui TTS dengan model VITS Bahasa Indonesia.
Mendukung pemrosesan per segmen untuk teks code-switching.
"""

import os
import time
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR         = Path(__file__).resolve().parent.parent
TTS_MODEL_PATH   = os.getenv("TTS_MODEL_PATH",   "app/coqui_tts/config.json")
TTS_CKPT_PATH    = os.getenv("TTS_CHECKPOINT_PATH", "app/coqui_tts/checkpoint_100000.pth")
TTS_SPEAKERS     = os.getenv("TTS_SPEAKERS_PATH",   "app/coqui_tts/speakers.pth")
TEMP_DIR         = BASE_DIR / "temp"
OUTPUT_AUDIO_DIR = BASE_DIR / "log" / "tts_output"


def _ensure_dirs():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def _get_tts_model(lang: str = "id"):
    """
    Muat model Coqui TTS yang sesuai.
    Untuk Bahasa Indonesia gunakan model lokal.
    Untuk Inggris dan Arab gunakan model default Coqui.
    """
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError(
            "Coqui TTS belum terinstall.\n"
            "Jalankan: pip install coqui-tts\n"
            "Kemudian: pip install transformers==5.0.0"
        )
    
    config_path = _resolve(TTS_MODEL_PATH)
    ckpt_path   = _resolve(TTS_CKPT_PATH)
    
    if lang == "id" and config_path.exists() and ckpt_path.exists():
        # Gunakan model lokal Indonesia
        tts = TTS(
            model_path=str(ckpt_path),
            config_path=str(config_path),
        )
    elif lang == "ar":
        # Model Arab - gunakan default Coqui (multi-lingual)
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception:
            # Fallback ke model sederhana
            tts = TTS("tts_models/en/ljspeech/vits")
    else:
        # Default: model Inggris atau multilingual
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception:
            tts = TTS("tts_models/en/ljspeech/vits")
    
    return tts


def synthesize(
    text: str,
    output_path: str | None = None,
    lang: str = "id",
    speaker: str | None = None,
) -> dict:
    """
    Sintesis teks ke audio.
    
    Args:
        text:        Teks yang akan disintesis
        output_path: Path output .wav (jika None, simpan ke temp)
        lang:        Bahasa: 'id' | 'en' | 'ar'
        speaker:     Speaker ID (opsional, untuk model multi-speaker)
    
    Returns:
        dict: {
            "audio_path": str,
            "duration_s": float,
            "processing_time_s": float,
            "text": str,
            "lang": str
        }
    """
    _ensure_dirs()
    
    if not text.strip():
        raise ValueError("Teks tidak boleh kosong untuk sintesis.")
    
    if output_path is None:
        uid = uuid.uuid4().hex[:8]
        output_path = str(TEMP_DIR / f"tts_{uid}.wav")
    
    start = time.time()
    
    tts = _get_tts_model(lang)
    
    # Sesuaikan argumen berdasarkan model
    tts_kwargs = {
        "text": text,
        "file_path": output_path,
    }
    
    # Model multi-lingual membutuhkan parameter language & speaker
    model_name = getattr(tts, "model_name", "")
    if "multilingual" in str(model_name) or "xtts" in str(model_name):
        tts_kwargs["language"] = lang
        if speaker:
            tts_kwargs["speaker"] = speaker
    
    tts.tts_to_file(**tts_kwargs)
    
    elapsed = time.time() - start
    duration = _get_audio_duration(output_path)
    
    return {
        "audio_path": output_path,
        "duration_s": duration,
        "processing_time_s": round(elapsed, 2),
        "text": text,
        "lang": lang
    }


def synthesize_multilingual(
    segments: list[dict],
    output_path: str | None = None,
) -> dict:
    """
    Sintesis teks code-switching dengan pemrosesan per segmen bahasa.
    Tiap segmen diproses dengan model yang sesuai, lalu digabungkan.
    
    Args:
        segments: List of {"text": str, "lang": str}
        output_path: Path file output final
    
    Returns:
        dict dengan informasi output audio gabungan
    """
    import soundfile as sf
    import numpy as np
    
    _ensure_dirs()
    
    if not segments:
        raise ValueError("Segments tidak boleh kosong.")
    
    if output_path is None:
        uid = uuid.uuid4().hex[:8]
        output_path = str(OUTPUT_AUDIO_DIR / f"tts_multi_{uid}.wav")
    
    start = time.time()
    segment_files = []
    temp_files = []
    
    try:
        for i, seg in enumerate(segments):
            if not seg["text"].strip():
                continue
            
            tmp_path = str(TEMP_DIR / f"seg_{i}_{uuid.uuid4().hex[:6]}.wav")
            result = synthesize(
                text=seg["text"],
                output_path=tmp_path,
                lang=seg.get("lang", "id"),
            )
            segment_files.append(tmp_path)
            temp_files.append(tmp_path)
        
        if not segment_files:
            raise RuntimeError("Tidak ada segmen yang berhasil disintesis.")
        
        # Gabungkan semua audio
        combined_audio = []
        sample_rate = 22050  # Default; akan diambil dari file pertama
        silence_samples = int(0.3 * sample_rate)  # 300ms jeda antar segmen
        
        for path in segment_files:
            data, sr = sf.read(path)
            sample_rate = sr
            combined_audio.append(data)
            combined_audio.append(np.zeros(silence_samples, dtype=data.dtype))
        
        final_audio = np.concatenate(combined_audio)
        sf.write(output_path, final_audio, sample_rate)
        
    finally:
        # Bersihkan file temp
        for f in temp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
    
    elapsed = time.time() - start
    duration = _get_audio_duration(output_path)
    
    return {
        "audio_path": output_path,
        "duration_s": duration,
        "processing_time_s": round(elapsed, 2),
        "segments_count": len(segment_files),
        "lang": "mixed"
    }


def synthesize_with_fallback(text: str, output_path: str | None = None, lang: str = "id") -> dict:
    """
    Coba sintesis dengan Coqui TTS; jika gagal, buat audio dummy.
    Berguna untuk pengembangan ketika model belum tersedia.
    """
    try:
        return synthesize(text, output_path, lang)
    except Exception as e:
        print(f"[TTS] Peringatan: Coqui TTS gagal ({e}). Menggunakan gTTS sebagai fallback...")
        return _gtts_fallback(text, output_path, lang)


def _gtts_fallback(text: str, output_path: str | None, lang: str = "id") -> dict:
    """
    Fallback menggunakan gTTS (Google TTS, membutuhkan internet).
    Hanya digunakan jika Coqui TTS tidak tersedia.
    """
    try:
        from gtts import gTTS
        import io
    except ImportError:
        raise ImportError("gTTS juga tidak tersedia. Jalankan: pip install gtts")
    
    _ensure_dirs()
    if output_path is None:
        uid = uuid.uuid4().hex[:8]
        output_path = str(TEMP_DIR / f"gtts_{uid}.wav")
    
    # gTTS menghasilkan mp3, konversi ke wav
    mp3_path = output_path.replace(".wav", ".mp3")
    gtts_lang = {"id": "id", "en": "en", "ar": "ar"}.get(lang, "id")
    
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(mp3_path)
    
    # Konversi mp3 → wav menggunakan soundfile + numpy jika ffmpeg tersedia
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", mp3_path, output_path, "-y"],
            capture_output=True, check=True
        )
        Path(mp3_path).unlink(missing_ok=True)
    except Exception:
        # Jika tidak ada ffmpeg, simpan tetap sebagai .mp3
        output_path = mp3_path
    
    duration = _get_audio_duration(output_path)
    
    return {
        "audio_path": output_path,
        "duration_s": duration,
        "processing_time_s": 0.0,
        "text": text,
        "lang": lang,
        "source": "gTTS (fallback)"
    }


def _get_audio_duration(audio_path: str) -> float:
    """Dapatkan durasi audio dalam detik."""
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
        return round(len(data) / sr, 2)
    except Exception:
        return 0.0


if __name__ == "__main__":
    print("Test TTS dengan teks Indonesia...")
    result = synthesize_with_fallback(
        text="Halo, selamat datang di sistem percakapan code-switching.",
        lang="id"
    )
    print(f"Output: {result['audio_path']}")
    print(f"Durasi: {result['duration_s']}s")
    print(f"Waktu proses: {result['processing_time_s']}s")
